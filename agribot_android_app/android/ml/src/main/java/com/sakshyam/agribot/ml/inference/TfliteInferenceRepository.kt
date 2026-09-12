package com.sakshyam.agribot.ml.inference

import android.content.Context
import com.sakshyam.agribot.domain.logic.ModelBundleReadinessChecker
import com.sakshyam.agribot.domain.model.AGRIBOT_LABEL_ORDER
import com.sakshyam.agribot.domain.model.AnalysisFrame
import com.sakshyam.agribot.domain.model.FramePrediction
import com.sakshyam.agribot.domain.model.FrontOverviewCandidate
import com.sakshyam.agribot.domain.model.ModelBundleReadiness
import com.sakshyam.agribot.domain.model.ModelEntry
import com.sakshyam.agribot.domain.model.ModelManifest
import com.sakshyam.agribot.domain.model.ScanConstants
import com.sakshyam.agribot.domain.repository.InferenceRepository
import com.sakshyam.agribot.ml.di.InferenceDispatcher
import com.sakshyam.agribot.ml.preprocessing.RgbImage
import dagger.hilt.android.qualifiers.ApplicationContext
import java.io.IOException
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.ExperimentalCoroutinesApi
import org.json.JSONArray
import org.json.JSONObject

@Singleton
@OptIn(ExperimentalCoroutinesApi::class)
class TfliteInferenceRepository @Inject constructor(
    @ApplicationContext private val context: Context,
    @InferenceDispatcher private val inferenceDispatcher: CoroutineDispatcher =
        Dispatchers.Default.limitedParallelism(1),
) : InferenceRepository {
    /**
     * Model inspection includes checksum verification and a synthetic TFLite
     * inference. It must never run while Compose/Hilt is constructing the
     * first screen: doing that made the permission dialog appear frozen on
     * slower field devices. Keep the public flows immediately available and
     * complete them from a background dispatcher instead.
     */
    private val repositoryScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val manifestReady = CompletableDeferred<ModelManifest?>()
    private val manifestState = MutableStateFlow<ModelManifest?>(null)
    private val readinessState = MutableStateFlow(
        ModelBundleReadiness(
            isReady = false,
            statusText = "Checking on-device model…",
            errors = emptyList(),
            progress = 0f,
        ),
    )
    private var classifier: TFLiteClassifier? = null
    private var detector: TFLiteDetector? = null
    @Volatile private var confidenceThresholdOverride: Float? = null
    @Volatile private var requestedCpuThreads = ScanConstants.DEFAULT_CPU_THREADS
    private var activeCpuThreads = ScanConstants.DEFAULT_CPU_THREADS

    override fun setCpuThreads(threads: Int) {
        requestedCpuThreads = threads.coerceIn(1, 4)
    }

    // Called only on the serialized inference dispatcher, never from the UI.
    private fun applyRuntimeSettings() {
        val requested = requestedCpuThreads
        if (requested == activeCpuThreads) return
        classifier?.close()
        detector?.close()
        classifier = null
        detector = null
        activeCpuThreads = requested
    }

    override fun setConfidenceThreshold(threshold: Float) {
        confidenceThresholdOverride = threshold.coerceIn(0.1f, 0.95f)
    }

    init {
        repositoryScope.launch {
            reportReadinessProgress(0.1f, "Reading model manifest...")
            val manifest = loadManifest()
            manifestState.value = manifest
            manifestReady.complete(manifest)
            readinessState.value = loadReadiness(manifest)
        }
    }

    override suspend fun classifySideFrame(frame: AnalysisFrame): FramePrediction {
        return withContext(inferenceDispatcher) {
            applyRuntimeSettings()
            val manifest = manifestReady.await()
                ?: throw ModelUnavailableException("model_manifest.json missing")
            val tflite = classifierFor(manifest)
            val image = RgbImage(frame.width, frame.height, frame.rgbPixels)
            retryWithBackoff(maxRetries = 2) {
                try {
                    tflite.classify(
                        image,
                        modelVersion = manifest.bundleId,
                        confidenceThresholdOverride = confidenceThresholdOverride,
                    )
                } catch (cancelled: CancellationException) {
                    throw cancelled
                } catch (error: Throwable) {
                    throw ModelUnavailableException(
                        "Classifier runtime validation failed: ${error.message ?: error::class.java.simpleName}",
                    )
                }
            }
        }
    }

    override suspend fun detectFrontFrame(
        frame: AnalysisFrame,
        maxClassifiedCandidates: Int,
        focusX: Float?,
        focusY: Float?,
    ): List<FrontOverviewCandidate> {
        return withContext(inferenceDispatcher) {
            applyRuntimeSettings()
            val manifest = manifestReady.await()
                ?: throw ModelUnavailableException("model_manifest.json missing")
            val detectorEntry = manifest.detector ?: throw ModelUnavailableException("Front Overview detector missing")
            if (!assetExists(detectorEntry.file)) {
                throw ModelUnavailableException("Missing detector asset: ${detectorEntry.file}")
            }
            val tflite = try {
                detector ?: run {
                    ModelAssetDigestVerifier.verify(
                        path = detectorEntry.file,
                        expectedSha256 = detectorEntry.sha256,
                    ) {
                        context.assets.open(detectorEntry.file)
                    }
                    TFLiteDetector(
                        context = context,
                        assetModelPath = detectorEntry.file,
                        inputWidth = detectorEntry.inputWidth,
                        inputHeight = detectorEntry.inputHeight,
                        threadCount = activeCpuThreads,
                        outputLayoutHint = detectorEntry.outputLayout,
                        expectedCandidates = detectorEntry.outputCandidates,
                        classCount = detectorEntry.classCount,
                        hasObjectness = detectorEntry.hasObjectness,
                        coordinateSpaceHint = detectorEntry.coordinateSpace,
                        labels = loadDetectorLabels(detectorEntry.labelsFile),
                    ).also { loadedDetector ->
                        detector = loadedDetector
                    }
                }
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Throwable) {
                throw ModelUnavailableException(
                    "Detector runtime validation failed: ${error.message ?: error::class.java.simpleName}",
                )
            }
            val image = RgbImage(frame.width, frame.height, frame.rgbPixels)
            val classifier = classifierFor(manifest)
            retryWithBackoff(maxRetries = 2) {
                try {
                    FrontCandidateClassifier.classifyCandidates(
                        image = image,
                        candidates = tflite.detect(image),
                        maxClassifiedCandidates = maxClassifiedCandidates.coerceIn(0, MAX_CLASSIFIED_CANDIDATES),
                        focusX = focusX,
                        focusY = focusY,
                    ) { crop ->
                        classifier.classifyPlantCrop(
                            crop,
                            modelVersion = manifest.bundleId,
                            confidenceThresholdOverride = confidenceThresholdOverride,
                        )
                    }
                } catch (cancelled: CancellationException) {
                    throw cancelled
                } catch (error: Throwable) {
                    throw ModelUnavailableException(
                        "Detector runtime validation failed: ${error.message ?: error::class.java.simpleName}",
                    )
                }
            }
        }
    }

    override fun observeModelManifest(): Flow<ModelManifest?> = manifestState

    override fun observeModelReadiness(): Flow<ModelBundleReadiness> = readinessState

    private fun loadManifest(): ModelManifest? =
        runCatching {
            val json = context.assets.open("model_manifest.json").bufferedReader().use { it.readText() }
            val root = JSONObject(json)
            val models = root.getJSONObject("models")
            val compatibility = root.getJSONObject("compatibility")
            ModelManifest(
                bundleId = root.getString("bundle_id"),
                createdAt = root.getString("created_at"),
                classifier = models.getJSONObject("classifier").toEntry(),
                detector = models.optJSONObject("detector")?.toEntry(),
                labelsSha256 = root.optString("labels_sha256").takeIf { it.isNotBlank() },
                minAndroidSdk = compatibility.getInt("min_android_sdk"),
                cpuDefault = compatibility.getBoolean("cpu_default"),
                requiresNetwork = compatibility.getBoolean("requires_network"),
            )
        }.getOrNull()

    private fun JSONObject.toEntry(): ModelEntry =
        ModelEntry(
            file = getString("file"),
            source = optString("source"),
            inputWidth = getInt("input_width"),
            inputHeight = getInt("input_height"),
            labelsFile = optString("labels_file").takeIf { it.isNotBlank() },
            sha256 = optString("sha256").takeIf { it.isNotBlank() },
            confidenceThreshold = optDouble("confidence_threshold", Double.NaN).takeIf { !it.isNaN() }?.toFloat(),
            highConfidenceThreshold = optDouble("high_confidence_threshold", Double.NaN).takeIf { !it.isNaN() }?.toFloat(),
            confidenceTemperature = optDouble(
                "confidence_temperature",
                ScanConstants.DEFAULT_CONFIDENCE_TEMPERATURE.toDouble(),
            ).takeIf { it.isFinite() }?.toFloat() ?: ScanConstants.DEFAULT_CONFIDENCE_TEMPERATURE,
            outputCandidates = if (has("output_candidates")) optInt("output_candidates") else null,
            outputLayout = optString("output_layout").takeIf { it.isNotBlank() },
            classCount = if (has("class_count")) optInt("class_count") else null,
            hasObjectness = optBoolean("has_objectness", false),
            coordinateSpace = optString("coordinate_space").takeIf { it.isNotBlank() },
            colorSpace = optString("color_space").takeIf { it.isNotBlank() },
            normalization = optString("normalization").takeIf { it.isNotBlank() },
        )

    private fun loadReadiness(manifest: ModelManifest?): ModelBundleReadiness {
        val labels = runCatching { loadLabels(manifest?.classifier?.labelsFile) }
            .getOrElse { emptyList() }
        reportReadinessProgress(0.2f, "Checking model files...")
        val baseline = ModelBundleReadinessChecker.check(
            manifest = manifest,
            labels = labels,
            assetExists = ::assetExists,
        )
        if (!baseline.isReady || manifest == null) {
            return baseline.copy(progress = 1f)
        }
        reportReadinessProgress(0.45f, "Verifying classifier...")
        val runtimeErrors = validateRuntime(manifest, labels)
        val errors = (baseline.errors + runtimeErrors).distinct()
        return ModelBundleReadiness(
            isReady = errors.isEmpty(),
            statusText = if (errors.isEmpty()) {
                baseline.statusText
            } else {
                "Not ready: ${errors.first()}"
            },
            errors = errors,
            progress = 1f,
        )
    }

    private fun validateRuntime(manifest: ModelManifest, labels: List<String>): List<String> =
        buildList {
            runCatching {
                ModelAssetDigestVerifier.verify(
                    path = manifest.classifier.file,
                    expectedSha256 = manifest.classifier.sha256,
                ) {
                    context.assets.open(manifest.classifier.file)
                }
                verifyLabelsDigest(manifest)
                TFLiteClassifier(
                    context = context,
                    assetModelPath = manifest.classifier.file,
                    labels = labels,
                    inputWidth = manifest.classifier.inputWidth,
                    inputHeight = manifest.classifier.inputHeight,
                    confidenceThreshold = manifest.classifier.confidenceThreshold ?: ScanConstants.DEFAULT_CONFIDENCE_THRESHOLD,
                    highConfidenceThreshold = manifest.classifier.highConfidenceThreshold ?: ScanConstants.DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
                    confidenceTemperature = manifest.classifier.confidenceTemperature,
                    threadCount = 1,
                ).use { classifier ->
                    classifier.classify(syntheticImage(manifest.classifier.inputWidth, manifest.classifier.inputHeight))
                }
            }.onFailure { error ->
                add("Classifier runtime validation failed: ${error.message ?: error::class.java.simpleName}")
            }

            reportReadinessProgress(0.7f, "Verifying detector...")
            val detector = manifest.detector
            if (detector != null) {
                runCatching {
                    ModelAssetDigestVerifier.verify(
                        path = detector.file,
                        expectedSha256 = detector.sha256,
                    ) {
                        context.assets.open(detector.file)
                    }
                    TFLiteDetector(
                        context = context,
                        assetModelPath = detector.file,
                        inputWidth = detector.inputWidth,
                        inputHeight = detector.inputHeight,
                        threadCount = 1,
                        outputLayoutHint = detector.outputLayout,
                        expectedCandidates = detector.outputCandidates,
                        classCount = detector.classCount,
                        hasObjectness = detector.hasObjectness,
                        coordinateSpaceHint = detector.coordinateSpace,
                        labels = loadDetectorLabels(detector.labelsFile),
                    ).use { tfliteDetector ->
                        tfliteDetector.detect(syntheticImage(detector.inputWidth, detector.inputHeight))
                    }
                }.onFailure { error ->
                    add("Detector runtime validation failed: ${error.message ?: error::class.java.simpleName}")
                }
            }
            reportReadinessProgress(0.9f, "Finalizing model...")
        }

    private fun reportReadinessProgress(progress: Float, statusText: String) {
        readinessState.value = readinessState.value.copy(
            progress = progress.coerceIn(0f, 1f),
            statusText = statusText,
        )
    }

    private fun syntheticImage(width: Int, height: Int): RgbImage =
        RgbImage(
            width = width,
            height = height,
            pixels = IntArray(width * height) { index ->
                val x = index % width
                val y = index / width
                if (x in width / 4 until width * 3 / 4 && y in height / 4 until height * 3 / 4) {
                    0xFF2FA84F.toInt()
                } else {
                    0xFF3C3024.toInt()
                }
            },
        )

    private fun loadLabels(labelsFile: String?): List<String> {
        if (labelsFile == null || !assetExists(labelsFile)) {
            return AGRIBOT_LABEL_ORDER.map { it.displayName }
        }
        val text = context.assets.open(labelsFile).bufferedReader().use { it.readText() }
        val labels = JSONArray(text)
        return List(labels.length()) { index -> labels.getString(index) }
    }

    private fun loadDetectorLabels(labelsFile: String?): List<String> {
        if (labelsFile == null || !assetExists(labelsFile)) return listOf("crop")
        val text = context.assets.open(labelsFile).bufferedReader().use { it.readText() }
        val labels = JSONArray(text)
        return List(labels.length()) { index -> labels.getString(index).trim().ifBlank { "crop" } }
    }

    private fun classifierFor(manifest: ModelManifest): TFLiteClassifier {
        val classifierEntry = manifest.classifier
        if (!assetExists(classifierEntry.file)) {
            throw ModelUnavailableException("Missing classifier asset: ${classifierEntry.file}")
        }
        return runCatching {
            classifier ?: run {
                ModelAssetDigestVerifier.verify(
                    path = classifierEntry.file,
                    expectedSha256 = classifierEntry.sha256,
                ) {
                    context.assets.open(classifierEntry.file)
                }
                verifyLabelsDigest(manifest)
                TFLiteClassifier(
                    context = context,
                    assetModelPath = classifierEntry.file,
                    labels = loadLabels(classifierEntry.labelsFile),
                    inputWidth = classifierEntry.inputWidth,
                    inputHeight = classifierEntry.inputHeight,
                    confidenceThreshold = classifierEntry.confidenceThreshold ?: ScanConstants.DEFAULT_CONFIDENCE_THRESHOLD,
                    highConfidenceThreshold = classifierEntry.highConfidenceThreshold ?: ScanConstants.DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
                    confidenceTemperature = classifierEntry.confidenceTemperature,
                    threadCount = activeCpuThreads,
                ).also { loadedClassifier ->
                    classifier = loadedClassifier
                }
            }
        }.getOrElse { error ->
            throw ModelUnavailableException("Classifier runtime validation failed: ${error.message ?: error::class.java.simpleName}")
        }
    }

    private fun verifyLabelsDigest(manifest: ModelManifest) {
        val labelsFile = manifest.classifier.labelsFile
            ?: throw ModelUnavailableException("Missing classifier labels file in manifest")
        ModelAssetDigestVerifier.verify(
            path = labelsFile,
            expectedSha256 = manifest.labelsSha256,
        ) {
            context.assets.open(labelsFile)
        }
    }

    private fun assetExists(path: String): Boolean =
        try {
            context.assets.open(path).close()
            true
        } catch (_: IOException) {
            false
        }

    /**
     * Retry a block with exponential backoff for transient local-runtime contention.
     * Used for inference operations that may fail due to transient issues.
     */
    private suspend fun <T> retryWithBackoff(maxRetries: Int, block: suspend () -> T): T {
        var lastError: Throwable? = null
        repeat(maxRetries + 1) { attempt ->
            try {
                return block()
            } catch (error: ModelUnavailableException) {
                lastError = error
                if (attempt < maxRetries) {
                    delay((50L * (1L shl attempt)).coerceAtMost(200L))
                }
            }
        }
        throw lastError ?: ModelUnavailableException("Unknown error after retries")
    }

    private companion object {
        const val MAX_CLASSIFIED_CANDIDATES = 8
    }
}
