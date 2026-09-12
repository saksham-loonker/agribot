package com.sakshyam.agribot.ml.inference

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.sakshyam.agribot.domain.model.AGRIBOT_LABEL_ORDER
import com.sakshyam.agribot.domain.model.FramePrediction
import com.sakshyam.agribot.ml.preprocessing.MaskCropPreprocessor
import com.sakshyam.agribot.ml.preprocessing.RgbImage
import com.sakshyam.agribot.ml.inference.RgbImageResizer
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.ln
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter

class TFLiteClassifier(
    context: Context,
    private val assetModelPath: String,
    private val labels: List<String> = AGRIBOT_LABEL_ORDER.map { it.displayName },
    private val inputWidth: Int = 224,
    private val inputHeight: Int = 224,
    private val confidenceThreshold: Float = MaskCropPreprocessor.CLF_CONF_THRESHOLD,
    private val highConfidenceThreshold: Float = MaskCropPreprocessor.HIGH_CONF_THRESHOLD,
    private val confidenceTemperature: Float = ClassifierProbabilityCalibrator.DEFAULT_TEMPERATURE,
    private val threadCount: Int = 4,
) : AutoCloseable {
    private val interpreter: Interpreter = Interpreter(loadMappedAsset(context, assetModelPath), Interpreter.Options().apply {
        setNumThreads(threadCount.coerceAtLeast(1))
        setUseXNNPACK(true)
    })
    private val inputShape: IntArray = interpreter.getInputTensor(0).shape().copyOf()
    private val outputShape: IntArray = interpreter.getOutputTensor(0).shape().copyOf()
    private val inputDataType: DataType = interpreter.getInputTensor(0).dataType()
    private val outputDataType: DataType = interpreter.getOutputTensor(0).dataType()
    private val inputBuffer: ByteBuffer = ByteBuffer
        .allocateDirect(checkedInputBufferBytes(inputWidth, inputHeight))
        .order(ByteOrder.nativeOrder())
    private val outputBuffer: Array<FloatArray> = Array(1) { FloatArray(checkedLabelCount(labels)) }
    @Volatile
    var lastDiagnostics: ClassifierDiagnostics = ClassifierDiagnostics.empty(
        inputShape = inputShape,
        outputShape = outputShape,
        inputDataType = inputDataType,
        outputDataType = outputDataType,
    )
        private set

    init {
        require(labels.isNotEmpty()) { "Classifier labels must not be empty" }
        require(inputWidth > 0 && inputHeight > 0) { "Classifier input dimensions must be positive" }
        require(inputWidth.toLong() * inputHeight.toLong() <= MAX_INPUT_PIXELS) {
            "Classifier input dimensions exceed the bounded pixel budget"
        }
        require(inputDataType == DataType.FLOAT32 && outputDataType == DataType.FLOAT32) {
            "Classifier requires FLOAT32 input/output; got input=$inputDataType output=$outputDataType"
        }
        require(inputShape.contentEquals(intArrayOf(1, inputHeight, inputWidth, 3))) {
            "Classifier input tensor ${inputShape.contentToString()} does not match RGB HWC ${inputHeight}x${inputWidth}"
        }
        require(outputShape.contentEquals(intArrayOf(1, labels.size))) {
            "Classifier output tensor ${outputShape.contentToString()} does not match ${labels.size} labels"
        }
    }

    fun classify(bitmap: Bitmap, latencyMs: Double = 0.0, modelVersion: String = "embedded"): FramePrediction {
        val rgb = RgbImage.fromBitmap(bitmap)
        return classify(rgb, latencyMs, modelVersion)
    }

    /** Classifies a complete Side Scan frame using the crop the model was trained on. */
    fun classify(
        image: RgbImage,
        latencyMs: Double = 0.0,
        modelVersion: String = "embedded",
        confidenceThresholdOverride: Float? = null,
    ): FramePrediction = classifyInternal(
        image = MaskCropPreprocessor.preprocessForSideScan(image),
        latencyMs = latencyMs,
        modelVersion = modelVersion,
        confidenceThreshold = confidenceThresholdOverride ?: this.confidenceThreshold,
    )

    /**
     * Classifies a detector crop directly.  Detector crops already isolate a
     * plant; applying the whole-frame vegetation mask a second time can crop
     * out the diseased leaf and changes the training distribution.
     */
    fun classifyPlantCrop(
        image: RgbImage,
        latencyMs: Double = 0.0,
        modelVersion: String = "embedded",
        confidenceThresholdOverride: Float? = null,
    ): FramePrediction = classifyInternal(
        image = image,
        latencyMs = latencyMs,
        modelVersion = modelVersion,
        confidenceThreshold = confidenceThresholdOverride ?: this.confidenceThreshold,
    )

    private fun classifyInternal(
        image: RgbImage,
        latencyMs: Double,
        modelVersion: String,
        confidenceThreshold: Float,
    ): FramePrediction {
        val top = synchronized(this) {
            val startNanos = System.nanoTime()
            interpreter.run(toInputBuffer(image), outputBuffer)
            val measuredLatencyMs = (System.nanoTime() - startNanos) / 1_000_000.0
            val probabilities = ClassifierProbabilityCalibrator.temperatureScale(
                toProbabilities(outputBuffer[0]),
                confidenceTemperature,
            )
            val index = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
            val ranked = probabilities.sortedDescending()
            val top2Margin = if (ranked.size >= 2) {
                (ranked[0] - ranked[1]).coerceIn(0f, 1f)
            } else {
                null
            }
            val entropy = if (probabilities.size > 1) {
                val maximumEntropy = ln(probabilities.size.toDouble())
                (-probabilities.sumOf { probability ->
                    if (probability <= 0f) 0.0 else probability.toDouble() * ln(probability.toDouble())
                } / maximumEntropy).toFloat().coerceIn(0f, 1f)
            } else {
                null
            }
            val healthyConfidence = labels.indexOf("Healthy")
                .takeIf { it >= 0 }
                ?.let { probabilities.getOrNull(it) }
                ?.takeIf { it.isFinite() }
                ?.coerceIn(0f, 1f)
            ClassificationTop(
                confidence = probabilities[index],
                rawLabel = labels.getOrElse(index) { "Unknown" },
                latencyMs = measuredLatencyMs,
                top2Margin = top2Margin,
                entropy = entropy,
                healthyConfidence = healthyConfidence,
            )
        }
        lastDiagnostics = ClassifierDiagnostics(
            inputShape = inputShape.copyOf(),
            outputShape = outputShape.copyOf(),
            inputDataType = inputDataType,
            outputDataType = outputDataType,
            latencyMs = top.latencyMs,
            label = top.rawLabel,
            confidence = top.confidence,
        )
        Log.i(TAG, lastDiagnostics.toLogString())
        return ClassifierDecision.toPrediction(
            rawLabel = top.rawLabel,
            confidence = top.confidence,
            threshold = confidenceThreshold,
            highConfidenceThreshold = highConfidenceThreshold,
            latencyMs = top.latencyMs.coerceAtLeast(latencyMs),
            modelVersion = modelVersion,
            top2Margin = top.top2Margin,
            entropy = top.entropy,
            healthyConfidence = top.healthyConfidence,
        )
    }

    override fun close() {
        interpreter.close()
    }

    private fun toInputBuffer(image: RgbImage): ByteBuffer {
        val resized = RgbImageResizer.resizeBilinear(image, inputWidth, inputHeight)
        inputBuffer.clear()
        for (pixel in resized.pixels) {
            inputBuffer.putFloat(((pixel shr 16) and 0xff) / 255f)
            inputBuffer.putFloat(((pixel shr 8) and 0xff) / 255f)
            inputBuffer.putFloat((pixel and 0xff) / 255f)
        }
        inputBuffer.rewind()
        return inputBuffer
    }

    private fun toProbabilities(values: FloatArray): FloatArray {
        if (values.isEmpty() || values.any { !it.isFinite() }) {
            return if (values.isEmpty()) values.copyOf() else FloatArray(values.size) { 1f / values.size }
        }
        val finiteValues = values.map { it.isFinite() }
        if (finiteValues.all { it } && values.all { it in 0f..1f }) {
            val sum = values.sum()
            if (sum in 0.98f..1.02f) return values.copyOf()
        }

        val maximum = values.maxOrNull() ?: 0f
        val exponentials = values.map { value -> exp((value - maximum).coerceIn(-80f, 80f)) }
        val total = exponentials.sum().takeIf { it.isFinite() && it > 0f } ?: 1f
        return FloatArray(values.size) { index -> (exponentials[index] / total).coerceIn(0f, 1f) }
    }

    private fun loadMappedAsset(context: Context, path: String): MappedByteBuffer {
        context.assets.openFd(path).use { descriptor ->
            FileInputStream(descriptor.fileDescriptor).use { input ->
                val channel = input.channel
                return channel.map(FileChannel.MapMode.READ_ONLY, descriptor.startOffset, descriptor.declaredLength)
            }
        }
    }

    private data class ClassificationTop(
        val confidence: Float,
        val rawLabel: String,
        val latencyMs: Double,
        val top2Margin: Float?,
        val entropy: Float?,
        val healthyConfidence: Float?,
    )

    private companion object {
        const val MAX_INPUT_PIXELS = 1_000_000L
        const val MAX_LABEL_COUNT = 128
        const val MAX_BUFFER_BYTES = 16 * 1024 * 1024

        fun checkedLabelCount(labels: List<String>): Int {
            require(labels.isNotEmpty() && labels.size <= MAX_LABEL_COUNT) {
                "Classifier label count must be between 1 and $MAX_LABEL_COUNT"
            }
            return labels.size
        }

        fun checkedInputBufferBytes(width: Int, height: Int): Int {
            val bytes = width.toLong() * height.toLong() * 3L * Float.SIZE_BYTES
            require(width > 0 && height > 0 && bytes <= MAX_BUFFER_BYTES) {
                "Classifier input buffer exceeds the bounded memory budget"
            }
            return bytes.toInt()
        }
    }
}

data class ClassifierDiagnostics(
    val inputShape: IntArray,
    val outputShape: IntArray,
    val inputDataType: DataType,
    val outputDataType: DataType,
    val latencyMs: Double,
    val label: String,
    val confidence: Float,
) {
    fun toLogString(): String =
        "classifier input=${inputShape.contentToString()}($inputDataType) output=${outputShape.contentToString()}($outputDataType) " +
            "latencyMs=$latencyMs label=$label confidence=$confidence"

    companion object {
        fun empty(
            inputShape: IntArray,
            outputShape: IntArray,
            inputDataType: DataType,
            outputDataType: DataType,
        ) = ClassifierDiagnostics(
            inputShape = inputShape.copyOf(),
            outputShape = outputShape.copyOf(),
            inputDataType = inputDataType,
            outputDataType = outputDataType,
            latencyMs = 0.0,
            label = "not_run",
            confidence = 0f,
        )
    }
}

private const val TAG = "AgribotClassifier"
