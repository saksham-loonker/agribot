package com.sakshyam.agribot.ml.vision

import android.content.Context
import android.content.SharedPreferences
import android.os.SystemClock
import com.sakshyam.agribot.domain.model.BoundingBox
import com.sakshyam.agribot.domain.scan.AnalyzedLeaf
import com.sakshyam.agribot.domain.scan.FrameAnalysis
import com.sakshyam.agribot.domain.scan.LeafVision
import com.sakshyam.agribot.domain.scan.VisionFrame
import com.sakshyam.agribot.domain.scan.VisionModelInfo
import com.sakshyam.agribot.domain.scan.VisionStatus
import com.sakshyam.agribot.ml.di.InferenceDispatcher
import dagger.hilt.android.qualifiers.ApplicationContext
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.security.MessageDigest
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter

/** One LiteRT interpreter with reusable direct input/output buffers. Not thread-safe. */
internal class LiteRtRunner(model: MappedByteBuffer, threads: Int, useXnnpack: Boolean) : AutoCloseable {
    private val interpreter = Interpreter(model, Interpreter.Options().setNumThreads(threads).setUseXNNPACK(useXnnpack))
    val inputShape: IntArray = interpreter.getInputTensor(0).shape()
    val outputShape: IntArray = interpreter.getOutputTensor(0).shape()
    private val input = ByteBuffer.allocateDirect(inputShape.fold(4) { a, b -> a * b }).order(ByteOrder.nativeOrder())
    private val output = ByteBuffer.allocateDirect(outputShape.fold(4) { a, b -> a * b }).order(ByteOrder.nativeOrder())
    val outputSize: Int = outputShape.fold(1) { a, b -> a * b }

    fun run(values: FloatArray): FloatArray {
        require(values.size * 4 == input.capacity()) { "input ${values.size} floats != tensor ${input.capacity() / 4}" }
        input.rewind(); input.asFloatBuffer().put(values)
        output.rewind()
        interpreter.run(input, output)
        output.rewind()
        val out = FloatArray(outputSize)
        output.asFloatBuffer().get(out)
        return out
    }

    override fun close() = interpreter.close()
}

/**
 * On-device leaf detector + disease classifier ensemble. Everything model-specific comes from
 * assets/model_manifest.json; preprocessing is [CanonicalImageOps], identical to training/export.
 */
@Singleton
class AgribotLeafVision @Inject constructor(
    @ApplicationContext private val context: Context,
    @InferenceDispatcher private val dispatcher: CoroutineDispatcher,
) : LeafVision {
    private val _status = MutableStateFlow<VisionStatus>(VisionStatus.Loading)
    override val status: StateFlow<VisionStatus> = _status.asStateFlow()

    private val scope = CoroutineScope(SupervisorJob() + dispatcher)
    private val loadLock = Mutex()
    private var manifest: VisionManifest? = null
    private var detector: LiteRtRunner? = null
    private var members: List<LiteRtRunner> = emptyList()
    private var decision: EnsembleDecision? = null

    override fun warmUp() {
        if (_status.value is VisionStatus.Ready) return
        scope.launch { ensureLoaded() }
    }

    private suspend fun ensureLoaded(): Boolean = loadLock.withLock {
        if (manifest != null) return@withLock true
        _status.value = VisionStatus.Loading
        runCatching {
            val m = VisionManifest.parse(context.assets.open(MANIFEST).bufferedReader().use { it.readText() })
            verifyOnce(m)
            val threads = Runtime.getRuntime().availableProcessors().coerceIn(2, 4)
            val xnn = !isEmulator()
            val det = LiteRtRunner(map(m.detector.file), threads, xnn)
            check(det.inputShape.contentEquals(intArrayOf(1, m.detector.inputSize, m.detector.inputSize, 3))) { "detector input ${det.inputShape.toList()}" }
            check(det.outputSize == 5 * m.detector.candidates) { "detector output ${det.outputShape.toList()}" }
            val mem = m.members.map { mm ->
                LiteRtRunner(map(mm.file), threads, xnn).also { r ->
                    check(r.inputShape.contentEquals(intArrayOf(1, mm.inputSize, mm.inputSize, 3))) { "${mm.file} input ${r.inputShape.toList()}" }
                    check(r.outputSize == m.labels.size) { "${mm.file} output ${r.outputShape.toList()}" }
                }
            }
            detector = det; members = mem; manifest = m
            decision = EnsembleDecision(m.labels.size, m.otherIndex, m.otherLogitBias, m.temperature, m.energyRejectBelow)
            _status.value = VisionStatus.Ready(VisionModelInfo(m.bundleId, m.labels, m.healthyIndex, m.otherIndex, m.members.size))
        }.onFailure { e ->
            close()
            _status.value = VisionStatus.Failed(e.message ?: e.javaClass.simpleName)
        }.isSuccess
    }

    override suspend fun analyze(frame: VisionFrame, maxLeaves: Int): FrameAnalysis = withContext(dispatcher) {
        check(ensureLoaded()) { (status.value as? VisionStatus.Failed)?.reason ?: "models unavailable" }
        val m = manifest!!
        val rgb = RgbFrame(frame.width, frame.height, frame.argb)
        val t0 = SystemClock.elapsedRealtimeNanos()
        val (detIn, lb) = CanonicalImageOps.detectorInput(rgb, m.detector.inputSize)
        val dets = LeafDetectionDecoder.decode(
            detector!!.run(detIn), m.detector.candidates, lb, frame.width, frame.height,
            m.detector.scoreThreshold, m.detector.nmsIou, m.detector.maxDetections,
        )
        val t1 = SystemClock.elapsedRealtimeNanos()
        val leaves = dets.take(maxLeaves).mapNotNull { d ->
            val logits = m.members.zip(members).map { (mm, runner) ->
                runner.run(CanonicalImageOps.classifierInput(rgb, mm.inputSize, d.box) ?: return@mapNotNull null)
            }
            val r = decision!!.combine(logits)
            AnalyzedLeaf(BoundingBox(d.box.left, d.box.top, d.box.right, d.box.bottom), d.score, r.probabilities, r.labelIndex)
        }
        val t2 = SystemClock.elapsedRealtimeNanos()
        FrameAnalysis(frame.width, frame.height, leaves, dets.size, (t1 - t0) / 1e6, (t2 - t1) / 1e6)
    }

    /** Classify the whole image (no detector); used for golden tests and as a fallback. */
    suspend fun classifyWhole(frame: VisionFrame): Pair<List<FloatArray>, EnsembleDecision.Result> = withContext(dispatcher) {
        check(ensureLoaded())
        val m = manifest!!
        val rgb = RgbFrame(frame.width, frame.height, frame.argb)
        val logits = m.members.zip(members).map { (mm, r) -> r.run(CanonicalImageOps.classifierInput(rgb, mm.inputSize)!!) }
        logits to decision!!.combine(logits)
    }

    /** Raw detector output for golden tests. */
    @androidx.annotation.VisibleForTesting
    suspend fun detectRaw(frame: VisionFrame): Pair<FloatArray, CanonicalImageOps.Letterbox> = withContext(dispatcher) {
        check(ensureLoaded())
        val (inp, lb) = CanonicalImageOps.detectorInput(RgbFrame(frame.width, frame.height, frame.argb), manifest!!.detector.inputSize)
        detector!!.run(inp) to lb
    }

    /**
     * Android emulators on Apple silicon advertise SVE2 to the guest but cannot execute it, and XNNPACK's
     * SVE kernels then die with SIGILL. Emulators are QA-only, so they use LiteRT's built-in kernels
     * (same model and ops, slower); real phones keep XNNPACK.
     */
    private fun isEmulator(): Boolean {
        val qemu = runCatching {
            Class.forName("android.os.SystemProperties").getMethod("get", String::class.java).invoke(null, "ro.kernel.qemu") as String
        }.getOrDefault("")
        return qemu == "1" || android.os.Build.FINGERPRINT.contains("emulator") || android.os.Build.HARDWARE.contains("ranchu")
    }

    private fun map(file: String): MappedByteBuffer = context.assets.openFd(MODEL_DIR + file).use { fd ->
        FileInputStream(fd.fileDescriptor).channel.use { ch -> ch.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength) }
    }

    /** SHA-256 of every model file, checked once per installed APK + bundle and then cached. */
    private fun verifyOnce(m: VisionManifest) {
        val prefs: SharedPreferences = context.getSharedPreferences("agribot_models", Context.MODE_PRIVATE)
        val installed = context.packageManager.getPackageInfo(context.packageName, 0).lastUpdateTime
        val key = "${m.bundleId}@$installed"
        if (prefs.getString("verified", null) == key) return
        val files = m.members.map { it.file to it.sha256 } + (m.detector.file to m.detector.sha256)
        for ((file, expected) in files) {
            val md = MessageDigest.getInstance("SHA-256")
            context.assets.open(MODEL_DIR + file).use { s ->
                val buf = ByteArray(1 shl 16)
                while (true) { val n = s.read(buf); if (n < 0) break; md.update(buf, 0, n) }
            }
            val actual = md.digest().joinToString("") { "%02x".format(it) }
            check(actual == expected) { "$file is corrupted (sha256 mismatch)" }
        }
        prefs.edit().putString("verified", key).apply()
    }

    private fun close() {
        detector?.close(); members.forEach { it.close() }
        detector = null; members = emptyList(); manifest = null; decision = null
    }

    companion object {
        const val MANIFEST = "model_manifest.json"
        const val MODEL_DIR = "models/"
    }
}
