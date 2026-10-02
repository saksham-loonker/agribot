package com.sakshyam.agribot

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sakshyam.agribot.domain.scan.VisionFrame
import com.sakshyam.agribot.ml.vision.AgribotLeafVision
import com.sakshyam.agribot.ml.vision.InferenceOptions
import com.sakshyam.agribot.ml.vision.InferenceBackend
import com.sakshyam.agribot.ml.vision.LeafDetectionDecoder
import com.sakshyam.agribot.ml.vision.VisionManifest
import java.util.concurrent.Executors
import kotlinx.coroutines.CancellationException
import kotlin.math.abs
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.float
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Developer benchmark: identical golden pixels, fresh runtimes, no result caching. */
@RunWith(AndroidJUnit4::class)
class AgribotRuntimeBenchmarkTest {
    @Test fun compareRuntimesWithGoldenParity() = runBlocking {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val target = instrumentation.targetContext
        val assets = instrumentation.context.assets
        val entries = Json.parseToJsonElement(assets.open("golden/golden_expected.json")
            .bufferedReader().use { it.readText() }).jsonObject["entries"]!!.jsonArray
        val frames = entries.map { entry ->
            val name = entry.jsonObject["file"]!!.jsonPrimitive.content
            val bitmap = assets.open("golden/$name").use {
                BitmapFactory.decodeStream(it, null, BitmapFactory.Options().apply {
                    inPreferredConfig = Bitmap.Config.ARGB_8888; inScaled = false
                })!!
            }
            val pixels = IntArray(bitmap.width * bitmap.height)
            bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
            VisionFrame(bitmap.width, bitmap.height, pixels, 0L).also { bitmap.recycle() }
        }
        val frame = frames[entries.indexOfFirst { it.jsonObject["detections"]!!.jsonArray.isNotEmpty() }]
        val manifest = VisionManifest.parse(target.assets.open(AgribotLeafVision.MANIFEST).bufferedReader().use { it.readText() })
        val baseline = Executors.newSingleThreadExecutor().asCoroutineDispatcher().use { dispatcher ->
            val vision = AgribotLeafVision(target, dispatcher, InferenceOptions())
            try { vision.analyze(frame, 4) } finally { vision.shutdown() }
        }
        val cpuOptions = listOf(1, 2, 4, 6, 8).filter { it <= Runtime.getRuntime().availableProcessors() }
            .map { InferenceOptions(it, it) }
        val optionsToCompare = cpuOptions + listOf(
            InferenceOptions(detectorBackend = InferenceBackend.GPU),
            InferenceOptions(classifierBackend = InferenceBackend.GPU),
            InferenceOptions(detectorBackend = InferenceBackend.GPU, classifierBackend = InferenceBackend.GPU),
            InferenceOptions(detectorBackend = InferenceBackend.NNAPI, classifierBackend = InferenceBackend.NNAPI),
        )
        for (options in optionsToCompare) {
            Executors.newSingleThreadExecutor().asCoroutineDispatcher().use { dispatcher ->
                val vision = AgribotLeafVision(target, dispatcher, options)
                try {
                    var maxDelta = 0f
                    for (i in entries.indices) {
                        val entry = entries[i].jsonObject
                        val reference = entries[i].jsonObject["whole_image"]!!.jsonObject
                        val (logits, result) = vision.classifyWhole(frames[i])
                        val expected = reference["member_logits"]!!.jsonArray
                        for (m in logits.indices) for (k in logits[m].indices) {
                            maxDelta = maxOf(maxDelta, abs(logits[m][k] - expected[m].jsonArray[k].jsonPrimitive.float))
                        }
                        assertEquals(reference["label"]!!.jsonPrimitive.content,
                            vision.status.value.let { (it as com.sakshyam.agribot.domain.scan.VisionStatus.Ready).info.labels[result.labelIndex] })
                        val (raw, letterbox) = vision.detectRaw(frames[i])
                        val detector = manifest.detector
                        val detections = LeafDetectionDecoder.decode(raw, detector.candidates, letterbox,
                            frames[i].width, frames[i].height, detector.scoreThreshold, detector.nmsIou, detector.maxDetections)
                        val expectedBoxes = entry["detections"]!!.jsonArray.map { box -> box.jsonArray.map { it.jsonPrimitive.float } }
                        val confident = expectedBoxes.filter { it[4] >= detector.scoreThreshold + 0.02f }
                        for (box in confident) assertTrue("$options: missing golden box $box", detections.any {
                            abs(it.box.left - box[0]) <= 2f && abs(it.box.top - box[1]) <= 2f &&
                                abs(it.box.right - box[2]) <= 2f && abs(it.box.bottom - box[3]) <= 2f
                        })
                        assertTrue("$options: detection count differs", abs(detections.size - expectedBoxes.size) <= expectedBoxes.size - confident.size)
                    }
                    assertTrue("$options maxLogitDelta=$maxDelta", maxDelta <= 2e-2f)
                    repeat(3) { vision.analyze(frame, 4) }
                    val samples = DoubleArray(10)
                    var detector = 0.0
                    var classifier = 0.0
                    var leaves = 0
                    for (i in samples.indices) {
                        val start = SystemClock.elapsedRealtimeNanos()
                        val analysis = vision.analyze(frame, 4)
                        samples[i] = (SystemClock.elapsedRealtimeNanos() - start) / 1e6
                        assertEquals("$options: workload leaf count", baseline.leaves.size, analysis.leaves.size)
                        assertEquals("$options: detector count", baseline.detectedLeaves, analysis.detectedLeaves)
                        assertEquals("$options: crop labels", baseline.leaves.map { it.labelIndex }, analysis.leaves.map { it.labelIndex })
                        detector += analysis.detectMs
                        classifier += analysis.classifyMs
                        leaves = analysis.leaves.size
                    }
                    val sorted = samples.sorted()
                    Log.i("AgribotTune", "$options avg=${samples.average()} ms median=${(sorted[4] + sorted[5]) / 2} ms p95=${sorted.last()} ms detectAvg=${detector / samples.size} ms classifyAvg=${classifier / samples.size} ms leaves=$leaves maxLogitDelta=$maxDelta")
                } catch (failure: Throwable) {
                    if (failure is CancellationException || failure is VirtualMachineError || failure is ThreadDeath || failure is LinkageError) throw failure
                    if (options in cpuOptions) throw failure
                    Log.w("AgribotTune", "REJECTED $options: ${failure.message}")
                } finally {
                    vision.shutdown()
                }
            }
        }
    }
}
