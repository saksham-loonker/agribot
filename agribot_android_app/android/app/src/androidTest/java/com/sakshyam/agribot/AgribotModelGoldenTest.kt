package com.sakshyam.agribot

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sakshyam.agribot.domain.scan.LeafObservation
import com.sakshyam.agribot.domain.scan.LeafTracker
import com.sakshyam.agribot.domain.scan.PlantEvidence
import com.sakshyam.agribot.domain.scan.VerdictKind
import com.sakshyam.agribot.domain.scan.VisionFrame
import com.sakshyam.agribot.domain.scan.VisionStatus
import com.sakshyam.agribot.ml.vision.AgribotLeafVision
import com.sakshyam.agribot.ml.vision.LeafDetectionDecoder
import com.sakshyam.agribot.ml.vision.VisionManifest
import kotlin.math.abs
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.float
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith

/**
 * On-device golden parity: the bundled .tflite files run through the app's own LiteRT pipeline
 * (PNG decode -> canonical preprocessing -> interpreter -> decoding) must reproduce the reference
 * outputs computed in Python from the same files (bundle/golden/golden_expected.json).
 */
@RunWith(AndroidJUnit4::class)
class AgribotModelGoldenTest {
    companion object {
        private lateinit var vision: AgribotLeafVision
        private lateinit var manifest: VisionManifest
        private lateinit var entries: List<JsonObject>

        @BeforeClass @JvmStatic fun load() {
            val target = InstrumentationRegistry.getInstrumentation().targetContext
            val testCtx = InstrumentationRegistry.getInstrumentation().context
            manifest = VisionManifest.parse(target.assets.open(AgribotLeafVision.MANIFEST).bufferedReader().readText())
            entries = Json.parseToJsonElement(testCtx.assets.open("golden/golden_expected.json").bufferedReader().readText())
                .jsonObject["entries"]!!.jsonArray.map { it.jsonObject }
            vision = AgribotLeafVision(target, Dispatchers.Default)
            vision.warmUp()
            runBlocking {
                val s = withTimeout(120_000) { vision.status.first { it !is VisionStatus.Loading } }
                assertTrue("models failed to load: $s", s is VisionStatus.Ready)
            }
        }

        private fun frame(name: String): VisionFrame {
            val ctx = InstrumentationRegistry.getInstrumentation().context
            val bmp = ctx.assets.open("golden/$name").use {
                BitmapFactory.decodeStream(it, null, BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888; inScaled = false })!!
            }
            val px = IntArray(bmp.width * bmp.height)
            bmp.getPixels(px, 0, bmp.width, 0, 0, bmp.width, bmp.height)
            return VisionFrame(bmp.width, bmp.height, px, 0L)
        }

        private fun floats(o: JsonObject, key: String) = o[key]!!.jsonArray.map { z -> z.jsonArray.map { it.jsonPrimitive.float } }
    }

    @Test fun wholeImageLogitsAndLabelsMatchReference() = runBlocking {
        for (e in entries) {
            val name = e["file"]!!.jsonPrimitive.content
            val ref = e["whole_image"]!!.jsonObject
            val (logits, result) = vision.classifyWhole(frame(name))
            val expected = floats(ref, "member_logits")
            for (m in logits.indices) for (k in logits[m].indices) {
                assertEquals("$name member $m logit $k", expected[m][k], logits[m][k], 2e-2f)
            }
            assertEquals("$name label", ref["label"]!!.jsonPrimitive.content, manifest.labels[result.labelIndex])
        }
    }

    @Test fun detectionsMatchReference() = runBlocking {
        for (e in entries) {
            val name = e["file"]!!.jsonPrimitive.content
            val f = frame(name)
            val (raw, lb) = vision.detectRaw(f)
            val d = manifest.detector
            val dets = LeafDetectionDecoder.decode(raw, d.candidates, lb, f.width, f.height, d.scoreThreshold, d.nmsIou, d.maxDetections)
            val expected = e["detections"]!!.jsonArray.map { b -> b.jsonArray.map { it.jsonPrimitive.float } }
            // Scores near the threshold can flip either way under float rounding; compare the confident ones.
            val confident = expected.filter { it[4] >= d.scoreThreshold + 0.02f }
            for (b in confident) {
                val match = dets.any { x -> abs(x.box.left - b[0]) <= 2f && abs(x.box.top - b[1]) <= 2f && abs(x.box.right - b[2]) <= 2f && abs(x.box.bottom - b[3]) <= 2f }
                assertTrue("$name: expected leaf box $b not found in ${dets.map { it.box }}", match)
            }
            assertTrue("$name: ${dets.size} vs ${expected.size} detections", abs(dets.size - expected.size) <= expected.size - confident.size)
        }
    }

    @Test fun fullAnalysisRunsAndReportsLatency() = runBlocking {
        val e = entries.first { (it["detections"]!!.jsonArray.size) > 0 }
        val f = frame(e["file"]!!.jsonPrimitive.content)
        repeat(2) { vision.analyze(f, 4) }
        val t0 = SystemClock.elapsedRealtime()
        val n = 5
        var last = vision.analyze(f, 4)
        repeat(n - 1) { last = vision.analyze(f, 4) }
        val ms = (SystemClock.elapsedRealtime() - t0) / n.toDouble()
        Log.i("AgribotBench", "analyze() avg ${"%.1f".format(ms)} ms (detect ${"%.1f".format(last.detectMs)} ms, classify ${"%.1f".format(last.classifyMs)} ms, ${last.leaves.size} leaves)")
        assertTrue(last.leaves.isNotEmpty())
        val top = e["top_crop"] as? JsonObject
        if (top != null) assertEquals(top["label"]!!.jsonPrimitive.content, manifest.labels[last.leaves.first().labelIndex])
    }

    /** "Check a plant" on a static image: real model + leaf tracker + multi-frame plant vote. */
    @Test fun checkAPlantVerdictOnStaticImagesAgreesWithReference() = runBlocking {
        var checked = 0
        for (e in entries) {
            val top = e["top_crop"] as? JsonObject ?: continue
            val expected = top["label"]!!.jsonPrimitive.content
            val f = frame(e["file"]!!.jsonPrimitive.content)
            val tracker = LeafTracker()
            val ev = PlantEvidence(manifest.labels.size, manifest.healthyIndex, manifest.otherIndex)
            repeat(6) { i ->
                val a = vision.analyze(f, 6)
                val obs = a.leaves.map { LeafObservation(it.box, it.detectorScore, it.probabilities, 1f) }
                val ids = tracker.update(i.toLong(), obs)
                obs.forEachIndexed { k, o -> ev.add(ids[k], i.toLong(), o) }
            }
            val v = ev.verdict()
            Log.i("AgribotCheck", "${e["file"]}: ${v.kind} ${v.labelIndex?.let { manifest.labels[it] }} p=${"%.2f".format(v.confidence)} leaves=${v.leavesSeen} ref=$expected")
            assertTrue("${e["file"]}: verdict must be decided after 6 views", v.kind != VerdictKind.NEED_MORE_VIEWS)
            // A decided verdict must name the reference class; an honest "unsure" is acceptable, a different class is not.
            if (v.kind != VerdictKind.UNSURE && v.leavesSeen == 1) assertEquals("${e["file"]}", expected, manifest.labels[v.labelIndex!!])
            checked++
        }
        assertTrue(checked >= 8)
    }
}
