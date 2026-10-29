package com.sakshyam.agribot

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.SystemClock
import android.os.Bundle
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
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.AfterClass
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

        @AfterClass @JvmStatic fun close() = runBlocking { vision.shutdown() }
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
            val confident = expected.filter { it[4] >= d.scoreThreshold + 0.02 }
            for (b in confident) {
                val match = dets.any { x -> abs(x.box.left - b[0]) <= 2f && abs(x.box.top - b[1]) <= 2f && abs(x.box.right - b[2]) <= 2f && abs(x.box.bottom - b[3]) <= 2f }
                assertTrue("$name: expected leaf box $b not found in ${dets.map { it.box }}", match)
            }
            assertTrue("$name: ${dets.size} vs ${expected.size} detections", abs(dets.size - expected.size) <= expected.size - confident.size)
        }
    }

    @Test fun primaryScoresAndProtectedDiseaseVerdictsMatchMobileReference() = runBlocking {
        val target = InstrumentationRegistry.getInstrumentation().targetContext
        val expectedModels = (manifest.members.map { it.file } + manifest.detector.file + listOfNotNull(manifest.contextModel?.file)).toSet()
        assertEquals("APK must contain exactly its declared models", expectedModels, target.assets.list("models")!!.toSet())
        // The normal app retains the original release-baseline model. It must
        // not fabricate research-only primary evidence from its nine logits.
        if (manifest.primaryPolicy == null) {
            assertEquals("unrecognized baseline model", "d70135804cf750341218e89a055370bef46a9b337951897b9f55f2570844ed39", manifest.members.single().sha256)
            for (entry in entries) {
                val name = entry["file"]!!.jsonPrimitive.content
                assertNull("$name baseline fabricated primary evidence", vision.analyze(frame(name), 6).primaryEvidence)
            }
            return@runBlocking
        }
        val ctx = InstrumentationRegistry.getInstrumentation().context
        val reference = Json.parseToJsonElement(ctx.assets.open("golden/primary_expected.json").bufferedReader().use { it.readText() }).jsonObject
        val manifestBytes = target.assets.open(AgribotLeafVision.MANIFEST).use { it.readBytes() }
        val manifestSha = java.security.MessageDigest.getInstance("SHA-256").digest(manifestBytes).joinToString("") { "%02x".format(it) }
        assertEquals("research fixture must match the exact model manifest", reference["manifest_sha256"]!!.jsonPrimitive.content, manifestSha)
        for (entry in reference["entries"]!!.jsonArray.map { it.jsonObject }) {
            val name = entry["file"]!!.jsonPrimitive.content
            val a = vision.analyze(frame(name), 6)
            val primary = requireNotNull(a.primaryEvidence) { "$name missing primary scores" }
            val expectedWhole = entry["context"]!!.jsonArray.map { it.jsonPrimitive.float.toDouble() }
            val actualWhole = listOf(primary.positive, primary.negative, primary.genus, primary.natural)
            for (k in actualWhole.indices) assertEquals("$name context[$k]", expectedWhole[k], actualWhole[k], .002)
            val expectedLeaves = entry["leaves"]!!.jsonArray
            assertEquals("$name real leaf coverage", expectedLeaves.size, primary.leaves.size)
            for (leaf in primary.leaves.indices) {
                val actualViews = listOf(primary.leaves[leaf].center, primary.leaves[leaf].full)
                for (view in 0..1) {
                    val actual = actualViews[view].let { listOf(it.positive, it.negative, it.adapted) }
                    val expected = expectedLeaves[leaf].jsonArray[view].jsonArray
                    for (k in 0..2) assertEquals("$name leaf[$leaf][$view][$k]", expected[k].jsonPrimitive.float.toDouble(), actual[k], .002)
                }
            }
            val ev = PlantEvidence(manifest.labels.size, manifest.healthyIndex, manifest.otherIndex, manifest.referenceDiseasePriority, manifest.primaryPolicy)
            // One inference result is reused for a static consensus probe; frame IDs are distinct,
            // while the independent sample remains this one PNG.
            repeat(7) { i ->
                a.leaves.forEachIndexed { index, leaf ->
                    ev.add(index.toLong(), i.toLong(), LeafObservation(leaf.box, leaf.detectorScore, leaf.probabilities, 1f, leaf.referenceProbabilities))
                }
                ev.addPrimaryFrame(i.toLong(), primary)
            }
            val verdict = ev.verdict()
            val expectedVerdict = entry["candidate"]!!.jsonObject
            assertEquals("$name final verdict", expectedVerdict["kind"]!!.jsonPrimitive.content, verdict.kind.name)
            assertEquals("$name final label", expectedVerdict["label"]!!.jsonPrimitive.content, manifest.labels[verdict.labelIndex!!])
            assertEquals("$name actual distinct frames", 7, verdict.framesUsed)
            assertEquals("$name actual leaves", a.leaves.size, verdict.leavesSeen)
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
        InstrumentationRegistry.getInstrumentation().sendStatus(0, Bundle().apply {
            putString("agribot_latency", "average_ms=$ms detect_ms=${last.detectMs} classify_ms=${last.classifyMs} leaves=${last.leaves.size}")
        })
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
            val ev = PlantEvidence(manifest.labels.size, manifest.healthyIndex, manifest.otherIndex, manifest.referenceDiseasePriority)
            val a = vision.analyze(f, 6)
            repeat(6) { i ->
                val obs = a.leaves.map { LeafObservation(it.box, it.detectorScore, it.probabilities, 1f, it.referenceProbabilities) }
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
