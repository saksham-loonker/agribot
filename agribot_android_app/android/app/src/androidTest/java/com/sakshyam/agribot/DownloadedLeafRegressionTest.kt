package com.sakshyam.agribot

import android.graphics.BitmapFactory
import android.graphics.Bitmap
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sakshyam.agribot.domain.scan.*
import com.sakshyam.agribot.ml.vision.AgribotLeafVision
import java.io.File
import java.security.MessageDigest
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import kotlinx.serialization.json.*
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Opt-in private downloaded fixtures; no third-party images are added to the source repository. */
@RunWith(AndroidJUnit4::class)
class DownloadedLeafRegressionTest {
    @Test fun downloadedPhotosMatchRetainedReferenceVerdicts() = runBlocking {
        val directory = InstrumentationRegistry.getArguments().getString("agribotFixtures")
        assumeTrue("Downloaded device fixtures not requested", directory != null)
        val root = File(directory!!)
        val cases = Json.parseToJsonElement(File(root, "cases.json").readText()).jsonArray
        val target = InstrumentationRegistry.getInstrumentation().targetContext
        val vision = AgribotLeafVision(target, Dispatchers.Default)
        try {
            vision.warmUp()
            val status = withTimeout(120_000) { vision.status.first { it !is VisionStatus.Loading } }
            assertTrue("Models failed: $status", status is VisionStatus.Ready)
            val info = (status as VisionStatus.Ready).info
            for (item in cases) {
                val c = item.jsonObject
                File(root, "progress.txt").writeText("Analyzing ${c["id"]}")
                val bitmap = BitmapFactory.decodeFile(File(root, c["file"]!!.jsonPrimitive.content).path,
                    BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888; inScaled = false })!!
                val pixels = IntArray(bitmap.width * bitmap.height)
                bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
                val rgb = ByteArray(pixels.size * 3)
                pixels.forEachIndexed { n, pixel ->
                    rgb[n * 3] = (pixel ushr 16).toByte()
                    rgb[n * 3 + 1] = (pixel ushr 8).toByte()
                    rgb[n * 3 + 2] = pixel.toByte()
                }
                val hash = MessageDigest.getInstance("SHA-256").digest(rgb).joinToString("") { "%02x".format(it) }
                assertEquals("Fixture RGB decoding must match the host", c["rgb_sha256"]!!.jsonPrimitive.content, hash)
                val frame = VisionFrame(bitmap.width, bitmap.height, pixels, 1)
                bitmap.recycle()
                val tracker = LeafTracker()
                val evidence = PlantEvidence(info.labels.size, info.healthyIndex, info.otherIndex, info.referenceDiseasePriority)
                repeat(7) { n ->
                    val checked = vision.analyzeUsable(frame, 6)
                    val a = checked.analysis ?: error("Regression fixture has no usable detail")
                    val observations = a.leaves.map { LeafObservation(it.box, it.detectorScore, it.probabilities, 1f, it.referenceProbabilities) }
                    val ids = tracker.update(n.toLong(), observations)
                    observations.forEachIndexed { k, o -> evidence.add(ids[k], n.toLong(), o) }
                }
                val actual = evidence.verdict()
                File(root, "progress.txt").writeText("Completed ${c["id"]}: $actual")
                if (c["source_label"]!!.jsonPrimitive.content != "Healthy") {
                    assertNotEquals("A disease fixture must not become Healthy: ${c["id"]}", "HEALTHY", actual.kind.name)
                }
                assertEquals(c["id"]!!.jsonPrimitive.content, c["kind"]!!.jsonPrimitive.content, actual.kind.name)
                c["label"]?.jsonPrimitive?.contentOrNull?.let { label ->
                    assertEquals(label, actual.labelIndex?.let { info.labels[it] })
                }
            }
            File(root, "progress.txt").appendText("\nAll downloaded cases passed")
        } finally {
            vision.shutdown()
        }
    }
}
