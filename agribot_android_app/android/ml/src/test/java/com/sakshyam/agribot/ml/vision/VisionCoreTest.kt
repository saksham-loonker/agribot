package com.sakshyam.agribot.ml.vision

import java.io.File
import kotlin.math.abs
import kotlin.math.exp
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

private fun solid(w: Int, h: Int, rgb: Int) = RgbFrame(w, h, IntArray(w * h) { (0xFF shl 24) or rgb })

class CanonicalImageOpsTest {
    @Test fun `identity resize is exact`() {
        val f = RgbFrame(5, 3, IntArray(15) { (0xFF shl 24) or (it * 17 shl 16) or (it * 7 shl 8) or it })
        val out = CanonicalImageOps.resize(f, 0, 0, 5, 3, 5, 3)
        for (i in 0 until 15) assertEquals(((f.pixels[i] shr 16) and 0xFF).toFloat(), out[i * 3], 1e-4f)
    }

    @Test fun `constant images stay constant when shrinking and enlarging`() {
        val f = solid(37, 23, 0x406080)
        for ((w, h) in listOf(5 to 4, 100 to 61, 37 to 23)) {
            val out = CanonicalImageOps.resize(f, 0, 0, 37, 23, w, h)
            for (i in out.indices step 3) { assertEquals(64f, out[i], 1e-3f); assertEquals(96f, out[i + 1], 1e-3f); assertEquals(128f, out[i + 2], 1e-3f) }
        }
    }

    @Test fun `downscale averages areas`() {
        // 2x1 image (0, 200) -> 1x1 must be the mean 100.
        val f = RgbFrame(2, 1, intArrayOf(0xFF000000.toInt(), (0xFF shl 24) or (200 shl 16)))
        assertEquals(100f, CanonicalImageOps.resize(f, 0, 0, 2, 1, 1, 1)[0], 1e-4f)
    }

    @Test fun `letterbox geometry for a 4 by 3 frame`() {
        val (input, lb) = CanonicalImageOps.detectorInput(solid(640, 480, 0xFFFFFF), 320)
        assertEquals(0.5, lb.scale, 1e-12)
        assertEquals(0, lb.padX); assertEquals(40, lb.padY)
        assertEquals(114f / 255f, input[0], 1e-6f)                     // top pad row is grey
        assertEquals(1f, input[(100 * 320 + 160) * 3], 1e-5f)          // image area is white
        val (x, y) = lb.toFrame(160f, 160f)
        assertEquals(320f, x, 1e-3f); assertEquals(240f, y, 1e-3f)
    }

    @Test fun `letterbox rounding matches python round half even`() {
        // 5x2 frame into 4: k = 0.8, h = 1.6 -> rint 2 ; w = 4
        val (_, lb) = CanonicalImageOps.detectorInput(solid(5, 2, 0), 4)
        assertEquals(0, lb.padX); assertEquals(1, lb.padY)
        // 8x5 into 4: k = 0.5, h*k = 2.5 -> half-even 2 (Math.round would give 3), pad = 1
        assertEquals(1, CanonicalImageOps.detectorInput(solid(8, 5, 0), 4).second.padY)
    }

    @Test fun `crop rect pads and clamps and degenerate crops are rejected`() {
        assertArrayEquals(intArrayOf(9, 18, 31, 42), CanonicalImageOps.cropRect(100, 100, PixelBox(11f, 20f, 29f, 40f), 0.10))
        assertArrayEquals(intArrayOf(0, 0, 100, 100), CanonicalImageOps.cropRect(100, 100, PixelBox(-50f, -5f, 150f, 120f), 0.10))
        assertNull(CanonicalImageOps.classifierInput(solid(50, 50, 0), 32, PixelBox(10f, 10f, 10.5f, 30f), 0.0))
    }

    @Test fun `classifier input is normalised with imagenet statistics`() {
        val t = CanonicalImageOps.classifierInput(solid(10, 20, 0xFFFFFF), 8)!!
        assertEquals(8 * 8 * 3, t.size)
        assertEquals((1f - 0.485f) / 0.229f, t[0], 1e-4f)
        assertEquals((1f - 0.406f) / 0.225f, t[2], 1e-4f)
    }
}

class LeafDetectionDecoderTest {
    private val lb = CanonicalImageOps.Letterbox(0.5, 0, 40, 320)

    private fun out(vararg boxes: FloatArray): FloatArray {
        val n = boxes.size
        return FloatArray(5 * n).also { o -> boxes.forEachIndexed { i, b -> for (r in 0 until 5) o[r * n + i] = b[r] } }
    }

    @Test fun `maps letterboxed corners to frame pixels and filters by score`() {
        val o = out(floatArrayOf(10f, 50f, 110f, 150f, 0.9f), floatArrayOf(0f, 40f, 50f, 90f, 0.2f))
        val d = LeafDetectionDecoder.decode(o, 2, lb, 640, 480, 0.35, 0.6, 16)
        assertEquals(1, d.size)
        assertEquals(PixelBox(20f, 20f, 220f, 220f), d[0].box)
    }

    @Test fun `nms keeps the strongest of overlapping boxes, clamps and caps`() {
        val o = out(
            floatArrayOf(10f, 50f, 110f, 150f, 0.8f), floatArrayOf(12f, 52f, 112f, 152f, 0.95f),
            floatArrayOf(200f, 60f, 400f, 300f, 0.7f), floatArrayOf(250f, 200f, 260f, 200.5f, 0.99f),
        )
        val d = LeafDetectionDecoder.decode(o, 4, lb, 640, 480, 0.35, 0.6, 16)
        assertEquals(listOf(0.95f, 0.7f), d.map { it.score })                   // tiny box dropped, duplicate suppressed
        assertEquals(640f, d[1].box.right, 1e-4f)                                 // clamped to frame
        assertEquals(1, LeafDetectionDecoder.decode(o, 4, lb, 640, 480, 0.35, 0.6, 1).size)
    }

    @Test fun `rejects truncated output`() {
        assertThrows(IllegalArgumentException::class.java) { LeafDetectionDecoder.decode(FloatArray(9), 2, lb, 10, 10, 0.3, 0.5, 4) }
    }
}

class EnsembleDecisionTest {
    @Test fun `single member is a biased tempered softmax`() {
        val d = EnsembleDecision(3, 2, otherBias = 1.0, temperature = 2.0, energyRejectBelow = null)
        val r = d.combine(listOf(floatArrayOf(2f, 0f, 0f)))
        val z = doubleArrayOf(2.0, 0.0, 1.0).map { it - kotlin.math.ln(exp(2.0) + 2) }.map { it / 2 }
        val e = z.map { exp(it) }; val s = e.sum()
        for (k in 0..2) assertEquals(e[k] / s, r.probabilities[k].toDouble(), 1e-6)
        assertEquals(0, r.labelIndex)
        assertTrue(abs(r.probabilities.sum() - 1f) < 1e-5f)
    }

    @Test fun `members are averaged in probability space`() {
        val d = EnsembleDecision(2, 1, 0.0, 1.0, null)
        val r = d.combine(listOf(floatArrayOf(10f, 0f), floatArrayOf(0f, 10f)))
        assertEquals(0.5f, r.probabilities[0], 1e-4f)
    }

    @Test fun `energy gate rejects low-evidence inputs to Other`() {
        val d = EnsembleDecision(3, 2, 0.0, 1.0, energyRejectBelow = 5.0)
        val r = d.combine(listOf(floatArrayOf(1f, 0f, 0f)))
        assertTrue(r.rejectedByEnergy); assertEquals(2, r.labelIndex)
    }

    @Test fun `wrong class count is rejected`() {
        assertThrows(IllegalArgumentException::class.java) { EnsembleDecision(3, 2, 0.0, 1.0, null).combine(listOf(floatArrayOf(1f))) }
    }
}

class VisionManifestTest {
    private val text = File("../app/src/main/assets/model_manifest.json").readText()

    @Test fun `bundled manifest parses into a complete contract`() {
        val m = VisionManifest.parse(text)
        assertEquals(9, m.labels.size)
        assertEquals("Other", m.labels[m.otherIndex])
        assertEquals("Healthy", m.labels[m.healthyIndex])
        assertEquals(320, m.detector.inputSize)
        assertEquals(2100, m.detector.candidates)
        assertTrue(m.members.all { it.sha256.length == 64 && it.inputSize > 0 })
        assertTrue(m.temperature > 0)
    }

    @Test fun `unsupported schema or detector layout is rejected`() {
        assertThrows(IllegalArgumentException::class.java) { VisionManifest.parse(text.replace("\"schema\": 2", "\"schema\": 1")) }
        assertThrows(IllegalArgumentException::class.java) { VisionManifest.parse(text.replace("\"x1\"", "\"cx\"")) }
    }
}
