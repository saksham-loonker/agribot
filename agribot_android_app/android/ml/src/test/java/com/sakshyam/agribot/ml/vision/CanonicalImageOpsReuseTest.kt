package com.sakshyam.agribot.ml.vision

import java.io.File
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.random.Random
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotSame
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

/** The independent reference below preserves the pre-optimization float operation order. */
class CanonicalImageOpsReuseTest {
    @Test fun `resize remains bit exact for random identity downscale upscale and cropped frames`() {
        val random = Random(3821)
        repeat(60) {
            val frame = randomFrame(random, random.nextInt(1, 43), random.nextInt(1, 39))
            val l = random.nextInt(frame.width)
            val t = random.nextInt(frame.height)
            val w = random.nextInt(1, frame.width - l + 1)
            val h = random.nextInt(1, frame.height - t + 1)
            val outW = random.nextInt(1, 53)
            val outH = random.nextInt(1, 49)
            assertExact(referenceResize(frame, l, t, w, h, outW, outH),
                CanonicalImageOps.resize(frame, l, t, w, h, outW, outH))
        }
    }

    @Test fun `classifier reusable input matches reference at identity downscale and upscale sizes`() {
        val random = Random(6197)
        for ((w, h) in listOf(1 to 1, 9 to 9, 11 to 21, 31 to 8, 37 to 23)) {
            val frame = randomFrame(random, w, h)
            for (size in listOf(1, 7, 16, 48)) {
                val destination = FloatArray(size * size * 3) { Float.NaN }
                assertTrue(CanonicalImageOps.classifierInputInto(frame, size, destination))
                assertExact(referenceClassifier(frame, size)!!, destination)
            }
        }
    }

    @Test fun `classifier crop padding clipping reversed coordinates and centred square match reference`() {
        val frame = randomFrame(Random(8139), 73, 59)
        val boxes = listOf(
            PixelBox(11f, 20f, 29f, 40f),
            PixelBox(-10f, -5f, 80f, 70f),
            PixelBox(60.125f, 48.75f, 7.375f, 4.25f),
            PixelBox(0f, 0f, 2f, 2f),
            PixelBox(17.000002f, 12.1f, 51.999996f, 37.9f),
        )
        for (box in boxes) {
            for (pad in listOf(0.0, 0.10, 0.35)) {
                val destination = FloatArray(19 * 19 * 3) { Float.NaN }
                assertTrue(CanonicalImageOps.classifierInputInto(frame, 19, destination, box, pad))
                assertExact(referenceClassifier(frame, 19, box, pad)!!, destination)
            }
        }
    }

    @Test fun `degenerate classifier crop is rejected without modifying destination`() {
        val frame = randomFrame(Random(912), 50, 50)
        val destination = FloatArray(8 * 8 * 3) { it.toFloat() }
        val original = destination.copyOf()
        for (box in listOf(PixelBox(10f, 10f, 10.5f, 30f), PixelBox(80f, 80f, 90f, 90f))) {
            assertFalse(CanonicalImageOps.classifierInputInto(frame, 8, destination, box, 0.0))
            assertExact(original, destination)
        }
    }

    @Test fun `detector reusable input overwrites image and every padding area on each call`() {
        val random = Random(287)
        val destination = FloatArray(32 * 32 * 3) { Float.NaN }
        for ((w, h) in listOf(37 to 8, 9 to 31, 32 to 32, 1 to 1, 8 to 5, 5 to 8, 47 to 29)) {
            val frame = randomFrame(random, w, h)
            val reference = referenceDetector(frame, 32)
            val letterbox = CanonicalImageOps.detectorInputInto(frame, 32, destination)
            assertEquals(reference.second, letterbox)
            assertExact(reference.first, destination)
        }
    }

    @Test fun `classifier destination is completely overwritten when source and crop change`() {
        val random = Random(7493)
        val destination = FloatArray(23 * 23 * 3) { Float.NaN }
        repeat(25) {
            val frame = randomFrame(random, random.nextInt(6, 56), random.nextInt(6, 61))
            val box = if (it % 2 == 0) null else PixelBox(1f, 2f, frame.width - 1f, frame.height - 1f)
            assertTrue(CanonicalImageOps.classifierInputInto(frame, 23, destination, box))
            assertExact(referenceClassifier(frame, 23, box)!!, destination)
        }
    }

    @Test fun `detector padding and half even letterbox geometry retain the contract`() {
        val frame = RgbFrame(8, 5, IntArray(40) { 0xFFFFFFFF.toInt() })
        val destination = FloatArray(4 * 4 * 3)
        val letterbox = CanonicalImageOps.detectorInputInto(frame, 4, destination)
        assertEquals(CanonicalImageOps.Letterbox(0.5, 0, 1, 4), letterbox)
        for (i in 0 until 4 * 3) {
            assertEquals(114f / 255f, destination[i], 0f)
            assertEquals(114f / 255f, destination[3 * 4 * 3 + i], 0f)
        }
        for (i in 4 * 3 until 3 * 4 * 3) assertEquals(1f, destination[i], 0f)
    }

    @Test fun `classifier RGB normalization retains channel order and exact divisions`() {
        val frame = RgbFrame(3, 5, IntArray(15) { 0xFF406080.toInt() })
        val destination = FloatArray(8 * 8 * 3)
        assertTrue(CanonicalImageOps.classifierInputInto(frame, 8, destination))
        for (i in destination.indices step 3) {
            assertEquals((64f / 255f - 0.485f) / 0.229f, destination[i], 0f)
            assertEquals((96f / 255f - 0.456f) / 0.224f, destination[i + 1], 0f)
            assertEquals((128f / 255f - 0.406f) / 0.225f, destination[i + 2], 0f)
        }
    }

    @Test fun `allocation APIs keep independently owned outputs across subsequent calls`() {
        val frame = randomFrame(Random(4234), 37, 23)
        val classifier = CanonicalImageOps.classifierInput(frame, 16)!!
        val classifierSnapshot = classifier.copyOf()
        val classifierNext = CanonicalImageOps.classifierInput(frame, 16)!!
        assertNotSame(classifier, classifierNext)
        classifierNext.fill(-99f)
        assertExact(classifierSnapshot, classifier)
        val detector = CanonicalImageOps.detectorInput(frame, 32).first
        val detectorSnapshot = detector.copyOf()
        val detectorNext = CanonicalImageOps.detectorInput(frame, 32).first
        assertNotSame(detector, detectorNext)
        detectorNext.fill(-99f)
        assertExact(detectorSnapshot, detector)
    }

    @Test fun `changing many dimensions cannot reuse incorrect resampling weights`() {
        val frame = randomFrame(Random(629), 61, 53)
        for (size in 1..40) {
            val destination = FloatArray(size * size * 3)
            assertTrue(CanonicalImageOps.classifierInputInto(frame, size, destination))
            assertExact(referenceClassifier(frame, size)!!, destination)
        }
        val destination = FloatArray(7 * 7 * 3)
        assertTrue(CanonicalImageOps.classifierInputInto(frame, 7, destination))
        assertExact(referenceClassifier(frame, 7)!!, destination)
    }

    @Test fun `concurrent callers keep resampling scratch and cached dimensions isolated`() {
        val executor = Executors.newFixedThreadPool(2)
        try {
            val futures = executor.invokeAll((0..1).map { worker ->
                Callable {
                    val random = Random(724 + worker)
                    repeat(20) {
                        val frame = randomFrame(random, random.nextInt(9, 72), random.nextInt(7, 58))
                        val size = 7 + worker * 13
                        val destination = FloatArray(size * size * 3)
                        assertTrue(CanonicalImageOps.classifierInputInto(frame, size, destination))
                        assertExact(referenceClassifier(frame, size)!!, destination)
                        val reference = referenceDetector(frame, size)
                        assertEquals(reference.second, CanonicalImageOps.detectorInputInto(frame, size, destination))
                        assertExact(reference.first, destination)
                    }
                }
            })
            futures.forEach { it.get() }
        } finally {
            executor.shutdownNow()
        }
    }

    @Test fun `reusable tensors reject invalid size and incorrectly sized storage`() {
        val frame = randomFrame(Random(102), 6, 9)
        for (destination in listOf(FloatArray(3), FloatArray(8 * 8 * 3 + 1))) {
            assertThrows(IllegalArgumentException::class.java) {
                CanonicalImageOps.classifierInputInto(frame, 8, destination)
            }
            assertThrows(IllegalArgumentException::class.java) {
                CanonicalImageOps.detectorInputInto(frame, 8, destination)
            }
        }
        assertThrows(IllegalArgumentException::class.java) {
            CanonicalImageOps.classifierInputInto(frame, 0, FloatArray(0))
        }
        assertThrows(IllegalArgumentException::class.java) {
            CanonicalImageOps.detectorInputInto(frame, -1, FloatArray(3))
        }
        assertThrows(IllegalArgumentException::class.java) {
            CanonicalImageOps.classifierInputInto(frame, Int.MAX_VALUE, FloatArray(0))
        }
        assertThrows(IllegalArgumentException::class.java) {
            CanonicalImageOps.detectorInputInto(frame, Int.MAX_VALUE, FloatArray(0))
        }
    }

    @Test fun `reusable APIs are bit exact against the preserved reference on bundled golden images`() {
        val dir = requireNotNull(File(javaClass.classLoader!!.getResource("golden/golden_expected.json")!!.toURI()).parentFile)
        val manifest = VisionManifest.parse(File("../app/src/main/assets/model_manifest.json").readText())
        val images = dir.listFiles { file -> file.extension == "png" }!!.sortedBy { it.name }
        assertTrue(images.size >= 10)
        val detectorDestination = FloatArray(manifest.detector.inputSize * manifest.detector.inputSize * 3)
        for (image in images) {
            val frame = TestPng.read(image)
            for (member in manifest.members) {
                val destination = FloatArray(member.inputSize * member.inputSize * 3)
                assertTrue(CanonicalImageOps.classifierInputInto(frame, member.inputSize, destination))
                assertExact(referenceClassifier(frame, member.inputSize)!!, destination)
            }
            val reference = referenceDetector(frame, manifest.detector.inputSize)
            assertEquals(reference.second,
                CanonicalImageOps.detectorInputInto(frame, manifest.detector.inputSize, detectorDestination))
            assertExact(reference.first, detectorDestination)
        }
    }

    private fun randomFrame(random: Random, w: Int, h: Int) =
        RgbFrame(w, h, IntArray(w * h) { random.nextInt() })

    private fun assertExact(expected: FloatArray, actual: FloatArray) {
        assertArrayEquals(expected, actual, 0f)
    }

    private fun referenceClassifier(
        frame: RgbFrame, size: Int, box: PixelBox? = null, pad: Double = 0.10,
    ): FloatArray? {
        var l = 0
        var t = 0
        var w = frame.width
        var h = frame.height
        if (box != null) {
            val x1 = min(box.left, box.right).toDouble()
            val x2 = max(box.left, box.right).toDouble()
            val y1 = min(box.top, box.bottom).toDouble()
            val y2 = max(box.top, box.bottom).toDouble()
            l = max(0, floor(x1 - pad * (x2 - x1)).toInt())
            t = max(0, floor(y1 - pad * (y2 - y1)).toInt())
            w = min(frame.width, ceil(x2 + pad * (x2 - x1)).toInt()) - l
            h = min(frame.height, ceil(y2 + pad * (y2 - y1)).toInt()) - t
            if (w < 2 || h < 2) return null
        }
        val side = min(w, h)
        val rgb = referenceResize(frame, l + (w - side) / 2, t + (h - side) / 2, side, side, size, size)
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)
        for (i in rgb.indices) rgb[i] = (rgb[i] / 255f - mean[i % 3]) / std[i % 3]
        return rgb
    }

    private fun referenceDetector(frame: RgbFrame, size: Int): Pair<FloatArray, CanonicalImageOps.Letterbox> {
        val k = min(size.toDouble() / frame.width, size.toDouble() / frame.height)
        val nw = Math.rint(frame.width * k).toInt().coerceIn(1, size)
        val nh = Math.rint(frame.height * k).toInt().coerceIn(1, size)
        val padX = (size - nw) / 2
        val padY = (size - nh) / 2
        val out = FloatArray(size * size * 3) { 114f }
        val resized = referenceResize(frame, 0, 0, frame.width, frame.height, nw, nh)
        for (y in 0 until nh) {
            System.arraycopy(resized, y * nw * 3, out, ((y + padY) * size + padX) * 3, nw * 3)
        }
        for (i in out.indices) out[i] /= 255f
        return out to CanonicalImageOps.Letterbox(k, padX, padY, size)
    }

    private class ReferenceAxis(val start: IntArray, val taps: Int, val weights: FloatArray)

    private fun referenceAxis(nIn: Int, nOut: Int): ReferenceAxis {
        val s = nIn.toDouble() / nOut
        val taps = if (s > 1.0) ceil(s).toInt() + 1 else 2
        val start = IntArray(nOut)
        val weights = FloatArray(nOut * taps)
        for (i in 0 until nOut) {
            if (s > 1.0) {
                val a = i * s
                val b = (i + 1) * s
                val j0 = floor(a).toInt()
                val j1 = min(nIn, ceil(b).toInt())
                start[i] = j0
                for (j in j0 until j1) {
                    weights[i * taps + (j - j0)] =
                        ((min(b, j + 1.0) - max(a, j.toDouble())) / s).toFloat()
                }
            } else {
                val x = min(max((i + 0.5) * s - 0.5, 0.0), (nIn - 1).toDouble())
                val j0 = floor(x).toInt()
                val f = x - j0
                start[i] = j0
                if (j0 + 1 <= nIn - 1) {
                    weights[i * taps] = (1.0 - f).toFloat()
                    weights[i * taps + 1] = f.toFloat()
                } else weights[i * taps] = 1f
            }
        }
        return ReferenceAxis(start, taps, weights)
    }

    private fun referenceResize(
        frame: RgbFrame, cropL: Int, cropT: Int, cropW: Int, cropH: Int, outW: Int, outH: Int,
    ): FloatArray {
        val wy = referenceAxis(cropH, outH)
        val wx = referenceAxis(cropW, outW)
        val tmp = FloatArray(outH * cropW * 3)
        for (oy in 0 until outH) {
            val j0 = wy.start[oy]
            for (t in 0 until wy.taps) {
                val wgt = wy.weights[oy * wy.taps + t]
                if (wgt == 0f) continue
                val sy = j0 + t
                if (sy >= cropH) break
                val rowBase = (cropT + sy) * frame.width + cropL
                val outBase = oy * cropW * 3
                for (x in 0 until cropW) {
                    val c = frame.pixels[rowBase + x]
                    val o = outBase + x * 3
                    tmp[o] += wgt * ((c shr 16) and 0xFF)
                    tmp[o + 1] += wgt * ((c shr 8) and 0xFF)
                    tmp[o + 2] += wgt * (c and 0xFF)
                }
            }
        }
        val out = FloatArray(outH * outW * 3)
        for (oy in 0 until outH) {
            val inBase = oy * cropW * 3
            val outBase = oy * outW * 3
            for (ox in 0 until outW) {
                val j0 = wx.start[ox]
                var r = 0f
                var g = 0f
                var b = 0f
                for (t in 0 until wx.taps) {
                    val wgt = wx.weights[ox * wx.taps + t]
                    if (wgt == 0f) continue
                    val sx = j0 + t
                    if (sx >= cropW) break
                    val i = inBase + sx * 3
                    r += wgt * tmp[i]
                    g += wgt * tmp[i + 1]
                    b += wgt * tmp[i + 2]
                }
                val o = outBase + ox * 3
                out[o] = r
                out[o + 1] = g
                out[o + 2] = b
            }
        }
        return out
    }
}
