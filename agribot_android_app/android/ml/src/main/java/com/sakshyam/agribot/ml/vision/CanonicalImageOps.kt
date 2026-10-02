package com.sakshyam.agribot.ml.vision

import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

/**
 * Packed-ARGB image (alpha ignored). Pure Kotlin so preprocessing is unit-testable on the JVM.
 */
class RgbFrame(val width: Int, val height: Int, val pixels: IntArray) {
    init {
        require(width > 0 && height > 0) { "frame must be non-empty" }
        require(pixels.size == width * height) { "pixels must be width * height" }
    }
}

/** Axis-aligned box in source-image pixels. */
data class PixelBox(val left: Float, val top: Float, val right: Float, val bottom: Float) {
    val width: Float get() = right - left
    val height: Float get() = bottom - top
    val centerX: Float get() = (left + right) / 2f
    val centerY: Float get() = (top + bottom) / 2f
    val area: Float get() = max(0f, width) * max(0f, height)

    fun iou(other: PixelBox): Float {
        val ix = max(0f, min(right, other.right) - max(left, other.left))
        val iy = max(0f, min(bottom, other.bottom) - max(top, other.top))
        val inter = ix * iy
        val union = area + other.area - inter
        return if (union > 0f) inter / union else 0f
    }
}

/**
 * Sparse 1-D resampling weights, identical to `axis_weights` in golden/canon_reference.py:
 * downscale (s > 1) is an exact area average over [i*s, (i+1)*s); upscale is bilinear with
 * half-pixel centres, clamped at the borders.
 */
internal class AxisWeights private constructor(
    val start: IntArray,
    val taps: Int,
    val weights: FloatArray,
) {
    companion object {
        fun create(nIn: Int, nOut: Int): AxisWeights {
            require(nIn > 0 && nOut > 0)
            val s = nIn.toDouble() / nOut
            val taps = if (s > 1.0) ceil(s).toInt() + 1 else 2
            val start = IntArray(nOut)
            val w = FloatArray(nOut * taps)
            for (i in 0 until nOut) {
                if (s > 1.0) {
                    val a = i * s
                    val b = (i + 1) * s
                    val j0 = floor(a).toInt()
                    val j1 = min(nIn, ceil(b).toInt())
                    start[i] = j0
                    for (j in j0 until j1) {
                        w[i * taps + (j - j0)] = ((min(b, j + 1.0) - max(a, j.toDouble())) / s).toFloat()
                    }
                } else {
                    val x = min(max((i + 0.5) * s - 0.5, 0.0), (nIn - 1).toDouble())
                    val j0 = floor(x).toInt()
                    val f = x - j0
                    start[i] = j0
                    if (j0 + 1 <= nIn - 1) {
                        w[i * taps] = (1.0 - f).toFloat()
                        w[i * taps + 1] = f.toFloat()
                    } else {
                        w[i * taps] = 1f
                    }
                }
            }
            return AxisWeights(start, taps, w)
        }
    }
}

object CanonicalImageOps {
    val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    const val LETTERBOX_PAD = 114f
    private val scratch = ThreadLocal.withInitial { FloatArray(0) }
    private const val AXIS_CACHE_ENTRIES = 32
    private val axisCache = ThreadLocal.withInitial { LinkedHashMap<Long, AxisWeights>(AXIS_CACHE_ENTRIES, 0.75f, true) }

    /** Tables are immutable after construction and bounded independently on each calling thread. */
    private fun axisWeights(nIn: Int, nOut: Int): AxisWeights {
        val key = (nIn.toLong() shl 32) or (nOut.toLong() and 0xFFFFFFFFL)
        val cache = axisCache.get()!!
        cache[key]?.let { return it }
        val weights = AxisWeights.create(nIn, nOut)
        cache[key] = weights
        if (cache.size > AXIS_CACHE_ENTRIES) cache.remove(cache.keys.iterator().next())
        return weights
    }

    private fun tensorElements(width: Int, height: Int): Int {
        require(width > 0 && height > 0) { "tensor dimensions must be positive" }
        val pixels = width.toLong() * height
        require(pixels <= Int.MAX_VALUE / 3) { "tensor dimensions are too large" }
        return pixels.toInt() * 3
    }

    /**
     * Resizes the [cropW]x[cropH] region at ([cropL],[cropT]) to [outW]x[outH].
     * Returns interleaved RGB floats in 0..255, row-major (HWC).
     */
    fun resize(frame: RgbFrame, cropL: Int, cropT: Int, cropW: Int, cropH: Int, outW: Int, outH: Int): FloatArray {
        val out = FloatArray(tensorElements(outW, outH))
        resizeInto(frame, cropL, cropT, cropW, cropH, outW, outH, out, 0, outW * 3)
        return out
    }

    /** Writes directly to a packed tensor or its letterboxed image region, without an output copy. */
    private fun resizeInto(
        frame: RgbFrame, cropL: Int, cropT: Int, cropW: Int, cropH: Int, outW: Int, outH: Int,
        destination: FloatArray, destinationOffset: Int, destinationRowStride: Int,
    ) {
        require(cropL >= 0 && cropT >= 0 && cropW > 0 && cropH > 0)
        require(cropL + cropW <= frame.width && cropT + cropH <= frame.height) { "crop outside frame" }
        val wy = axisWeights(cropH, outH)
        val wx = axisWeights(cropW, outW)
        val tmp = resizeVertical(frame, cropL, cropT, cropW, cropH, outH, wy)
        resizeHorizontal(tmp, cropW, outW, outH, wx, destination, destinationOffset, destinationRowStride)
    }

    private fun resizeVertical(
        frame: RgbFrame, cropL: Int, cropT: Int, cropW: Int, cropH: Int, outH: Int, wy: AxisWeights,
    ): FloatArray {
        // Vertical pass: (outH, cropW, 3) into a per-thread scratch buffer (avoids MBs of garbage per frame).
        val need = tensorElements(cropW, outH)
        val tmp = scratch.get()!!.let { if (it.size >= need) it else FloatArray(need).also { a -> scratch.set(a) } }
        java.util.Arrays.fill(tmp, 0, need, 0f)
        val px = frame.pixels
        val stride = frame.width
        for (oy in 0 until outH) {
            val j0 = wy.start[oy]
            for (t in 0 until wy.taps) {
                val wgt = wy.weights[oy * wy.taps + t]
                if (wgt == 0f) continue
                val sy = j0 + t
                if (sy >= cropH) break
                val rowBase = (cropT + sy) * stride + cropL
                val outBase = oy * cropW * 3
                for (x in 0 until cropW) {
                    val c = px[rowBase + x]
                    val o = outBase + x * 3
                    tmp[o] += wgt * ((c shr 16) and 0xFF)
                    tmp[o + 1] += wgt * ((c shr 8) and 0xFF)
                    tmp[o + 2] += wgt * (c and 0xFF)
                }
            }
        }
        return tmp
    }

    private fun resizeHorizontal(
        tmp: FloatArray, cropW: Int, outW: Int, outH: Int, wx: AxisWeights,
        destination: FloatArray, destinationOffset: Int, destinationRowStride: Int,
    ) {
        // Horizontal pass: (outH, outW, 3)
        for (oy in 0 until outH) {
            val inBase = oy * cropW * 3
            val outBase = destinationOffset + oy * destinationRowStride
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
                destination[o] = r
                destination[o + 1] = g
                destination[o + 2] = b
            }
        }
    }

    /** Integer crop rectangle (l, t, r, b) for [box] grown by [pad] of its size, clamped to the frame. */
    fun cropRect(frameW: Int, frameH: Int, box: PixelBox, pad: Double): IntArray {
        // Double precision, like the Python reference (avoids 1-px shifts at integer boundaries).
        val x1 = min(box.left, box.right).toDouble()
        val x2 = max(box.left, box.right).toDouble()
        val y1 = min(box.top, box.bottom).toDouble()
        val y2 = max(box.top, box.bottom).toDouble()
        val p = pad
        val bw = x2 - x1
        val bh = y2 - y1
        val l = max(0, floor(x1 - p * bw).toInt())
        val t = max(0, floor(y1 - p * bh).toInt())
        val r = min(frameW, ceil(x2 + p * bw).toInt())
        val b = min(frameH, ceil(y2 + p * bh).toInt())
        return intArrayOf(l, t, r, b)
    }

    /**
     * Classifier input, NHWC float32: crop (optional, +pad) -> centred square of the short side ->
     * canonical resize to size x size -> /255 -> ImageNet mean/std.
     * Returns null when the crop is degenerate.
     */
    fun classifierInput(frame: RgbFrame, size: Int, box: PixelBox? = null, pad: Double = 0.10): FloatArray? {
        val out = FloatArray(tensorElements(size, size))
        return if (classifierInputInto(frame, size, out, box, pad)) out else null
    }

    /**
     * Same classifier contract, writing into caller-owned storage. A degenerate crop returns false
     * without changing [destination]; successful calls overwrite the entire tensor.
     */
    fun classifierInputInto(
        frame: RgbFrame, size: Int, destination: FloatArray, box: PixelBox? = null, pad: Double = 0.10,
    ): Boolean {
        require(destination.size == tensorElements(size, size)) { "destination must be size * size * 3" }
        var l = 0
        var t = 0
        var w = frame.width
        var h = frame.height
        if (box != null) {
            val r = cropRect(frame.width, frame.height, box, pad)
            l = r[0]; t = r[1]; w = r[2] - r[0]; h = r[3] - r[1]
            if (w < 2 || h < 2) return false
        }
        val side = min(w, h)
        val ox = l + (w - side) / 2
        val oy = t + (h - side) / 2
        resizeInto(frame, ox, oy, side, side, size, size, destination, 0, size * 3)
        // Preserve division/subtraction order while avoiding a channel modulo per element.
        val mean = IMAGENET_MEAN
        val std = IMAGENET_STD
        for (i in destination.indices step 3) {
            destination[i] = (destination[i] / 255f - mean[0]) / std[0]
            destination[i + 1] = (destination[i + 1] / 255f - mean[1]) / std[1]
            destination[i + 2] = (destination[i + 2] / 255f - mean[2]) / std[2]
        }
        return true
    }

    /** Letterbox geometry: model px = frame px * scale + pad. */
    data class Letterbox(val scale: Double, val padX: Int, val padY: Int, val inputSize: Int) {
        fun toFrameX(modelX: Float): Double = (modelX - padX) / scale
        fun toFrameY(modelY: Float): Double = (modelY - padY) / scale
        fun toFrame(modelX: Float, modelY: Float): Pair<Float, Float> = toFrameX(modelX).toFloat() to toFrameY(modelY).toFloat()
    }

    /** Detector input, NHWC float32 in 0..1 with 114-grey centred letterbox. */
    fun detectorInput(frame: RgbFrame, size: Int): Pair<FloatArray, Letterbox> {
        val out = FloatArray(tensorElements(size, size))
        return out to detectorInputInto(frame, size, out)
    }

    /** Same detector contract without allocating or copying a temporary resized image. */
    fun detectorInputInto(frame: RgbFrame, size: Int, destination: FloatArray): Letterbox {
        require(destination.size == tensorElements(size, size)) { "destination must be size * size * 3" }
        val k = min(size.toDouble() / frame.width, size.toDouble() / frame.height)
        val nw = Math.rint(frame.width * k).toInt().coerceIn(1, size)   // rint == Python round (half-even)
        val nh = Math.rint(frame.height * k).toInt().coerceIn(1, size)
        val padX = (size - nw) / 2
        val padY = (size - nh) / 2
        java.util.Arrays.fill(destination, LETTERBOX_PAD)
        resizeInto(frame, 0, 0, frame.width, frame.height, nw, nh,
            destination, (padY * size + padX) * 3, size * 3)
        for (i in destination.indices) destination[i] = destination[i] / 255f
        return Letterbox(k, padX, padY, size)
    }
}
