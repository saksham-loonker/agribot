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

    /**
     * Resizes the [cropW]x[cropH] region at ([cropL],[cropT]) to [outW]x[outH].
     * Returns interleaved RGB floats in 0..255, row-major (HWC).
     */
    fun resize(frame: RgbFrame, cropL: Int, cropT: Int, cropW: Int, cropH: Int, outW: Int, outH: Int): FloatArray {
        require(cropL >= 0 && cropT >= 0 && cropW > 0 && cropH > 0)
        require(cropL + cropW <= frame.width && cropT + cropH <= frame.height) { "crop outside frame" }
        val wy = AxisWeights.create(cropH, outH)
        val wx = AxisWeights.create(cropW, outW)
        // Vertical pass: (outH, cropW, 3)
        val tmp = FloatArray(outH * cropW * 3)
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
        // Horizontal pass: (outH, outW, 3)
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

    /** Integer crop rectangle (l, t, r, b) for [box] grown by [pad] of its size, clamped to the frame. */
    fun cropRect(frameW: Int, frameH: Int, box: PixelBox, pad: Float): IntArray {
        val x1 = min(box.left, box.right)
        val x2 = max(box.left, box.right)
        val y1 = min(box.top, box.bottom)
        val y2 = max(box.top, box.bottom)
        val bw = x2 - x1
        val bh = y2 - y1
        val l = max(0, floor(x1 - pad * bw).toInt())
        val t = max(0, floor(y1 - pad * bh).toInt())
        val r = min(frameW, ceil(x2 + pad * bw).toInt())
        val b = min(frameH, ceil(y2 + pad * bh).toInt())
        return intArrayOf(l, t, r, b)
    }

    /**
     * Classifier input, NHWC float32: crop (optional, +pad) -> centred square of the short side ->
     * canonical resize to size x size -> /255 -> ImageNet mean/std.
     * Returns null when the crop is degenerate.
     */
    fun classifierInput(frame: RgbFrame, size: Int, box: PixelBox? = null, pad: Float = 0.10f): FloatArray? {
        var l = 0
        var t = 0
        var w = frame.width
        var h = frame.height
        if (box != null) {
            val r = cropRect(frame.width, frame.height, box, pad)
            l = r[0]; t = r[1]; w = r[2] - r[0]; h = r[3] - r[1]
            if (w < 2 || h < 2) return null
        }
        val side = min(w, h)
        val ox = l + (w - side) / 2
        val oy = t + (h - side) / 2
        val rgb = resize(frame, ox, oy, side, side, size, size)
        for (i in rgb.indices) {
            val c = i % 3
            rgb[i] = (rgb[i] / 255f - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        }
        return rgb
    }

    /** Letterbox geometry: model px = frame px * scale + pad. */
    data class Letterbox(val scale: Float, val padX: Int, val padY: Int, val inputSize: Int) {
        fun toFrame(modelX: Float, modelY: Float): Pair<Float, Float> = ((modelX - padX) / scale) to ((modelY - padY) / scale)
    }

    /** Detector input, NHWC float32 in 0..1 with 114-grey centred letterbox. */
    fun detectorInput(frame: RgbFrame, size: Int): Pair<FloatArray, Letterbox> {
        val k = min(size.toDouble() / frame.width, size.toDouble() / frame.height)
        val nw = Math.rint(frame.width * k).toInt().coerceIn(1, size)   // rint == Python round (half-even)
        val nh = Math.rint(frame.height * k).toInt().coerceIn(1, size)
        val padX = (size - nw) / 2
        val padY = (size - nh) / 2
        val out = FloatArray(size * size * 3) { LETTERBOX_PAD }
        val resized = resize(frame, 0, 0, frame.width, frame.height, nw, nh)
        for (y in 0 until nh) {
            System.arraycopy(resized, y * nw * 3, out, ((y + padY) * size + padX) * 3, nw * 3)
        }
        for (i in out.indices) out[i] = out[i] / 255f
        return out to Letterbox(k.toFloat(), padX, padY, size)
    }
}
