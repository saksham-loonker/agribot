package com.sakshyam.agribot.camera

import kotlin.math.max
import kotlin.math.min

object Yuv420FrameConverter {
    fun toRgbPixels(
        width: Int,
        height: Int,
        y: ByteArray,
        u: ByteArray,
        v: ByteArray,
        yRowStride: Int,
        uRowStride: Int,
        vRowStride: Int,
        uPixelStride: Int,
        vPixelStride: Int,
        cropLeft: Int = 0,
        cropTop: Int = 0,
    ): IntArray {
        if (width <= 0 || height <= 0 ||
            width.toLong() * height.toLong() > MAX_RGB_PIXELS ||
            yRowStride <= 0 || uRowStride <= 0 || vRowStride <= 0 ||
            uPixelStride <= 0 || vPixelStride <= 0
        ) return IntArray(0)
        val safeCropLeft = cropLeft.coerceAtLeast(0)
        val safeCropTop = cropTop.coerceAtLeast(0)
        val out = IntArray(width * height)
        for (row in 0 until height) {
            val sourceRow = row + safeCropTop
            val chromaRow = sourceRow / 2
            for (col in 0 until width) {
                val sourceCol = col + safeCropLeft
                val chromaCol = sourceCol / 2
                // CameraX normally supplies valid plane buffers, but a
                // malformed vendor frame must not crash the analyzer thread.
                // Missing luma becomes dark (and is rejected by the quality
                // gate); missing chroma becomes neutral rather than shifting
                // the colour channels unpredictably.
                val yValue = y.safePlaneByte(sourceRow, sourceCol, yRowStride, 1) ?: return@toRgbPixels IntArray(0)
                val uValue = u.safePlaneByte(chromaRow, chromaCol, uRowStride, uPixelStride) ?: 128
                val vValue = v.safePlaneByte(chromaRow, chromaCol, vRowStride, vPixelStride) ?: 128
                out[row * width + col] = yuvToRgb(yValue, uValue, vValue)
            }
        }
        return out
    }

    private fun yuvToRgb(y: Int, u: Int, v: Int): Int {
        val c = max(0, y - 16)
        val d = u - 128
        val e = v - 128
        val red = clamp((298 * c + 409 * e + 128) shr 8)
        val green = clamp((298 * c - 100 * d - 208 * e + 128) shr 8)
        val blue = clamp((298 * c + 516 * d + 128) shr 8)
        return (red shl 16) or (green shl 8) or blue
    }

    private fun clamp(value: Int): Int = min(255, max(0, value))

    private fun ByteArray.safePlaneByte(row: Int, column: Int, rowStride: Int, pixelStride: Int): Int? {
        val index = row.toLong() * rowStride.toLong() + column.toLong() * pixelStride.toLong()
        if (index < 0L || index >= size.toLong()) return null
        return this[index.toInt()].toInt().and(0xff)
    }

    private const val MAX_RGB_PIXELS = 2_000_000L
}
