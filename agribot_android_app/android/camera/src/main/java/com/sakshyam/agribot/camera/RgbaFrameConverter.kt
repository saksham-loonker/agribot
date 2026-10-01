package com.sakshyam.agribot.camera

/**
 * Converts a CameraX RGBA_8888 plane (row stride may exceed width * 4) into an upright packed-ARGB
 * array: crops to the viewport rectangle first, then rotates clockwise by [rotationDegrees] so the
 * result matches what the user sees in the preview.
 */
object RgbaFrameConverter {
    class Upright(val width: Int, val height: Int, val argb: IntArray)

    fun toUpright(
        rgba: ByteArray,
        rowStride: Int,
        cropLeft: Int,
        cropTop: Int,
        cropWidth: Int,
        cropHeight: Int,
        rotationDegrees: Int,
    ): Upright {
        require(cropWidth > 0 && cropHeight > 0)
        require(rotationDegrees == 0 || rotationDegrees == 90 || rotationDegrees == 180 || rotationDegrees == 270)
        val rotated = rotationDegrees == 90 || rotationDegrees == 270
        val outW = if (rotated) cropHeight else cropWidth
        val outH = if (rotated) cropWidth else cropHeight
        val out = IntArray(outW * outH)
        for (y in 0 until cropHeight) {
            val row = (cropTop + y) * rowStride + cropLeft * 4
            for (x in 0 until cropWidth) {
                val i = row + x * 4
                val argb = (0xFF shl 24) or ((rgba[i].toInt() and 0xFF) shl 16) or
                    ((rgba[i + 1].toInt() and 0xFF) shl 8) or (rgba[i + 2].toInt() and 0xFF)
                val (ox, oy) = when (rotationDegrees) {
                    90 -> (cropHeight - 1 - y) to x
                    180 -> (cropWidth - 1 - x) to (cropHeight - 1 - y)
                    270 -> y to (cropWidth - 1 - x)
                    else -> x to y
                }
                out[oy * outW + ox] = argb
            }
        }
        return Upright(outW, outH, out)
    }
}
