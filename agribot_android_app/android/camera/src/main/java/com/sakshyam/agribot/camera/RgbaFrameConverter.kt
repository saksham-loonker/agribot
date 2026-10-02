package com.sakshyam.agribot.camera

import java.nio.ByteBuffer
import java.nio.ByteOrder

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
    ): Upright = toUpright(ByteBuffer.wrap(rgba), rowStride, cropLeft, cropTop, cropWidth, cropHeight, rotationDegrees)

    /**
     * Reads directly from the plane while its ImageProxy remains open. Index zero is the buffer's
     * frame origin (including for a slice); its position, limit, mark and byte order are unchanged.
     * Every result owns a new ARGB array, so retaining a frame for evidence cannot corrupt it when
     * CameraX releases or reuses the source plane.
     */
    fun toUpright(
        rgba: ByteBuffer,
        rowStride: Int,
        cropLeft: Int,
        cropTop: Int,
        cropWidth: Int,
        cropHeight: Int,
        rotationDegrees: Int,
    ): Upright {
        require(cropLeft >= 0 && cropTop >= 0) { "Crop origin must be nonnegative" }
        require(cropWidth > 0 && cropHeight > 0) { "Crop dimensions must be positive" }
        require(rotationDegrees == 0 || rotationDegrees == 90 || rotationDegrees == 180 || rotationDegrees == 270) {
            "Rotation must be 0, 90, 180 or 270 degrees"
        }
        val rowEnd = (cropLeft.toLong() + cropWidth) * 4
        require(rowStride > 0 && rowEnd <= rowStride) { "Crop exceeds the RGBA row stride" }
        val requiredBytes = (cropTop.toLong() + cropHeight - 1) * rowStride + rowEnd
        require(requiredBytes <= rgba.limit()) { "RGBA buffer does not contain the complete crop" }
        require(cropWidth.toLong() * cropHeight <= Int.MAX_VALUE) { "Crop is too large" }

        val rotated = rotationDegrees == 90 || rotationDegrees == 270
        val outW = if (rotated) cropHeight else cropWidth
        val outH = if (rotated) cropWidth else cropHeight
        val out = IntArray(outW * outH)
        // Each row starts one output stride further on; each pixel uses a constant output stride.
        // Selecting these once avoids allocating a Pair and branching on rotation for every pixel.
        val firstOutput = when (rotationDegrees) {
            90 -> cropHeight - 1
            180 -> out.size - 1
            270 -> (cropWidth - 1) * outW
            else -> 0
        }
        val outputRowStep = when (rotationDegrees) {
            90 -> -1
            180 -> -outW
            270 -> 1
            else -> outW
        }
        val outputPixelStep = when (rotationDegrees) {
            90 -> outW
            180 -> -1
            270 -> -outW
            else -> 1
        }
        val source = rgba.duplicate().order(ByteOrder.BIG_ENDIAN)
        for (y in 0 until cropHeight) {
            var sourceIndex = (cropTop + y) * rowStride + cropLeft * 4
            var outputIndex = firstOutput + y * outputRowStep
            for (x in 0 until cropWidth) {
                // CameraX exposes R, G, B, A bytes. Read all four once and preserve opaque output.
                out[outputIndex] = (0xFF shl 24) or (source.getInt(sourceIndex) ushr 8)
                sourceIndex += 4
                outputIndex += outputPixelStep
            }
        }
        return Upright(outW, outH, out)
    }
}
