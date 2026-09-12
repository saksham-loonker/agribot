package com.sakshyam.agribot.camera

import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.sakshyam.agribot.domain.model.AnalysisFrame
import java.nio.ByteBuffer

class SideScanAnalyzer(
    targetFps: Int,
    private val onFrame: (AnalysisFrame) -> Unit,
) : ImageAnalysis.Analyzer {
    private val sampler = FrameSampler(targetFps)
    private val lock = Any()

    fun updateTargetFps(targetFps: Int) {
        sampler.updateTargetFps(targetFps)
    }

    override fun analyze(image: ImageProxy) {
        try {
            synchronized(lock) {
                if (!sampler.shouldAccept(image.imageInfo.timestamp)) return
                runCatching { image.toAnalysisFrame() }
                    .getOrNull()
                    ?.let(onFrame)
            }
        } finally {
            image.close()
        }
    }

    private fun ImageProxy.toAnalysisFrame(): AnalysisFrame {
        val crop = cropRect
        require(crop.left >= 0 && crop.top >= 0 && crop.right > crop.left && crop.bottom > crop.top)
        require(crop.width().toLong() * crop.height().toLong() <= MAX_ANALYSIS_PIXELS)
        require(planes.size >= 3)
        val yBytes = planes[0].buffer.toByteArray(MAX_PLANE_BYTES)
        val uBytes = planes[1].buffer.toByteArray(MAX_PLANE_BYTES)
        val vBytes = planes[2].buffer.toByteArray(MAX_PLANE_BYTES)
        val rotated = rotateRgbPixels(
            width = crop.width(),
            height = crop.height(),
            pixels = Yuv420FrameConverter.toRgbPixels(
                width = crop.width(),
                height = crop.height(),
                y = yBytes,
                u = uBytes,
                v = vBytes,
                yRowStride = planes[0].rowStride,
                uRowStride = planes[1].rowStride,
                vRowStride = planes[2].rowStride,
                uPixelStride = planes[1].pixelStride,
                vPixelStride = planes[2].pixelStride,
                cropLeft = crop.left,
                cropTop = crop.top,
            ),
            degrees = imageInfo.rotationDegrees,
        )
        require(rotated.pixels.isNotEmpty())
        return AnalysisFrame(
            width = rotated.width,
            height = rotated.height,
            rotationDegrees = imageInfo.rotationDegrees,
            timestampNanos = imageInfo.timestamp,
            rgbPixels = rotated.pixels,
            jpegBytes = JpegEvidenceEncoder.encode(rotated.width, rotated.height, rotated.pixels),
        )
    }

    private data class RotatedRgb(
        val width: Int,
        val height: Int,
        val pixels: IntArray,
    )

    private fun rotateRgbPixels(width: Int, height: Int, pixels: IntArray, degrees: Int): RotatedRgb {
        if (width <= 0 || height <= 0 || width.toLong() * height.toLong() > MAX_ANALYSIS_PIXELS ||
            pixels.size.toLong() != width.toLong() * height.toLong()
        ) return RotatedRgb(0, 0, IntArray(0))
        when ((degrees % 360 + 360) % 360) {
            0 -> return RotatedRgb(width, height, pixels)
            90 -> {
                val out = IntArray(height * width)
                for (y in 0 until width) {
                    for (x in 0 until height) {
                        val sourceY = height - 1 - x
                        val sourceX = y
                        out[y * height + x] = pixels[sourceY * width + sourceX]
                    }
                }
                return RotatedRgb(height, width, out)
            }
            180 -> {
                val out = IntArray(width * height)
                for (y in 0 until height) {
                    for (x in 0 until width) {
                        out[y * width + x] = pixels[(height - 1 - y) * width + (width - 1 - x)]
                    }
                }
                return RotatedRgb(width, height, out)
            }
            270 -> {
                val out = IntArray(height * width)
                for (y in 0 until width) {
                    for (x in 0 until height) {
                        val sourceY = x
                        val sourceX = width - 1 - y
                        out[y * height + x] = pixels[sourceY * width + sourceX]
                    }
                }
                return RotatedRgb(height, width, out)
            }
            else -> return RotatedRgb(width, height, pixels)
        }
    }

    fun rotateForTest(pixels: IntArray, width: Int, height: Int, degrees: Int): IntArray =
        rotateRgbPixels(width, height, pixels, degrees).pixels

    private fun ByteBuffer.toByteArray(maxBytes: Int): ByteArray {
        val copy = duplicate()
        copy.rewind()
        if (copy.remaining() > maxBytes) return ByteArray(0)
        val bytes = ByteArray(copy.remaining())
        copy.get(bytes)
        return bytes
    }

    private companion object {
        const val MAX_ANALYSIS_PIXELS = 2_000_000L
        const val MAX_PLANE_BYTES = 8 * 1024 * 1024
    }
}
