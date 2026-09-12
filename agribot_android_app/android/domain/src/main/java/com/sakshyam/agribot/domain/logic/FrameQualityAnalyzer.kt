package com.sakshyam.agribot.domain.logic

import kotlin.math.max
import kotlin.math.sqrt

enum class FrameQualityStatus {
    ACCEPT,
    RETRY,
}

data class FrameQualityAssessment(
    val status: FrameQualityStatus,
    val brightness: Float,
    val contrast: Float,
    val sharpness: Float,
    val message: String,
    val action: String,
)

/**
 * A small, bounded image-quality gate. It samples the frame instead of making
 * another full-resolution copy, and deliberately reports a retry reason rather
 * than inventing a plant result from unusable pixels.
 */
object FrameQualityAnalyzer {
    fun evaluate(width: Int, height: Int, pixels: IntArray): FrameQualityAssessment {
        if (width <= 0 || height <= 0 || pixels.size != width * height) {
            return retry(0f, 0f, 0f, "Camera frame is invalid", "Hold still and try again")
        }

        val stride = max(1, sqrt((pixels.size / MAX_SAMPLES).toDouble()).toInt())
        var count = 0
        var sum = 0.0
        var sumSquares = 0.0
        var gradient = 0.0
        var gradientPairs = 0
        val luminance = { pixel: Int ->
            (0.2126 * ((pixel shr 16) and 0xff)) +
                (0.7152 * ((pixel shr 8) and 0xff)) +
                (0.0722 * (pixel and 0xff))
        }

        for (y in 0 until height step stride) {
            for (x in 0 until width step stride) {
                val current = luminance(pixels[y * width + x])
                count += 1
                sum += current
                sumSquares += current * current
                if (x + stride < width) {
                    gradient += kotlin.math.abs(current - luminance(pixels[y * width + x + stride]))
                    gradientPairs += 1
                }
                if (y + stride < height) {
                    gradient += kotlin.math.abs(current - luminance(pixels[(y + stride) * width + x]))
                    gradientPairs += 1
                }
            }
        }

        if (count == 0) return retry(0f, 0f, 0f, "Camera frame is empty", "Point the camera at a plant")
        val brightness = (sum / count / 255.0).toFloat()
        val variance = ((sumSquares / count) - (sum / count) * (sum / count)).coerceAtLeast(0.0)
        val contrast = (sqrt(variance) / 255.0).toFloat()
        val sharpness = if (gradientPairs == 0) 0f else (gradient / gradientPairs / 255.0).toFloat()

        return when {
            brightness < MIN_BRIGHTNESS -> retry(brightness, contrast, sharpness, "Photo is too dark", "Find more even light")
            brightness > MAX_BRIGHTNESS -> retry(brightness, contrast, sharpness, "Photo has too much glare", "Tilt the phone to remove the reflection")
            contrast < MIN_CONTRAST && sharpness < MIN_SHARPNESS -> retry(brightness, contrast, sharpness, "Photo has too little detail", "Move closer to one plant")
            sharpness < MIN_SHARPNESS && contrast < BLUR_CONTRAST_LIMIT -> retry(brightness, contrast, sharpness, "Photo is not clear", "Hold steady")
            else -> FrameQualityAssessment(
                status = FrameQualityStatus.ACCEPT,
                brightness = brightness,
                contrast = contrast,
                sharpness = sharpness,
                message = "Frame clear enough to check",
                action = "Keep the whole plant inside the frame",
            )
        }
    }

    private fun retry(
        brightness: Float,
        contrast: Float,
        sharpness: Float,
        message: String,
        action: String,
    ): FrameQualityAssessment = FrameQualityAssessment(
        status = FrameQualityStatus.RETRY,
        brightness = brightness,
        contrast = contrast,
        sharpness = sharpness,
        message = message,
        action = action,
    )

    private const val MAX_SAMPLES = 24_000
    private const val MIN_BRIGHTNESS = 0.08f
    private const val MAX_BRIGHTNESS = 0.95f
    private const val MIN_CONTRAST = 0.015f
    private const val MIN_SHARPNESS = 0.012f
    private const val BLUR_CONTRAST_LIMIT = 0.08f
}
