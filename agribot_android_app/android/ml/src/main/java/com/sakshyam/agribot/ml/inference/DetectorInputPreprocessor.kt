package com.sakshyam.agribot.ml.inference

import com.sakshyam.agribot.ml.preprocessing.RgbImage
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.roundToInt
import kotlin.math.min

data class DetectorInputTransform(
    val inputWidth: Int,
    val inputHeight: Int,
    val frameWidth: Int,
    val frameHeight: Int,
    val scaleX: Float,
    val scaleY: Float,
    val padX: Float,
    val padY: Float,
) {
    fun modelXToFrame(value: Float): Float = (value - padX) / scaleX.coerceAtLeast(0.0001f)

    fun modelYToFrame(value: Float): Float = (value - padY) / scaleY.coerceAtLeast(0.0001f)
}

data class DetectorInput(
    val image: RgbImage,
    val transform: DetectorInputTransform,
)

/** Ultralytics-compatible square letterbox input for the detector. */
object DetectorInputPreprocessor {
    private const val LETTERBOX_GRAY = 114

    fun letterbox(image: RgbImage, inputWidth: Int, inputHeight: Int): DetectorInput {
        require(inputWidth > 0) { "inputWidth must be positive" }
        require(inputHeight > 0) { "inputHeight must be positive" }

        val scale = min(
            inputWidth.toFloat() / image.width,
            inputHeight.toFloat() / image.height,
        )
        val resizedWidth = max(1, (image.width * scale).roundToInt())
        val resizedHeight = max(1, (image.height * scale).roundToInt())
        val resized = RgbImageResizer.resizeBilinear(image, resizedWidth, resizedHeight)
        val padX = floor((inputWidth - resizedWidth) / 2f).toInt().coerceAtLeast(0)
        val padY = floor((inputHeight - resizedHeight) / 2f).toInt().coerceAtLeast(0)
        val output = IntArray(inputWidth * inputHeight) {
            (LETTERBOX_GRAY shl 16) or (LETTERBOX_GRAY shl 8) or LETTERBOX_GRAY
        }
        for (y in 0 until resizedHeight) {
            val destinationY = y + padY
            if (destinationY !in 0 until inputHeight) continue
            for (x in 0 until resizedWidth) {
                val destinationX = x + padX
                if (destinationX !in 0 until inputWidth) continue
                output[destinationY * inputWidth + destinationX] = resized.pixel(x, y)
            }
        }

        return DetectorInput(
            image = RgbImage(inputWidth, inputHeight, output),
            transform = DetectorInputTransform(
                inputWidth = inputWidth,
                inputHeight = inputHeight,
                frameWidth = image.width,
                frameHeight = image.height,
                scaleX = resizedWidth.toFloat() / image.width,
                scaleY = resizedHeight.toFloat() / image.height,
                padX = padX.toFloat(),
                padY = padY.toFloat(),
            ),
        )
    }
}

/** Small JVM-testable RGB resizer shared by model preprocessors. */
object RgbImageResizer {
    fun resizeBilinear(image: RgbImage, outWidth: Int, outHeight: Int): RgbImage {
        require(outWidth > 0) { "outWidth must be positive" }
        require(outHeight > 0) { "outHeight must be positive" }
        if (image.width == outWidth && image.height == outHeight) return image

        val output = IntArray(outWidth * outHeight)
        val xScale = image.width.toFloat() / outWidth
        val yScale = image.height.toFloat() / outHeight
        for (outY in 0 until outHeight) {
            val sourceY = ((outY + 0.5f) * yScale - 0.5f).coerceIn(0f, (image.height - 1).toFloat())
            val y0 = sourceY.toInt()
            val y1 = (y0 + 1).coerceAtMost(image.height - 1)
            val yWeight = sourceY - y0
            for (outX in 0 until outWidth) {
                val sourceX = ((outX + 0.5f) * xScale - 0.5f).coerceIn(0f, (image.width - 1).toFloat())
                val x0 = sourceX.toInt()
                val x1 = (x0 + 1).coerceAtMost(image.width - 1)
                val xWeight = sourceX - x0
                val top = blend(image.pixel(x0, y0), image.pixel(x1, y0), xWeight)
                val bottom = blend(image.pixel(x0, y1), image.pixel(x1, y1), xWeight)
                output[outY * outWidth + outX] = blend(top, bottom, yWeight)
            }
        }
        return RgbImage(outWidth, outHeight, output)
    }

    private fun blend(first: Int, second: Int, weight: Float): Int {
        fun channel(shift: Int): Int {
            val firstChannel = first shr shift and 0xff
            val secondChannel = second shr shift and 0xff
            return (firstChannel + (secondChannel - firstChannel) * weight)
                .roundToInt()
                .coerceIn(0, 255)
        }

        return (channel(16).coerceIn(0, 255) shl 16) or
            (channel(8).coerceIn(0, 255) shl 8) or
            channel(0).coerceIn(0, 255)
    }
}
