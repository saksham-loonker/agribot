package com.sakshyam.agribot.ml.inference

import com.sakshyam.agribot.domain.model.BoundingBox
import com.sakshyam.agribot.ml.preprocessing.CropRect
import com.sakshyam.agribot.ml.preprocessing.RgbImage
import kotlin.math.roundToInt

object FrontCandidateCropper {
    private const val PAD_RATIO = 0.05f

    fun crop(image: RgbImage, bbox: BoundingBox, padRatio: Float = PAD_RATIO): RgbImage =
        image.crop(cropRect(image, bbox, padRatio))

    fun cropRect(image: RgbImage, bbox: BoundingBox, padRatio: Float = PAD_RATIO): CropRect {
        val left = bbox.left.coerceIn(0f, image.width.toFloat())
        val top = bbox.top.coerceIn(0f, image.height.toFloat())
        val right = bbox.right.coerceIn(0f, image.width.toFloat())
        val bottom = bbox.bottom.coerceIn(0f, image.height.toFloat())
        val width = (right - left).coerceAtLeast(1f)
        val height = (bottom - top).coerceAtLeast(1f)
        val pad = (maxOf(width, height) * padRatio.coerceAtLeast(0f)).roundToInt()
        val cropLeft = (left.roundToInt() - pad).coerceIn(0, image.width - 1)
        val cropTop = (top.roundToInt() - pad).coerceIn(0, image.height - 1)
        val cropRight = (right.roundToInt() + pad).coerceIn(cropLeft + 1, image.width)
        val cropBottom = (bottom.roundToInt() + pad).coerceIn(cropTop + 1, image.height)
        return CropRect(cropLeft, cropTop, cropRight, cropBottom)
    }
}
