package com.sakshyam.agribot.ml.preprocessing

import com.sakshyam.agribot.ml.inference.RgbImageResizer
import com.sakshyam.agribot.domain.model.ScanConstants
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

object MaskCropPreprocessor {
    const val MAX_EDGE = 256
    const val MASK_CROP_SCALE = 0.55f
    const val MASK_CROP_PAD = 0.05f
    const val CLF_CONF_THRESHOLD = ScanConstants.DEFAULT_CONFIDENCE_THRESHOLD
    const val HIGH_CONF_THRESHOLD = ScanConstants.DEFAULT_HIGH_CONFIDENCE_THRESHOLD

    fun downscaleIfNeeded(image: RgbImage, maxEdge: Int = MAX_EDGE): RgbImage {
        if (max(image.width, image.height) <= maxEdge) return image
        val scale = maxEdge.toFloat() / max(image.width, image.height)
        val outWidth = max(1, (image.width * scale).roundToInt())
        val outHeight = max(1, (image.height * scale).roundToInt())
        return RgbImageResizer.resizeBilinear(image, outWidth, outHeight)
    }

    fun centerCropBounds(image: RgbImage, cropScale: Float = 0.70f): CropRect {
        val scale = cropScale.coerceIn(0.05f, 1.0f)
        val cropWidth = max(1, (image.width * scale).roundToInt())
        val cropHeight = max(1, (image.height * scale).roundToInt())
        val x1 = max(0, (image.width - cropWidth) / 2)
        val y1 = max(0, (image.height - cropHeight) / 2)
        return CropRect(x1, y1, x1 + cropWidth, y1 + cropHeight)
    }

    fun maskCropBounds(
        image: RgbImage,
        cropPad: Float = MASK_CROP_PAD,
        fallbackScale: Float = MASK_CROP_SCALE,
    ): CropRect {
        val width = image.width
        val height = image.height
        if (height < 4 || width < 4) return CropRect(0, 0, width, height)

        val mask = ByteArray(width * height)
        for (index in image.pixels.indices) {
            val pixel = image.pixels[index]
            val red = (pixel shr 16) and 0xff
            val green = (pixel shr 8) and 0xff
            val blue = pixel and 0xff
            // RgbImage stores channels as RGB.  Python's cv2 path receives a
            // BGR array, but cvtColor(BGR2HSV) restores the same RGB colour
            // interpretation.  Swapping channels here mirrored BGR twice and
            // made red/brown disease pixels look blue to the mask.
            val hsv = rgbToOpenCvHsv(red, green, blue)

            val greenYellow = hsv.hue >= 18f && hsv.hue <= 95f && hsv.saturation >= 25f && hsv.value >= 30f
            val redBrown = (hsv.hue <= 18f || hsv.hue >= 165f) && hsv.saturation >= 45f && hsv.value >= 30f
            val saturatedLeaf = hsv.saturation >= 45f && hsv.value >= 35f && hsv.value <= 245f
            mask[index] = if (greenYellow || redBrown || saturatedLeaf) 1 else 0
        }

        val opened = dilate(erode(mask, width, height), width, height)
        val closed = erode(dilate(opened, width, height), width, height)
        val boxes = connectedComponentBoxes(closed, width, height)
            .filter { it.area >= max(24.0, 0.004 * height * width) }

        if (boxes.isEmpty()) return centerCropBounds(image, fallbackScale)

        var x1 = Int.MAX_VALUE
        var y1 = Int.MAX_VALUE
        var x2 = Int.MIN_VALUE
        var y2 = Int.MIN_VALUE
        for (box in boxes) {
            x1 = min(x1, box.left)
            y1 = min(y1, box.top)
            x2 = max(x2, box.right)
            y2 = max(y2, box.bottom)
        }

        val boxWidth = x2 - x1
        val boxHeight = y2 - y1
        if (boxWidth <= 0 || boxHeight <= 0) return centerCropBounds(image, fallbackScale)
        if (boxWidth * boxHeight > 0.92 * height * width) return centerCropBounds(image, fallbackScale)

        val pad = (max(boxWidth, boxHeight) * max(0f, cropPad)).roundToInt()
        return CropRect(
            left = max(0, x1 - pad),
            top = max(0, y1 - pad),
            right = min(width, x2 + pad),
            bottom = min(height, y2 + pad),
        )
    }

    fun maskCrop(image: RgbImage, cropPad: Float = MASK_CROP_PAD, fallbackScale: Float = MASK_CROP_SCALE): RgbImage =
        image.crop(maskCropBounds(image, cropPad, fallbackScale))

    fun preprocessForSideScan(image: RgbImage): RgbImage {
        // Keep segmentation bounded, but extract from the original so small
        // lesions are not destroyed by a whole-frame resize before cropping.
        val preview = downscaleIfNeeded(image)
        val bounds = maskCropBounds(preview, MASK_CROP_PAD, MASK_CROP_SCALE)
        val scaleX = image.width.toDouble() / preview.width
        val scaleY = image.height.toDouble() / preview.height
        return image.crop(CropRect(
            left = kotlin.math.floor(bounds.left * scaleX).toInt().coerceAtLeast(0),
            top = kotlin.math.floor(bounds.top * scaleY).toInt().coerceAtLeast(0),
            right = kotlin.math.ceil(bounds.right * scaleX).toInt().coerceAtMost(image.width),
            bottom = kotlin.math.ceil(bounds.bottom * scaleY).toInt().coerceAtMost(image.height),
        ))
    }

    private data class Hsv(val hue: Float, val saturation: Float, val value: Float)

    private fun rgbToOpenCvHsv(red: Int, green: Int, blue: Int): Hsv {
        val r = red / 255f
        val g = green / 255f
        val b = blue / 255f
        val maxC = max(r, max(g, b))
        val minC = min(r, min(g, b))
        val delta = maxC - minC
        val hueDegrees = when {
            delta < 0.0001f -> 0f
            maxC == r -> (60f * ((g - b) / delta) + 360f) % 360f
            maxC == g -> 60f * ((b - r) / delta) + 120f
            else -> 60f * ((r - g) / delta) + 240f
        }
        val saturation = if (maxC == 0f) 0f else (delta / maxC) * 255f
        return Hsv(
            hue = hueDegrees / 2f,
            saturation = saturation,
            value = maxC * 255f,
        )
    }

    private fun erode(mask: ByteArray, width: Int, height: Int): ByteArray {
        val out = ByteArray(mask.size)
        for (y in 0 until height) {
            for (x in 0 until width) {
                var allOn = true
                for (ky in -2..2) {
                    for (kx in -2..2) {
                        val nx = (x + kx).coerceIn(0, width - 1)
                        val ny = (y + ky).coerceIn(0, height - 1)
                        if (mask[ny * width + nx] == 0.toByte()) {
                            allOn = false
                            break
                        }
                    }
                    if (!allOn) break
                }
                out[y * width + x] = if (allOn) 1 else 0
            }
        }
        return out
    }

    private fun dilate(mask: ByteArray, width: Int, height: Int): ByteArray {
        val out = ByteArray(mask.size)
        for (y in 0 until height) {
            for (x in 0 until width) {
                var anyOn = false
                for (ky in -2..2) {
                    for (kx in -2..2) {
                        val nx = (x + kx).coerceIn(0, width - 1)
                        val ny = (y + ky).coerceIn(0, height - 1)
                        if (mask[ny * width + nx] == 1.toByte()) {
                            anyOn = true
                            break
                        }
                    }
                    if (anyOn) break
                }
                out[y * width + x] = if (anyOn) 1 else 0
            }
        }
        return out
    }

    private data class ComponentBox(
        val left: Int,
        val top: Int,
        val right: Int,
        val bottom: Int,
        val area: Int,
    )

    private fun connectedComponentBoxes(mask: ByteArray, width: Int, height: Int): List<ComponentBox> {
        val visited = BooleanArray(mask.size)
        val boxes = mutableListOf<ComponentBox>()
        val stack = ArrayDeque<Int>()

        for (start in mask.indices) {
            if (mask[start] == 0.toByte() || visited[start]) continue
            visited[start] = true
            stack.add(start)
            var minX = start % width
            var maxX = minX
            var minY = start / width
            var maxY = minY
            var area = 0

            while (stack.isNotEmpty()) {
                val current = stack.removeLast()
                val x = current % width
                val y = current / width
                area += 1
                minX = min(minX, x)
                maxX = max(maxX, x)
                minY = min(minY, y)
                maxY = max(maxY, y)

                for (dy in -1..1) {
                    for (dx in -1..1) {
                        if (dx == 0 && dy == 0) continue
                        val nx = x + dx
                        val ny = y + dy
                        if (nx !in 0 until width || ny !in 0 until height) continue
                        val next = ny * width + nx
                        if (mask[next] == 1.toByte() && !visited[next]) {
                            visited[next] = true
                            stack.add(next)
                        }
                    }
                }
            }

            boxes.add(ComponentBox(minX, minY, maxX + 1, maxY + 1, area))
        }

        return boxes
    }
}
