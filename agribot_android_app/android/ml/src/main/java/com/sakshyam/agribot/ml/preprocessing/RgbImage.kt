package com.sakshyam.agribot.ml.preprocessing

import android.graphics.Bitmap
import android.graphics.Color

data class RgbImage(
    val width: Int,
    val height: Int,
    val pixels: IntArray,
) {
    init {
        require(width > 0) { "width must be positive" }
        require(height > 0) { "height must be positive" }
        require(pixels.size == width * height) { "pixels size must equal width * height" }
    }

    fun pixel(x: Int, y: Int): Int = pixels[y * width + x]

    fun crop(rect: CropRect): RgbImage {
        val outWidth = rect.right - rect.left
        val outHeight = rect.bottom - rect.top
        val out = IntArray(outWidth * outHeight)
        for (y in 0 until outHeight) {
            val src = (rect.top + y) * width + rect.left
            val dst = y * outWidth
            pixels.copyInto(out, destinationOffset = dst, startIndex = src, endIndex = src + outWidth)
        }
        return RgbImage(outWidth, outHeight, out)
    }

    fun toBitmap(): Bitmap {
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val argb = IntArray(pixels.size) { index ->
            val pixel = pixels[index]
            Color.rgb((pixel shr 16) and 0xff, (pixel shr 8) and 0xff, pixel and 0xff)
        }
        bitmap.setPixels(argb, 0, width, 0, 0, width, height)
        return bitmap
    }

    companion object {
        fun fromBitmap(bitmap: Bitmap): RgbImage {
            val width = bitmap.width
            val height = bitmap.height
            val argb = IntArray(width * height)
            bitmap.getPixels(argb, 0, width, 0, 0, width, height)
            val rgb = IntArray(argb.size) { index ->
                val color = argb[index]
                (Color.red(color) shl 16) or (Color.green(color) shl 8) or Color.blue(color)
            }
            return RgbImage(width, height, rgb)
        }
    }
}

data class CropRect(
    val left: Int,
    val top: Int,
    val right: Int,
    val bottom: Int,
)
