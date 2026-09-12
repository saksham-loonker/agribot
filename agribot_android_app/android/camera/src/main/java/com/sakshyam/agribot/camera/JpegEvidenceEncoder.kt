package com.sakshyam.agribot.camera

import android.graphics.Bitmap
import android.graphics.Color
import java.io.ByteArrayOutputStream

object JpegEvidenceEncoder {
    fun encode(width: Int, height: Int, rgbPixels: IntArray, quality: Int = 90): ByteArray {
        if (width <= 0 || height <= 0 || width.toLong() * height.toLong() > MAX_RGB_PIXELS ||
            rgbPixels.size.toLong() != width.toLong() * height.toLong()
        ) return ByteArray(0)
        val argb = IntArray(rgbPixels.size) { index ->
            val pixel = rgbPixels[index]
            Color.rgb((pixel shr 16) and 0xff, (pixel shr 8) and 0xff, pixel and 0xff)
        }
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        return try {
            bitmap.setPixels(argb, 0, width, 0, 0, width, height)
            BoundedByteArrayOutputStream(MAX_JPEG_BYTES).use { output ->
                val compressed = bitmap.compress(Bitmap.CompressFormat.JPEG, quality.coerceIn(1, 100), output)
                if (!compressed || output.overflowed) ByteArray(0) else output.toByteArray()
            }
        } finally {
            bitmap.recycle()
        }
    }

    private class BoundedByteArrayOutputStream(private val maxBytes: Int) : ByteArrayOutputStream() {
        var overflowed: Boolean = false
            private set

        override fun write(source: ByteArray, offset: Int, length: Int) {
            if (count + length > maxBytes) {
                overflowed = true
                return
            }
            super.write(source, offset, length)
        }

        override fun write(value: Int) {
            if (count >= maxBytes) {
                overflowed = true
                return
            }
            super.write(value)
        }
    }

    private const val MAX_RGB_PIXELS = 2_000_000L
    private const val MAX_JPEG_BYTES = 512 * 1024
}
