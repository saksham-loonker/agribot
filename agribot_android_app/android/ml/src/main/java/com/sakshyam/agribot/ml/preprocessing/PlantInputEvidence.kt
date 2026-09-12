package com.sakshyam.agribot.ml.preprocessing

/**
 * Rejects effectively achromatic inputs before a closed-set disease classifier.
 * This is only an input-quality guard: colored objects can still pass, and
 * grayscale/very desaturated plant photos require a new color capture.
 * It must never be interpreted as a plant detector or a health prediction.
 */
object PlantInputEvidence {
    fun hasUsableColor(image: RgbImage): Boolean {
        var usable = 0
        var sampled = 0
        val stride = (image.pixels.size / 4096).coerceAtLeast(1)
        for (index in image.pixels.indices step stride) {
            val pixel = image.pixels[index]
            val r = (pixel shr 16) and 255
            val g = (pixel shr 8) and 255
            val b = pixel and 255
            val high = maxOf(r, g, b)
            val low = minOf(r, g, b)
            sampled++
            if (high >= 30 && high - low >= 18 && (high - low).toFloat() / high >= 0.12f) usable++
        }
        return sampled > 0 && usable.toFloat() / sampled >= 0.02f
    }
}
