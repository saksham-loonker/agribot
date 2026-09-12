package com.sakshyam.agribot.domain.logic

import kotlin.math.max
import kotlin.math.sqrt

/**
 * Computes the fraction of "vegetation" pixels in a frame.
 *
 * "Vegetation" here means pixels whose RGB triple is dominated by green and
 * which have enough colour saturation to rule out grey surfaces, walls, and
 * bare concrete. This is not a learned plant detector: it is a cheap,
 * deterministic scene check that lets the pipeline reject obviously non-plant
 * scenes (a lab, an empty office, a tabletop) before turning every detector
 * box into a "Plant N: Late_blight" entry.
 *
 * The function samples the frame at a stride so it stays cheap on a phone and
 * never allocates another full-resolution copy.
 */
object VegetationCoverageAnalyzer {
    data class Assessment(
        val coverage: Float,
        val greenShare: Float,
        val leafLikeShare: Float,
        val sampleCount: Int,
    )

    fun evaluate(width: Int, height: Int, pixels: IntArray): Assessment {
        if (width <= 0 || height <= 0 || pixels.size != width * height) {
            return Assessment(coverage = 0f, greenShare = 0f, leafLikeShare = 0f, sampleCount = 0)
        }
        val stride = max(1, sqrt((pixels.size / MAX_SAMPLES).toDouble()).toInt())
        var sampled = 0
        var greenPixels = 0
        var leafLikePixels = 0
        for (y in 0 until height step stride) {
            val rowStart = y * width
            for (x in 0 until width step stride) {
                val pixel = pixels[rowStart + x]
                val r = (pixel shr 16) and 0xff
                val g = (pixel shr 8) and 0xff
                val b = pixel and 0xff
                sampled += 1
                if (isGreenDominant(r, g, b)) greenPixels += 1
                if (isLeafLike(r, g, b)) leafLikePixels += 1
            }
        }
        if (sampled == 0) {
            return Assessment(coverage = 0f, greenShare = 0f, leafLikeShare = 0f, sampleCount = 0)
        }
        val greenShare = greenPixels.toFloat() / sampled
        val leafLikeShare = leafLikePixels.toFloat() / sampled
        // The "coverage" we act on is the more conservative of the two
        // signals: pixels that look like actual leaves, not just anything
        // where green happens to be the largest channel.
        return Assessment(
            coverage = leafLikeShare,
            greenShare = greenShare,
            leafLikeShare = leafLikeShare,
            sampleCount = sampled,
        )
    }

    private fun isGreenDominant(r: Int, g: Int, b: Int): Boolean {
        if (g < MIN_GREEN) return false
        return g > r && g > b
    }

    private fun isLeafLike(r: Int, g: Int, b: Int): Boolean {
        if (g < LEAF_MIN_GREEN) return false
        if (r < LEAF_MIN_RED) return false
        // Leaves are not pure black, not blown out white, and the green channel
        // clearly dominates red.
        val max = maxOf(r, g, b)
        if (max < LEAF_MIN_MAX) return false
        if (max > LEAF_MAX_MAX) return false
        val greenOverRed = g.toFloat() / r.coerceAtLeast(1)
        if (greenOverRed < LEAF_MIN_GREEN_OVER_RED) return false
        val chroma = (g - kotlin.math.max(r, b)).toFloat()
        if (chroma < LEAF_MIN_CHROMA) return false
        return true
    }

    private const val MAX_SAMPLES = 24_000
    private const val MIN_GREEN = 70
    private const val LEAF_MIN_GREEN = 70
    private const val LEAF_MIN_RED = 35
    private const val LEAF_MIN_MAX = 60
    private const val LEAF_MAX_MAX = 245
    private const val LEAF_MIN_GREEN_OVER_RED = 1.05f
    private const val LEAF_MIN_CHROMA = 10f
}