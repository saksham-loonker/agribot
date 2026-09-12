package com.sakshyam.agribot.featurescan.tracking

import com.sakshyam.agribot.domain.model.AnalysisFrame
import kotlin.math.abs

/** Image-plane translation only. Fractions of frame width/height are not metres. */
data class VisualMotion(
    val cameraRight: Float,
    val cameraDown: Float,
    val inliers: Int,
    val accepted: Boolean,
    val reason: String,
)

/**
 * Bounded sparse patch matching for a rigid mount viewing a locally planar crop bed.
 * A robust consensus rejects individual moving leaves and ambiguous repetitive texture.
 * Rotation is gated using the phone attitude; scale/depth are intentionally not invented.
 */
class VisualMotionEstimator {
    private var previous: IntArray? = null
    private var previousTimestamp = 0L
    private var previousWidth = 0
    private var previousHeight = 0

    fun reset() {
        previous = null
        previousTimestamp = 0L
    }

    fun observe(frame: AnalysisFrame, rotationDeltaRadians: Float?): VisualMotion {
        val gray = IntArray(WIDTH * HEIGHT) { index ->
            val x = ((index % WIDTH + 0.5f) * frame.width / WIDTH).toInt().coerceAtMost(frame.width - 1)
            val y = ((index / WIDTH + 0.5f) * frame.height / HEIGHT).toInt().coerceAtMost(frame.height - 1)
            val rgb = frame.rgbPixels[y * frame.width + x]
            (77 * ((rgb shr 16) and 255) + 150 * ((rgb shr 8) and 255) + 29 * (rgb and 255)) shr 8
        }
        val old = previous
        val elapsed = frame.timestampNanos - previousTimestamp
        val sameSize = previousWidth == frame.width && previousHeight == frame.height
        previous = gray
        previousTimestamp = frame.timestampNanos
        previousWidth = frame.width
        previousHeight = frame.height
        if (old == null || !sameSize) return rejected("Camera motion initializing")
        if (elapsed !in 1..750_000_000L) return rejected("Camera frame gap · reacquiring motion")
        if (rotationDeltaRadians == null || !rotationDeltaRadians.isFinite()) return rejected("Phone attitude unavailable")
        if (abs(rotationDeltaRadians) > 0.015f) return rejected("Phone rotating · reacquiring translation")

        val shifts = mutableListOf<Pair<Int, Int>>()
        for (y in 12 until HEIGHT - 12 step 8) {
            for (x in 12 until WIDTH - 12 step 8) {
                var min = 255
                var max = 0
                for (py in -2..2) for (px in -2..2) {
                    val value = old[(y + py) * WIDTH + x + px]
                    min = minOf(min, value)
                    max = maxOf(max, value)
                }
                if (max - min < 35) continue
                var best = Int.MAX_VALUE
                var second = Int.MAX_VALUE
                var bestX = 0
                var bestY = 0
                for (dy in -SEARCH..SEARCH) for (dx in -SEARCH..SEARCH) {
                    var error = 0
                    for (py in -2..2) for (px in -2..2) {
                        error += abs(old[(y + py) * WIDTH + x + px] - gray[(y + dy + py) * WIDTH + x + dx + px])
                    }
                    if (error < best) {
                        second = best
                        best = error
                        bestX = dx
                        bestY = dy
                    } else if (error < second) second = error
                }
                if (best < 25 * 22 && best < second * 0.8f && abs(bestX) < SEARCH && abs(bestY) < SEARCH) {
                    shifts += bestX to bestY
                }
            }
        }
        if (shifts.size < 8) return rejected("Insufficient distinct camera features")
        val dx = shifts.map { it.first }.sorted()[shifts.size / 2]
        val dy = shifts.map { it.second }.sorted()[shifts.size / 2]
        val inliers = shifts.filter { abs(it.first - dx) <= 1 && abs(it.second - dy) <= 1 }
        if (inliers.size < 8 || inliers.size < shifts.size * 0.65f) return rejected("Scene motion inconsistent")
        return VisualMotion(
            cameraRight = 0f - inliers.map { it.first }.average().toFloat() / WIDTH,
            cameraDown = 0f - inliers.map { it.second }.average().toFloat() / HEIGHT,
            inliers = inliers.size,
            accepted = true,
            reason = "Camera translation tracked · scale uncalibrated",
        )
    }

    private fun rejected(reason: String) = VisualMotion(0f, 0f, 0, false, reason)

    private companion object {
        const val WIDTH = 96
        const val HEIGHT = 72
        const val SEARCH = 7
    }
}
