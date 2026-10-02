package com.sakshyam.agribot.domain.scan

import kotlin.math.PI
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.sin
import kotlin.math.sqrt

data class GpsFix(val latitude: Double, val longitude: Double, val accuracyM: Double, val timeMs: Long)

/**
 * Builds a field trail from GPS fixes without accumulating jitter: a fix is kept only when it is
 * accurate, the farmer is actually walking (recent steps), and it moved further than both a minimum
 * spacing and its own uncertainty. GPS is used for the trail only, never for plant counting.
 */
class GpsTrailFilter(
    private val maxAccuracyM: Double = 15.0,
    private val minSpacingM: Double = 3.0,
    private val maxSpeedMps: Double = 3.0,
    private val maxPoints: Int = 2000,
) {
    private val pts = ArrayList<GpsFix>()
    val points: List<GpsFix> get() = pts

    /** Returns true if [fix] was appended to the trail. */
    fun offer(fix: GpsFix, walkingRecently: Boolean): Boolean {
        if (fix.accuracyM <= 0.0 || fix.accuracyM > maxAccuracyM) return false
        val last = pts.lastOrNull()
        if (last == null) {
            pts += fix; return true
        }
        if (!walkingRecently) return false
        val d = distanceM(last, fix)
        if (d < max(minSpacingM, max(fix.accuracyM, last.accuracyM))) return false
        val dt = (fix.timeMs - last.timeMs) / 1000.0
        if (dt <= 0.0 || d / dt > maxSpeedMps) return false
        if (pts.size >= maxPoints) pts.removeAt(0)
        pts += fix
        return true
    }

    /** Trail length in metres. */
    fun lengthM(): Double = pts.zipWithNext { a, b -> distanceM(a, b) }.sum()

    fun clear() = pts.clear()

    companion object {
        private const val EARTH_RADIUS_M = 6_371_008.8

        /** Haversine distance. */
        fun distanceM(a: GpsFix, b: GpsFix): Double {
            val p1 = a.latitude * PI / 180
            val p2 = b.latitude * PI / 180
            val dp = p2 - p1
            val dl = (b.longitude - a.longitude) * PI / 180
            val h = sin(dp / 2) * sin(dp / 2) + cos(p1) * cos(p2) * sin(dl / 2) * sin(dl / 2)
            return 2 * EARTH_RADIUS_M * atan2(sqrt(h), sqrt(1 - h))
        }
    }
}
