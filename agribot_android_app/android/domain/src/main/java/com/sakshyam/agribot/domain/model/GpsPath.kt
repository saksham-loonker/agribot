package com.sakshyam.agribot.domain.model

import kotlin.math.cos
import kotlin.math.hypot

/**
 * The quality of a GPS-derived path shown on the field map.
 *
 * A path is deliberately separate from the distance measurement. Phone motion
 * can estimate how far the operator walked without knowing a trustworthy 2D
 * direction, so the UI must not draw a path from distance alone.
 */
enum class GpsPathStatus {
    UNAVAILABLE,
    WAITING_FOR_FIX,
    ANCHOR_ONLY,
    REFERENCE_PATH,
    STALE,
    MOTION_ONLY,
}

/**
 * A privacy-preserving GPS point in a run-local east/north coordinate system.
 * Raw latitude/longitude is intentionally not retained here.
 */
data class GpsPathPoint(
    val sequence: Int,
    val eastM: Double,
    val northM: Double,
    val accuracyM: Float?,
    val provider: String?,
    val elapsedRealtimeNanos: Long,
) {
    init {
        require(sequence > 0) { "sequence must be positive" }
        require(eastM.isFinite()) { "eastM must be finite" }
        require(northM.isFinite()) { "northM must be finite" }
        require(accuracyM == null || (accuracyM.isFinite() && accuracyM >= 0f)) {
            "accuracyM must be finite and non-negative when present"
        }
        require(elapsedRealtimeNanos >= 0L) { "elapsedRealtimeNanos must be non-negative" }
    }
}

/**
 * Pure geometry and bounded-history policy for GPS reference points.
 * Android Location objects stay at the device boundary in PlantTracker.
 */
object GpsPathPolicy {
    const val MAX_POINTS = 256
    const val MIN_POINT_SPACING_M = 0.5

    private const val EARTH_RADIUS_M = 6_378_137.0

    fun project(
        sequence: Int,
        originLatitude: Double,
        originLongitude: Double,
        latitude: Double,
        longitude: Double,
        accuracyM: Float?,
        provider: String?,
        elapsedRealtimeNanos: Long,
    ): GpsPathPoint? {
        if (!validCoordinate(originLatitude, originLongitude) ||
            !validCoordinate(latitude, longitude) ||
            sequence <= 0 ||
            elapsedRealtimeNanos < 0L
        ) {
            return null
        }
        if (accuracyM != null && (!accuracyM.isFinite() || accuracyM < 0f)) return null

        val latitudeRadians = Math.toRadians(originLatitude)
        val deltaLatitudeRadians = Math.toRadians(latitude - originLatitude)
        val deltaLongitudeDegrees = shortestLongitudeDelta(longitude - originLongitude)
        val deltaLongitudeRadians = Math.toRadians(deltaLongitudeDegrees)
        val northM = deltaLatitudeRadians * EARTH_RADIUS_M
        val eastM = deltaLongitudeRadians * EARTH_RADIUS_M * cos(latitudeRadians)
        if (!eastM.isFinite() || !northM.isFinite()) return null

        return GpsPathPoint(
            sequence = sequence,
            eastM = eastM,
            northM = northM,
            accuracyM = accuracyM,
            provider = provider?.trim()?.takeIf { it.isNotEmpty() },
            elapsedRealtimeNanos = elapsedRealtimeNanos,
        )
    }

    fun appendBounded(existing: List<GpsPathPoint>, next: GpsPathPoint): List<GpsPathPoint> {
        if (existing.isEmpty()) return listOf(next)
        val previous = existing.last()
        if (distanceM(previous, next) < MIN_POINT_SPACING_M) return existing
        if (existing.size < MAX_POINTS) return existing + next

        // Keep the oldest anchor and every second point, then append the newest
        // point. This preserves the overall path shape without unbounded memory.
        val compacted = existing.filterIndexed { index, _ -> index == 0 || index % 2 == 0 }
        return (compacted + next).takeLast(MAX_POINTS)
    }

    fun distanceM(first: GpsPathPoint, second: GpsPathPoint): Double =
        hypot(second.eastM - first.eastM, second.northM - first.northM)

    private fun validCoordinate(latitude: Double, longitude: Double): Boolean =
        latitude.isFinite() && longitude.isFinite() &&
            latitude in -90.0..90.0 && longitude in -180.0..180.0

    private fun shortestLongitudeDelta(deltaDegrees: Double): Double {
        var normalized = deltaDegrees % 360.0
        if (normalized > 180.0) normalized -= 360.0
        if (normalized < -180.0) normalized += 360.0
        return normalized
    }
}
