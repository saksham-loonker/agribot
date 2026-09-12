package com.sakshyam.agribot.domain

import com.sakshyam.agribot.domain.model.GpsPathPoint
import com.sakshyam.agribot.domain.model.GpsPathPolicy
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class GpsPathPolicyTest {
    @Test
    fun projectsAcceptedFixesIntoRunLocalEastNorthCoordinates() {
        val point = GpsPathPolicy.project(
            sequence = 1,
            originLatitude = 20.0,
            originLongitude = 77.0,
            latitude = 20.001,
            longitude = 77.001,
            accuracyM = 8f,
            provider = "gps",
            elapsedRealtimeNanos = 10L,
        )

        assertNotNull(point)
        assertTrue(point.northM in 100.0..115.0)
        assertTrue(point.eastM in 100.0..115.0)
        assertEquals("gps", point.provider)
    }

    @Test
    fun rejectsInvalidCoordinatesAndAccuracy() {
        assertNull(
            GpsPathPolicy.project(
                sequence = 1,
                originLatitude = 91.0,
                originLongitude = 77.0,
                latitude = 20.0,
                longitude = 77.0,
                accuracyM = 8f,
                provider = "gps",
                elapsedRealtimeNanos = 10L,
            ),
        )
        assertNull(
            GpsPathPolicy.project(
                sequence = 1,
                originLatitude = 20.0,
                originLongitude = 77.0,
                latitude = 20.0,
                longitude = 77.0,
                accuracyM = -1f,
                provider = "gps",
                elapsedRealtimeNanos = 10L,
            ),
        )
    }

    @Test
    fun handlesLongitudeWrapWithoutAFullWorldJump() {
        val point = GpsPathPolicy.project(
            sequence = 1,
            originLatitude = 0.0,
            originLongitude = 179.999,
            latitude = 0.0,
            longitude = -179.999,
            accuracyM = 10f,
            provider = "network",
            elapsedRealtimeNanos = 10L,
        )

        assertNotNull(point)
        assertTrue(point.eastM in 200.0..250.0)
    }

    @Test
    fun dropsSubMeterNoiseAndBoundsHistory() {
        val anchor = GpsPathPoint(1, 0.0, 0.0, 8f, "gps", 1L)
        val noisy = GpsPathPoint(2, 0.1, 0.1, 8f, "gps", 2L)
        assertEquals(listOf(anchor), GpsPathPolicy.appendBounded(listOf(anchor), noisy))

        var points = listOf(anchor)
        for (sequence in 2..400) {
            points = GpsPathPolicy.appendBounded(
                points,
                GpsPathPoint(sequence, sequence.toDouble(), 0.0, 8f, "gps", sequence.toLong()),
            )
        }

        assertTrue(points.size <= GpsPathPolicy.MAX_POINTS)
        assertEquals(1, points.first().sequence)
        assertEquals(400, points.last().sequence)
    }
}
