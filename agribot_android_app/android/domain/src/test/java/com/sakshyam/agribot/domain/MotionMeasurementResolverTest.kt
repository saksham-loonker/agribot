package com.sakshyam.agribot.domain

import com.sakshyam.agribot.domain.logic.MotionMeasurementInput
import com.sakshyam.agribot.domain.logic.MotionMeasurementResolver
import com.sakshyam.agribot.domain.model.MeasurementQuality
import com.sakshyam.agribot.domain.model.MeasurementSource
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class MotionMeasurementResolverTest {
    @Test
    fun tractorVibrationNeverBecomesStrideDistance() {
        val input = MotionMeasurementInput(
            trackingActive = true, motionEventsObserved = true, stepEventsObserved = true,
            sensorDistanceM = 12f, sensorSpeedMps = 2f, stepCount = 17,
            gpsReferenceEnabled = true, gpsFixAvailable = true,
            gpsDistanceM = 4f, gpsSpeedMps = 5f, gpsAccuracyM = 8f, gpsFixAgeSeconds = 1f,
            tractorMounted = true,
        )
        val result = MotionMeasurementResolver.resolve(input)
        assertEquals(4f, result.distance.value)
        assertEquals(5f, result.speed.value)
        assertEquals(null, result.stepCount)
        assertEquals(MeasurementSource.GPS_REFERENCE, result.source)
        assertEquals(MeasurementQuality.ESTIMATED, result.distance.quality)
        val stale = MotionMeasurementResolver.resolve(input.copy(gpsFixAgeSeconds = 8f))
        assertEquals(null, stale.distance.value)
        assertEquals(null, stale.speed.value)
    }

    @Test
    fun missingEventsAreUnavailableInsteadOfZero() {
        val result = MotionMeasurementResolver.resolve(
            MotionMeasurementInput(
                trackingActive = true,
                motionEventsObserved = false,
                sensorDistanceM = 0f,
                sensorSpeedMps = 0f,
                stepCount = 0,
                gpsReferenceEnabled = false,
                gpsFixAvailable = false,
                gpsDistanceM = null,
                gpsSpeedMps = null,
                gpsAccuracyM = null,
                gpsFixAgeSeconds = null,
            ),
        )

        assertNull(result.distance.value)
        assertEquals(MeasurementQuality.UNAVAILABLE, result.distance.quality)
        assertNull(result.stepCount)
    }

    @Test
    fun aFreshGpsFixCanReferenceZeroDistanceAtTheStartOfAWalk() {
        val result = MotionMeasurementResolver.resolve(
            MotionMeasurementInput(
                trackingActive = true,
                motionEventsObserved = false,
                sensorDistanceM = null,
                sensorSpeedMps = null,
                stepCount = null,
                gpsReferenceEnabled = true,
                gpsFixAvailable = true,
                gpsDistanceM = 0f,
                gpsSpeedMps = null,
                gpsAccuracyM = 8f,
                gpsFixAgeSeconds = 0.5f,
            ),
        )

        assertEquals(0f, result.distance.value)
        assertEquals(MeasurementQuality.REFERENCED, result.distance.quality)
        assertEquals(MeasurementSource.GPS_REFERENCE, result.source)
        assertNull(result.speed.value)
        assertEquals(MeasurementQuality.UNAVAILABLE, result.speed.quality)
    }

    @Test
    fun staleGpsFallsBackToObservedPhoneMotion() {
        val result = MotionMeasurementResolver.resolve(
            MotionMeasurementInput(
                trackingActive = true,
                motionEventsObserved = true,
                sensorDistanceM = 4.2f,
                sensorSpeedMps = 0.8f,
                stepCount = 6,
                gpsReferenceEnabled = true,
                gpsFixAvailable = true,
                gpsDistanceM = 2.0f,
                gpsSpeedMps = 0.2f,
                gpsAccuracyM = 7f,
                gpsFixAgeSeconds = 9f,
            ),
        )

        assertEquals(4.2f, result.distance.value)
        assertEquals(MeasurementQuality.ESTIMATED, result.distance.quality)
        assertEquals(MeasurementSource.PHONE_ACCELERATION, result.source)
        assertEquals(6, result.stepCount)
    }

    @Test
    fun weakGpsDoesNotOverrideObservedPhoneMotion() {
        val result = MotionMeasurementResolver.resolve(
            MotionMeasurementInput(
                trackingActive = true,
                motionEventsObserved = true,
                sensorDistanceM = 1.4f,
                sensorSpeedMps = 0.5f,
                stepCount = 2,
                gpsReferenceEnabled = true,
                gpsFixAvailable = true,
                gpsDistanceM = 20f,
                gpsSpeedMps = 0.2f,
                gpsAccuracyM = 48f,
                gpsFixAgeSeconds = 0.5f,
            ),
        )

        assertEquals(MeasurementSource.PHONE_ACCELERATION, result.source)
        assertEquals(1.4f, result.distance.value)
        assertEquals(MeasurementQuality.ESTIMATED, result.distance.quality)
    }

    @Test
    fun freshStepEventsExposeTheStepSensorAsTheActiveSource() {
        val result = MotionMeasurementResolver.resolve(
            MotionMeasurementInput(
                trackingActive = true,
                motionEventsObserved = true,
                stepEventsObserved = true,
                sensorDistanceM = 2.16f,
                sensorSpeedMps = 0.8f,
                stepCount = 3,
                gpsReferenceEnabled = false,
                gpsFixAvailable = false,
                gpsDistanceM = null,
                gpsSpeedMps = null,
                gpsAccuracyM = null,
                gpsFixAgeSeconds = null,
            ),
        )

        assertEquals(MeasurementSource.PHONE_STEP_SENSOR, result.source)
        assertEquals(MeasurementSource.PHONE_STEP_SENSOR, result.distance.source)
    }

    @Test
    fun freshFirstGpsFixDoesNotPinAnObservedPhonePathToZero() {
        val result = MotionMeasurementResolver.resolve(
            MotionMeasurementInput(
                trackingActive = true,
                motionEventsObserved = true,
                sensorDistanceM = 3.6f,
                sensorSpeedMps = 0.9f,
                stepCount = 5,
                gpsReferenceEnabled = true,
                gpsFixAvailable = true,
                gpsDistanceM = 0f,
                gpsSpeedMps = null,
                gpsAccuracyM = 8f,
                gpsFixAgeSeconds = 0.5f,
            ),
        )

        assertEquals(3.6f, result.distance.value)
        assertEquals(MeasurementSource.GPS_AND_PHONE_SENSORS, result.source)
        assertEquals(MeasurementQuality.REFERENCED, result.distance.quality)
    }

    @Test
    fun stoppingKeepsTheLastNumericSnapshotInsteadOfTurningItIntoZero() {
        val active = MotionMeasurementResolver.resolve(
            MotionMeasurementInput(
                trackingActive = true,
                motionEventsObserved = true,
                stepEventsObserved = true,
                sensorDistanceM = 8.4f,
                sensorSpeedMps = 1.2f,
                stepCount = 12,
                gpsReferenceEnabled = false,
                gpsFixAvailable = false,
                gpsDistanceM = null,
                gpsSpeedMps = null,
                gpsAccuracyM = null,
                gpsFixAgeSeconds = null,
            ),
        )

        val stopped = MotionMeasurementResolver.retainAfterStop(active)

        assertEquals(8.4f, stopped.distance.value)
        assertEquals(1.2f, stopped.speed.value)
        assertEquals(12, stopped.stepCount)
        assertEquals(active.source, stopped.source)
        assertEquals("Last scan measurement kept", stopped.statusMessage)
    }
}
