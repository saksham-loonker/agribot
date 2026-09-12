package com.sakshyam.agribot.featurescan.tracking

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class MotionDistanceEstimatorTest {
    @Test
    fun linearAccelerationFallbackProducesEstimatedStepsWhenNoHardwareStepEventsArrive() {
        val estimator = MotionDistanceEstimator()
        feedWalkingPulses(estimator, axis = Axis.X)

        val result = estimator.snapshot()
        assertTrue(result.distanceM > 0f)
        assertTrue(result.speedMps > 0f)
        assertTrue(result.stepCount >= 3)
        assertEquals(false, result.stepEventsObserved)
    }

    @Test
    fun linearAccelerationFallbackUsesTheThirdAxisWhenPhoneIsHeldUpright() {
        val estimator = MotionDistanceEstimator()
        feedWalkingPulses(estimator, axis = Axis.Z)

        val result = estimator.snapshot()
        assertTrue(result.distanceM > 0f)
        assertTrue(result.speedMps > 0f)
    }

    @Test
    fun staleStepEventsAutomaticallyReenableAccelerationFallback() {
        val estimator = MotionDistanceEstimator()
        estimator.onStep(1_000_000_000L)
        estimator.onLinearAcceleration(1_100_000_000L, 0.8f, 0f)
        estimator.onLinearAcceleration(2_600_000_000L, 0f, 0f)
        estimator.onLinearAcceleration(2_700_000_000L, 0.8f, 0f)
        estimator.onLinearAcceleration(2_800_000_000L, 0f, 0f)

        assertTrue(estimator.snapshot().distanceM > 0.72f)
    }

    @Test
    fun stationaryAccelerationDoesNotAccumulateUnboundedDistance() {
        val estimator = MotionDistanceEstimator()
        var timestamp = 1_000_000_000L
        repeat(100) {
            estimator.onLinearAcceleration(timestamp, 0f, 0f, 0f)
            timestamp += 20_000_000L
        }

        val result = estimator.snapshot()
        assertEquals(0f, result.distanceM)
        assertEquals(0f, result.speedMps)
        assertEquals(0, result.stepCount)
    }

    @Test
    fun stepDistanceAndSpeedAreReportedWhenStepEventsAreAvailable() {
        val estimator = MotionDistanceEstimator()
        estimator.onStep(1_000_000_000L)
        val result = estimator.onStep(1_500_000_000L)

        assertEquals(1.44f, result.distanceM, absoluteTolerance = 0.001f)
        assertTrue(result.speedMps > 0f)
        assertEquals(2, result.stepCount)
        assertEquals(true, result.stepEventsObserved)
    }

    @Test
    fun calibratedStepLengthScalesHardwareStepDistance() {
        val estimator = MotionDistanceEstimator(initialStepLengthM = 0.5f)

        estimator.onStep(1_000_000_000L)
        val result = estimator.onStep(1_500_000_000L)

        assertEquals(1f, result.distanceM, absoluteTolerance = 0.001f)
    }

    @Test
    fun calibratedStepLengthAlsoScalesAccelerationFallback() {
        val estimator = MotionDistanceEstimator(initialStepLengthM = 0.5f)
        feedWalkingPulses(estimator, axis = Axis.X)

        val result = estimator.snapshot()

        assertEquals(result.stepCount * 0.5f, result.distanceM, absoluteTolerance = 0.001f)
    }

    @Test
    fun invalidStepLengthIsRejectedWithoutChangingDistanceProfile() {
        val estimator = MotionDistanceEstimator(initialStepLengthM = 0.72f)

        assertFalse(estimator.setStepLengthM(2.5f))
        estimator.onStep(1_000_000_000L)

        assertEquals(0.72f, estimator.snapshot().distanceM, absoluteTolerance = 0.001f)
    }

    private enum class Axis { X, Z }

    private fun feedWalkingPulses(estimator: MotionDistanceEstimator, axis: Axis) {
        var timestamp = 1_000_000_000L
        repeat(4) {
            sendAcceleration(estimator, timestamp, 0f, 0f, 0f, axis)
            sendAcceleration(estimator, timestamp + 100_000_000L, 0.9f, 0.1f, 0.2f, axis)
            sendAcceleration(estimator, timestamp + 200_000_000L, 0.05f, 0.02f, 0.04f, axis)
            timestamp += 500_000_000L
        }
    }

    private fun sendAcceleration(
        estimator: MotionDistanceEstimator,
        timestamp: Long,
        x: Float,
        y: Float,
        z: Float,
        axis: Axis,
    ) {
        if (axis == Axis.X) {
            estimator.onLinearAcceleration(timestamp, x, y, z)
        } else {
            estimator.onLinearAcceleration(timestamp, 0f, 0f, x)
        }
    }
}
