package com.sakshyam.agribot.domain

import com.sakshyam.agribot.domain.logic.WalkingCalibrationPolicy
import com.sakshyam.agribot.domain.model.StepCalibrationStatus
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class WalkingCalibrationPolicyTest {
    @Test
    fun acceptsReasonableWalkAndReturnsCalibratedStepLength() {
        val result = WalkingCalibrationPolicy.calibrate(knownDistanceM = 10f, steps = 14)

        assertTrue(result.accepted)
        assertEquals(StepCalibrationStatus.CALIBRATED, result.status)
        assertEquals(10f / 14f, result.stepLengthM!!, absoluteTolerance = 0.0001f)
        assertEquals(14, result.steps)
        assertNotNull(result.message)
    }

    @Test
    fun rejectsTooFewStepsInsteadOfLearningFromAStationaryOrPartialWalk() {
        val result = WalkingCalibrationPolicy.calibrate(knownDistanceM = 10f, steps = 2)

        assertFalse(result.accepted)
        assertEquals(null, result.stepLengthM)
        assertEquals(StepCalibrationStatus.DEFAULT, result.status)
    }

    @Test
    fun rejectsImpossibleStrideAndLeavesDefaultProfileSafe() {
        val result = WalkingCalibrationPolicy.calibrate(knownDistanceM = 10f, steps = 3)

        assertFalse(result.accepted)
        assertEquals(null, result.stepLengthM)
        assertEquals(StepCalibrationStatus.DEFAULT, result.status)
    }

    @Test
    fun rejectsInvalidDistanceAndNonFiniteInputs() {
        assertFalse(WalkingCalibrationPolicy.calibrate(knownDistanceM = 1f, steps = 5).accepted)
        assertFalse(WalkingCalibrationPolicy.calibrate(knownDistanceM = Float.NaN, steps = 10).accepted)
        assertFalse(WalkingCalibrationPolicy.calibrate(knownDistanceM = Float.POSITIVE_INFINITY, steps = 10).accepted)
    }
}
