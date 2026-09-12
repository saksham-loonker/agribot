package com.sakshyam.agribot.domain.logic

import com.sakshyam.agribot.domain.model.StepCalibrationStatus

/**
 * Guardrails for the optional walking calibration used by phone-only distance
 * estimation. Calibration is an estimate of the operator's stride, not a
 * replacement for a surveyed field dimension or a fresh GPS reference.
 */
object WalkingCalibrationPolicy {
    const val DEFAULT_STEP_LENGTH_M = 0.72f
    const val MIN_STEP_LENGTH_M = 0.35f
    const val MAX_STEP_LENGTH_M = 1.50f
    const val MIN_WALK_DISTANCE_M = 3f
    const val MAX_WALK_DISTANCE_M = 100f
    const val MIN_STEPS = 3

    fun calibrate(knownDistanceM: Float, steps: Int): WalkingCalibrationResult {
        if (!isValidKnownDistance(knownDistanceM)) {
            return rejected("Use a known walking distance from 3 to 100 metres")
        }
        if (steps < MIN_STEPS) {
            return rejected("Walk the full distance so at least 3 steps are counted")
        }

        val stepLengthM = knownDistanceM / steps
        if (!stepLengthM.isFinite() ||
            stepLengthM !in MIN_STEP_LENGTH_M..MAX_STEP_LENGTH_M
        ) {
            return rejected("The walk did not produce a realistic stride; repeat it steadily")
        }

        return WalkingCalibrationResult(
            accepted = true,
            stepLengthM = stepLengthM,
            status = StepCalibrationStatus.CALIBRATED,
            steps = steps,
            knownDistanceM = knownDistanceM,
            message = "Walking calibration saved for this phone and operator",
        )
    }

    fun isValidStepLength(value: Float): Boolean =
        value.isFinite() && value in MIN_STEP_LENGTH_M..MAX_STEP_LENGTH_M

    fun isValidKnownDistance(value: Float): Boolean =
        value.isFinite() && value in MIN_WALK_DISTANCE_M..MAX_WALK_DISTANCE_M

    private fun rejected(message: String): WalkingCalibrationResult =
        WalkingCalibrationResult(
            accepted = false,
            stepLengthM = null,
            status = StepCalibrationStatus.DEFAULT,
            steps = 0,
            knownDistanceM = null,
            message = message,
        )
}

data class WalkingCalibrationResult(
    val accepted: Boolean,
    val stepLengthM: Float?,
    val status: StepCalibrationStatus,
    val steps: Int,
    val knownDistanceM: Float?,
    val message: String,
)
