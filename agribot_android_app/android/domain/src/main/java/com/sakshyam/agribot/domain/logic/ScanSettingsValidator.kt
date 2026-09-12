package com.sakshyam.agribot.domain.logic

import com.sakshyam.agribot.domain.model.ScanSettings
import com.sakshyam.agribot.domain.model.ScanConstants
import com.sakshyam.agribot.domain.model.StepCalibrationStatus

object ScanSettingsValidator {
    fun sanitize(settings: ScanSettings): ScanSettings {
        val legacyDefaults = settings.confidenceThreshold.isLegacyDefault() &&
            settings.highConfidenceThreshold.isLegacyDefault()
        val stepCalibrationIsValid = settings.stepCalibrationStatus == StepCalibrationStatus.CALIBRATED &&
            WalkingCalibrationPolicy.isValidStepLength(settings.stepLengthM)
        return settings.copy(
            targetFps = settings.targetFps.coerceIn(MIN_FPS, MAX_FPS),
            cpuThreads = settings.cpuThreads.coerceIn(MIN_THREADS, MAX_THREADS),
            confidenceThreshold = if (legacyDefaults) {
                ScanConstants.DEFAULT_CONFIDENCE_THRESHOLD
            } else {
                settings.confidenceThreshold.coerceIn(0f, 1f)
            },
            highConfidenceThreshold = if (legacyDefaults) {
                ScanConstants.DEFAULT_HIGH_CONFIDENCE_THRESHOLD
            } else {
                settings.highConfidenceThreshold.coerceIn(0f, 1f)
            },
            maxEdge = settings.maxEdge.coerceIn(MIN_MAX_EDGE, MAX_MAX_EDGE),
            operatorDisplayName = sanitizeOperatorDisplayName(settings.operatorDisplayName),
            stepLengthM = if (stepCalibrationIsValid) settings.stepLengthM else WalkingCalibrationPolicy.DEFAULT_STEP_LENGTH_M,
            stepCalibrationStatus = if (stepCalibrationIsValid) {
                StepCalibrationStatus.CALIBRATED
            } else {
                StepCalibrationStatus.DEFAULT
            },
        )
    }

    private fun sanitizeOperatorDisplayName(value: String?): String? =
        value
            ?.trim()
            ?.filterNot { it.isISOControl() }
            ?.replace(Regex("\\s+"), " ")
            ?.take(OPERATOR_DISPLAY_NAME_MAX)
            ?.takeIf { it.isNotBlank() }

    private fun Float.isLegacyDefault(): Boolean =
        kotlin.math.abs(this - LEGACY_DEFAULT_CONFIDENCE_THRESHOLD) < 0.0001f

    private const val MIN_FPS = 3
    private const val MAX_FPS = 10
    private const val MIN_THREADS = 1
    private const val MAX_THREADS = 8
    private const val MIN_MAX_EDGE = 128
    private const val MAX_MAX_EDGE = 512
    private const val OPERATOR_DISPLAY_NAME_MAX = 64
    private const val LEGACY_DEFAULT_CONFIDENCE_THRESHOLD = 0.765f
}
