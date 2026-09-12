package com.sakshyam.agribot.domain

import com.sakshyam.agribot.domain.logic.ScanSettingsValidator
import com.sakshyam.agribot.domain.model.CropMode
import com.sakshyam.agribot.domain.model.RecordingMode
import com.sakshyam.agribot.domain.model.ScanConstants
import com.sakshyam.agribot.domain.model.ScanSettings
import com.sakshyam.agribot.domain.model.StepCalibrationStatus
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse

class ScanSettingsValidatorTest {
    @Test
    fun defaultsPreservePiCompatibleSideScanProfile() {
        val settings = ScanSettingsValidator.sanitize(ScanSettings())

        assertEquals(RecordingMode.SIDE_SCAN, settings.defaultMode)
        assertEquals(5, settings.targetFps)
        assertEquals(4, settings.cpuThreads)
        assertEquals(ScanConstants.DEFAULT_CONFIDENCE_THRESHOLD, settings.confidenceThreshold)
        assertEquals(ScanConstants.DEFAULT_HIGH_CONFIDENCE_THRESHOLD, settings.highConfidenceThreshold)
        assertEquals(CropMode.MASK, settings.cropMode)
        assertEquals(256, settings.maxEdge)
    }

    @Test
    fun clampsOperatorPerformanceSettingsToNativeBounds() {
        val settings = ScanSettingsValidator.sanitize(
            ScanSettings(
                targetFps = 30,
                cpuThreads = 32,
                confidenceThreshold = -1f,
                highConfidenceThreshold = 2f,
                maxEdge = 2048,
            ),
        )

        assertEquals(10, settings.targetFps)
        assertEquals(8, settings.cpuThreads)
        assertEquals(0f, settings.confidenceThreshold)
        assertEquals(1f, settings.highConfidenceThreshold)
        assertEquals(512, settings.maxEdge)
    }

    @Test
    fun preservesEvidenceToggle() {
        val settings = ScanSettingsValidator.sanitize(ScanSettings(saveEvidenceFrames = false))

        assertFalse(settings.saveEvidenceFrames)
    }

    @Test
    fun migratesThePreviousSingleThresholdDefaultToTheStablePolicy() {
        val settings = ScanSettings(
            confidenceThreshold = 0.765f,
            highConfidenceThreshold = 0.765f,
        )

        val migrated = ScanSettingsValidator.sanitize(settings)

        assertEquals(ScanConstants.DEFAULT_CONFIDENCE_THRESHOLD, migrated.confidenceThreshold)
        assertEquals(ScanConstants.DEFAULT_HIGH_CONFIDENCE_THRESHOLD, migrated.highConfidenceThreshold)
    }

    @Test
    fun sanitizesOperatorDisplayNameForAuditPayloads() {
        val settings = ScanSettingsValidator.sanitize(
            ScanSettings(operatorDisplayName = "  Field Lead  "),
        )

        assertEquals("Field Lead", settings.operatorDisplayName)

        val blank = ScanSettingsValidator.sanitize(ScanSettings(operatorDisplayName = "   "))

        assertEquals(null, blank.operatorDisplayName)
    }

    @Test
    fun preservesOnlyAValidatedCalibratedStepLength() {
        val settings = ScanSettingsValidator.sanitize(
            ScanSettings(
                stepLengthM = 0.61f,
                stepCalibrationStatus = StepCalibrationStatus.CALIBRATED,
            ),
        )

        assertEquals(0.61f, settings.stepLengthM)
        assertEquals(StepCalibrationStatus.CALIBRATED, settings.stepCalibrationStatus)
    }

    @Test
    fun invalidOrUncalibratedStepLengthFallsBackToConservativeDefault() {
        val invalid = ScanSettingsValidator.sanitize(
            ScanSettings(
                stepLengthM = 2.5f,
                stepCalibrationStatus = StepCalibrationStatus.CALIBRATED,
            ),
        )
        val uncalibrated = ScanSettingsValidator.sanitize(
            ScanSettings(
                stepLengthM = 0.5f,
                stepCalibrationStatus = StepCalibrationStatus.DEFAULT,
            ),
        )

        assertEquals(0.72f, invalid.stepLengthM)
        assertEquals(StepCalibrationStatus.DEFAULT, invalid.stepCalibrationStatus)
        assertEquals(0.72f, uncalibrated.stepLengthM)
        assertEquals(StepCalibrationStatus.DEFAULT, uncalibrated.stepCalibrationStatus)
    }
}
