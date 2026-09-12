package com.sakshyam.agribot.data.settings

import androidx.datastore.preferences.core.preferencesOf
import com.sakshyam.agribot.domain.model.CropMode
import com.sakshyam.agribot.domain.model.RecordingMode
import com.sakshyam.agribot.domain.model.ScanSettings
import com.sakshyam.agribot.domain.model.StepCalibrationStatus
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse

class ScanSettingsPreferencesMapperTest {
    @Test
    fun mapsStoredPreferencesToDomainSettings() {
        val prefs = preferencesOf(
            ScanSettingsPreferencesKeys.DEFAULT_MODE to RecordingMode.FRONT_ROW_OVERVIEW.name,
            ScanSettingsPreferencesKeys.TARGET_FPS to 7,
            ScanSettingsPreferencesKeys.CPU_THREADS to 6,
            ScanSettingsPreferencesKeys.CONFIDENCE to 0.7f,
            ScanSettingsPreferencesKeys.HIGH_CONFIDENCE to 0.9f,
            ScanSettingsPreferencesKeys.CROP_MODE to CropMode.CENTER.name,
            ScanSettingsPreferencesKeys.MAX_EDGE to 320,
            ScanSettingsPreferencesKeys.SAVE_EVIDENCE to false,
            ScanSettingsPreferencesKeys.GPS_REFERENCE to false,
            ScanSettingsPreferencesKeys.OPERATOR_DISPLAY_NAME to "Field Lead",
            ScanSettingsPreferencesKeys.STEP_LENGTH_M to 0.61f,
            ScanSettingsPreferencesKeys.STEP_CALIBRATION_STATUS to StepCalibrationStatus.CALIBRATED.name,
        )

        val settings = ScanSettingsPreferencesMapper.fromPreferences(prefs)

        assertEquals(RecordingMode.FRONT_ROW_OVERVIEW, settings.defaultMode)
        assertEquals(7, settings.targetFps)
        assertEquals(6, settings.cpuThreads)
        assertEquals(0.7f, settings.confidenceThreshold)
        assertEquals(0.9f, settings.highConfidenceThreshold)
        assertEquals(CropMode.CENTER, settings.cropMode)
        assertEquals(320, settings.maxEdge)
        assertFalse(settings.saveEvidenceFrames)
        assertFalse(settings.gpsReferenceEnabled)
        assertEquals("Field Lead", settings.operatorDisplayName)
        assertEquals(0.61f, settings.stepLengthM)
        assertEquals(StepCalibrationStatus.CALIBRATED, settings.stepCalibrationStatus)
    }

    @Test
    fun sanitizesInvalidStoredPreferences() {
        val prefs = preferencesOf(
            ScanSettingsPreferencesKeys.DEFAULT_MODE to "STALE_MODE",
            ScanSettingsPreferencesKeys.TARGET_FPS to 99,
            ScanSettingsPreferencesKeys.CPU_THREADS to 0,
            ScanSettingsPreferencesKeys.CONFIDENCE to -1f,
            ScanSettingsPreferencesKeys.HIGH_CONFIDENCE to 2f,
            ScanSettingsPreferencesKeys.CROP_MODE to "STALE_CROP",
            ScanSettingsPreferencesKeys.MAX_EDGE to 9999,
            ScanSettingsPreferencesKeys.STEP_LENGTH_M to 2.5f,
            ScanSettingsPreferencesKeys.STEP_CALIBRATION_STATUS to StepCalibrationStatus.CALIBRATED.name,
        )

        val settings = ScanSettingsPreferencesMapper.fromPreferences(prefs)

        assertEquals(RecordingMode.SIDE_SCAN, settings.defaultMode)
        assertEquals(10, settings.targetFps)
        assertEquals(1, settings.cpuThreads)
        assertEquals(0f, settings.confidenceThreshold)
        assertEquals(1f, settings.highConfidenceThreshold)
        assertEquals(CropMode.MASK, settings.cropMode)
        assertEquals(512, settings.maxEdge)
        assertEquals(0.72f, settings.stepLengthM)
        assertEquals(StepCalibrationStatus.DEFAULT, settings.stepCalibrationStatus)
    }

    @Test
    fun serializesSanitizedDomainSettings() {
        val pairs = ScanSettingsPreferencesMapper.toPreferencePairs(
            ScanSettings(
                defaultMode = RecordingMode.FRONT_ROW_OVERVIEW,
                targetFps = 99,
                cpuThreads = 99,
                cropMode = CropMode.NONE,
                saveEvidenceFrames = false,
                gpsReferenceEnabled = false,
                operatorDisplayName = "  Lead Operator  ",
                stepLengthM = 0.61f,
                stepCalibrationStatus = StepCalibrationStatus.CALIBRATED,
            ),
        ).toMap()

        assertEquals(RecordingMode.FRONT_ROW_OVERVIEW.name, pairs[ScanSettingsPreferencesKeys.DEFAULT_MODE])
        assertEquals(10, pairs[ScanSettingsPreferencesKeys.TARGET_FPS])
        assertEquals(8, pairs[ScanSettingsPreferencesKeys.CPU_THREADS])
        assertEquals(CropMode.NONE.name, pairs[ScanSettingsPreferencesKeys.CROP_MODE])
        assertEquals(false, pairs[ScanSettingsPreferencesKeys.SAVE_EVIDENCE])
        assertEquals(false, pairs[ScanSettingsPreferencesKeys.GPS_REFERENCE])
        assertEquals("Lead Operator", pairs[ScanSettingsPreferencesKeys.OPERATOR_DISPLAY_NAME])
        assertEquals(0.61f, pairs[ScanSettingsPreferencesKeys.STEP_LENGTH_M])
        assertEquals(StepCalibrationStatus.CALIBRATED.name, pairs[ScanSettingsPreferencesKeys.STEP_CALIBRATION_STATUS])
    }
}
