package com.sakshyam.agribot.data.settings

import android.content.Context
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.floatPreferencesKey
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.preferencesDataStore
import com.sakshyam.agribot.domain.logic.ScanSettingsValidator
import com.sakshyam.agribot.domain.model.CropMode
import com.sakshyam.agribot.domain.model.RecordingMode
import com.sakshyam.agribot.domain.model.ScanSettings
import com.sakshyam.agribot.domain.model.ScanConstants
import com.sakshyam.agribot.domain.model.StepCalibrationStatus
import com.sakshyam.agribot.domain.repository.ScanSettingsRepository
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.scanDataStore by preferencesDataStore(name = "scan_settings")

class ScanSettingsDataStore @Inject constructor(
    @ApplicationContext private val context: Context,
) : ScanSettingsRepository {
    override fun observeSettings(): Flow<ScanSettings> = context.scanDataStore.data.map { prefs ->
        ScanSettingsPreferencesMapper.fromPreferences(prefs)
    }

    override suspend fun saveSettings(settings: ScanSettings) {
        val safe = ScanSettingsValidator.sanitize(settings)
        context.scanDataStore.edit { prefs ->
            prefs[ScanSettingsPreferencesKeys.DEFAULT_MODE] = safe.defaultMode.name
            prefs[ScanSettingsPreferencesKeys.TARGET_FPS] = safe.targetFps
            prefs[ScanSettingsPreferencesKeys.CPU_THREADS] = safe.cpuThreads
            prefs[ScanSettingsPreferencesKeys.CONFIDENCE] = safe.confidenceThreshold
            prefs[ScanSettingsPreferencesKeys.HIGH_CONFIDENCE] = safe.highConfidenceThreshold
            prefs[ScanSettingsPreferencesKeys.CROP_MODE] = safe.cropMode.name
            prefs[ScanSettingsPreferencesKeys.MAX_EDGE] = safe.maxEdge
            prefs[ScanSettingsPreferencesKeys.SAVE_EVIDENCE] = safe.saveEvidenceFrames
            prefs[ScanSettingsPreferencesKeys.DARK_MODE] = safe.darkMode
            prefs[ScanSettingsPreferencesKeys.SUN_MODE] = safe.sunMode
            prefs[ScanSettingsPreferencesKeys.GPS_REFERENCE] = safe.gpsReferenceEnabled
            prefs[ScanSettingsPreferencesKeys.SHOW_ONBOARDING] = safe.showOnboarding
            val cropType = safe.cropType
            if (cropType == null) {
                prefs.remove(ScanSettingsPreferencesKeys.CROP_TYPE)
            } else {
                prefs[ScanSettingsPreferencesKeys.CROP_TYPE] = cropType
            }
            val operatorDisplayName = safe.operatorDisplayName
            if (operatorDisplayName == null) {
                prefs.remove(ScanSettingsPreferencesKeys.OPERATOR_DISPLAY_NAME)
            } else {
                prefs[ScanSettingsPreferencesKeys.OPERATOR_DISPLAY_NAME] = operatorDisplayName
            }
            prefs[ScanSettingsPreferencesKeys.MODEL_BUNDLE_ID] = safe.modelBundleId
            prefs[ScanSettingsPreferencesKeys.STEP_LENGTH_M] = safe.stepLengthM
            prefs[ScanSettingsPreferencesKeys.STEP_CALIBRATION_STATUS] = safe.stepCalibrationStatus.name
        }
    }
}

internal object ScanSettingsPreferencesMapper {
    fun fromPreferences(prefs: Preferences): ScanSettings =
        ScanSettingsValidator.sanitize(
            ScanSettings(
                defaultMode = prefs[ScanSettingsPreferencesKeys.DEFAULT_MODE].toRecordingMode(),
                targetFps = prefs[ScanSettingsPreferencesKeys.TARGET_FPS] ?: 5,
                cpuThreads = prefs[ScanSettingsPreferencesKeys.CPU_THREADS] ?: 4,
                confidenceThreshold = prefs[ScanSettingsPreferencesKeys.CONFIDENCE] ?: ScanConstants.DEFAULT_CONFIDENCE_THRESHOLD,
                highConfidenceThreshold = prefs[ScanSettingsPreferencesKeys.HIGH_CONFIDENCE] ?: ScanConstants.DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
                cropMode = prefs[ScanSettingsPreferencesKeys.CROP_MODE].toCropMode(),
                maxEdge = prefs[ScanSettingsPreferencesKeys.MAX_EDGE] ?: 256,
                saveEvidenceFrames = prefs[ScanSettingsPreferencesKeys.SAVE_EVIDENCE] ?: true,
                operatorDisplayName = prefs[ScanSettingsPreferencesKeys.OPERATOR_DISPLAY_NAME],
                darkMode = prefs[ScanSettingsPreferencesKeys.DARK_MODE] ?: false,
                sunMode = prefs[ScanSettingsPreferencesKeys.SUN_MODE] ?: false,
                gpsReferenceEnabled = prefs[ScanSettingsPreferencesKeys.GPS_REFERENCE] ?: true,
                showOnboarding = prefs[ScanSettingsPreferencesKeys.SHOW_ONBOARDING] ?: true,
                cropType = prefs[ScanSettingsPreferencesKeys.CROP_TYPE],
                modelBundleId = prefs[ScanSettingsPreferencesKeys.MODEL_BUNDLE_ID] ?: "agribot-model-bundle-v001",
                stepLengthM = prefs[ScanSettingsPreferencesKeys.STEP_LENGTH_M] ?: 0.72f,
                stepCalibrationStatus = prefs[ScanSettingsPreferencesKeys.STEP_CALIBRATION_STATUS].toStepCalibrationStatus(),
            ),
        )

    fun toPreferencePairs(settings: ScanSettings): List<Pair<Preferences.Key<*>, Any>> {
        val safe = ScanSettingsValidator.sanitize(settings)
        return listOfNotNull(
            Pair(ScanSettingsPreferencesKeys.DEFAULT_MODE, safe.defaultMode.name),
            Pair(ScanSettingsPreferencesKeys.TARGET_FPS, safe.targetFps),
            Pair(ScanSettingsPreferencesKeys.CPU_THREADS, safe.cpuThreads),
            Pair(ScanSettingsPreferencesKeys.CONFIDENCE, safe.confidenceThreshold),
            Pair(ScanSettingsPreferencesKeys.HIGH_CONFIDENCE, safe.highConfidenceThreshold),
            Pair(ScanSettingsPreferencesKeys.CROP_MODE, safe.cropMode.name),
            Pair(ScanSettingsPreferencesKeys.MAX_EDGE, safe.maxEdge),
            Pair(ScanSettingsPreferencesKeys.SAVE_EVIDENCE, safe.saveEvidenceFrames),
            Pair(ScanSettingsPreferencesKeys.GPS_REFERENCE, safe.gpsReferenceEnabled),
            safe.operatorDisplayName?.let { Pair(ScanSettingsPreferencesKeys.OPERATOR_DISPLAY_NAME, it) },
            Pair(ScanSettingsPreferencesKeys.STEP_LENGTH_M, safe.stepLengthM),
            Pair(ScanSettingsPreferencesKeys.STEP_CALIBRATION_STATUS, safe.stepCalibrationStatus.name),
        )
    }

    private fun String?.toRecordingMode(): RecordingMode =
        runCatching {
            if (this == null) RecordingMode.SIDE_SCAN else RecordingMode.valueOf(this)
        }.getOrDefault(RecordingMode.SIDE_SCAN)

    private fun String?.toCropMode(): CropMode =
        runCatching {
            if (this == null) CropMode.MASK else CropMode.valueOf(this)
        }.getOrDefault(CropMode.MASK)

    private fun String?.toStepCalibrationStatus(): StepCalibrationStatus =
        runCatching {
            if (this == null) StepCalibrationStatus.DEFAULT else StepCalibrationStatus.valueOf(this)
        }.getOrDefault(StepCalibrationStatus.DEFAULT)
}

internal object ScanSettingsPreferencesKeys {
    val DEFAULT_MODE = stringPreferencesKey("default_mode")
    val TARGET_FPS = intPreferencesKey("target_fps")
    val CPU_THREADS = intPreferencesKey("cpu_threads")
    val CONFIDENCE = floatPreferencesKey("confidence")
    val HIGH_CONFIDENCE = floatPreferencesKey("high_confidence")
    val CROP_MODE = stringPreferencesKey("crop_mode")
    val MAX_EDGE = intPreferencesKey("max_edge")
    val SAVE_EVIDENCE = booleanPreferencesKey("save_evidence")
    val OPERATOR_DISPLAY_NAME = stringPreferencesKey("operator_display_name")
    val DARK_MODE = booleanPreferencesKey("dark_mode")
    val SUN_MODE = booleanPreferencesKey("sun_mode")
    val GPS_REFERENCE = booleanPreferencesKey("gps_reference")
    val SHOW_ONBOARDING = booleanPreferencesKey("show_onboarding")
    val CROP_TYPE = stringPreferencesKey("crop_type")
    val MODEL_BUNDLE_ID = stringPreferencesKey("model_bundle_id")
    val STEP_LENGTH_M = floatPreferencesKey("step_length_m")
    val STEP_CALIBRATION_STATUS = stringPreferencesKey("step_calibration_status")
}
