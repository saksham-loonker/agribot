package com.sakshyam.agribot.featurescan

import com.sakshyam.agribot.domain.model.DecisionStatus
import com.sakshyam.agribot.domain.model.DecisionFilter
import com.sakshyam.agribot.domain.model.DiagnosticsSnapshot
import com.sakshyam.agribot.domain.model.ExportedFile
import com.sakshyam.agribot.domain.model.FieldLayout
import com.sakshyam.agribot.domain.model.FrontCorridorSetup
import com.sakshyam.agribot.domain.model.PlantDecision
import com.sakshyam.agribot.domain.model.RecordingMode
import com.sakshyam.agribot.domain.model.RowSide
import com.sakshyam.agribot.domain.model.RunDecisionRow
import com.sakshyam.agribot.domain.model.RunFieldMap
import com.sakshyam.agribot.domain.model.RunHistoryItem
import com.sakshyam.agribot.domain.model.RunId
import com.sakshyam.agribot.domain.model.RunState
import com.sakshyam.agribot.domain.model.RunSummary
import com.sakshyam.agribot.domain.model.SideScanPosition
import com.sakshyam.agribot.domain.model.ScanConstants
import com.sakshyam.agribot.domain.model.ThermalStatus
import com.sakshyam.agribot.domain.model.MeasurementQuality
import com.sakshyam.agribot.domain.model.MeasurementSource
import com.sakshyam.agribot.domain.model.StepCalibrationStatus
import com.sakshyam.agribot.domain.model.GpsPathPoint
import com.sakshyam.agribot.domain.model.GpsPathStatus
import com.sakshyam.agribot.featurescan.tracking.PlantPosition3D
import com.sakshyam.agribot.featurescan.tracking.TrackedPlant

data class ScanUiState(
    val selectedMode: RecordingMode = RecordingMode.SIDE_SCAN,
    val runState: RunState = RunState.READY,
    val activeLayout: FieldLayout? = null,
    val activeRunId: RunId? = null,
    val currentPosition: SideScanPosition? = null,
    val latestDecision: PlantDecision? = null,
    val decisionsRecorded: Int = 0,
    val targetFps: Int = SideScanDefaults.TARGET_FPS,
    val modelBundleId: String = ScanConstants.DEFAULT_MODEL_BUNDLE_ID,
    val modelStatus: String = "Model bundle pending",
    val modelReady: Boolean = false,
    val modelLoadingProgress: Float = 0f,
    val modelReadinessErrors: List<String> = emptyList(),
    val performanceStatus: String = SideScanDefaults.performanceStatus(SideScanDefaults.TARGET_FPS),
    val cameraPermissionGranted: Boolean = false,
    val frontOverviewStatus: String = "Ready",
    val frontCorridorSetup: FrontCorridorSetup? = null,
    val lastExportPath: String? = null,
    val lastExportedFile: ExportedFile? = null,
    val runHistory: List<RunHistoryItem> = emptyList(),
    val selectedRunId: RunId? = null,
    val runDetailLoading: Boolean = false,
    val selectedDecisionFilter: DecisionFilter = DecisionFilter.ALL,
    val selectedRunRows: List<RunDecisionRow> = emptyList(),
    val selectedRunFieldMap: RunFieldMap? = null,
    /** Full row/plant map for the active or most recently completed run. */
    val activeRunFieldMap: RunFieldMap? = null,
    val selectedRunSummary: RunSummary? = null,
    val pendingDeleteRunId: RunId? = null,
    val pendingFrontReview: FrontReviewUiState? = null,
    val diagnostics: DiagnosticsSnapshot? = null,
    val analysisWidth: Int? = null,
    val analysisHeight: Int? = null,
    val latestLatencyMs: Double? = null,
    val cooldownRemainingMs: Long = 0,
    val cooldownStatus: String? = null,
    val cpuThreads: Int = 4,
    val saveEvidenceFrames: Boolean = true,
    val operatorDisplayName: String = "",
    val darkMode: Boolean = false,
    val sunMode: Boolean = false,
    val gpsReferenceEnabled: Boolean = true,
    val locationPermissionGranted: Boolean = false,
    val locationPermissionRequested: Boolean = false,
    val activityRecognitionPermissionGranted: Boolean = false,
    val activityRecognitionPermissionRequested: Boolean = false,
    val showOnboarding: Boolean = true,
    val cropType: String? = null,
    val manualSnapshotStatus: String? = null,
    val captureQualityStatus: String = "waiting",
    val captureQualityMessage: String = "Point the camera at one plant",
    val captureQualityAction: String = "Keep the whole plant inside the frame",
    val detectionStatus: String = "Point at one plant",
    val storageUsedBytes: Long = 0,
    val thermalBatteryWarnings: String = "none",
    val batteryPercent: Int? = null,
    val thermalStatus: ThermalStatus = ThermalStatus.NONE,
    val confidenceThreshold: Float = ScanConstants.DEFAULT_CONFIDENCE_THRESHOLD,
    val highConfidenceThreshold: Float = ScanConstants.DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
    val syncStatus: String = "Offline — data saved locally. Export to sync.",
    val error: String? = null,
    val trackedPlants: List<TrackedPlant> = emptyList(),
    val sensorData: SensorData = SensorData(),
    val walkingCalibration: WalkingCalibrationUiState = WalkingCalibrationUiState(),
    val plantFilter: PlantFilter = PlantFilter.ALL,
    val plantSearchQuery: String = "",
    val selectedPlantId: String? = null,
    val isDataMode: Boolean = false,
) {
    val isRecording: Boolean = runState == RunState.RECORDING
    val latestStatus: DecisionStatus? = latestDecision?.status

    /**
     * Progress through the current field scan, from 0.0 to 1.0.
     * Computed from the current plant position vs. the planned total.
     */
    val fieldProgress: Float?
        get() = currentPosition?.let { pos ->
            if (pos.plannedTotalPlants > 0 && pos.plantNumber > 0) {
                (pos.plantNumber.toFloat() / pos.plannedTotalPlants).coerceIn(0f, 1f)
            } else null
        }

    val filteredPlants: List<TrackedPlant>
        get() {
            val byFilter = when (plantFilter) {
                PlantFilter.ALL -> trackedPlants
                PlantFilter.HEALTHY -> trackedPlants.filter { it.healthStatus.equals("Healthy", ignoreCase = true) }
                PlantFilter.SICK -> trackedPlants.filter {
                    !it.isOccluded &&
                        !it.healthStatus.equals("Healthy", ignoreCase = true) &&
                        !it.healthStatus.equals("Uncertain", ignoreCase = true) &&
                        !it.healthStatus.equals("Unknown", ignoreCase = true)
                }
                PlantFilter.UNCERTAIN -> trackedPlants.filter {
                    it.classificationPending ||
                        it.healthStatus.equals("Uncertain", ignoreCase = true) ||
                        it.healthStatus.equals("Unknown", ignoreCase = true)
                }
                PlantFilter.OCCLUDED -> trackedPlants.filter { it.isOccluded }
            }
            val query = plantSearchQuery.trim()
            if (query.isEmpty()) return byFilter
            return byFilter.filter { plant ->
                plant.plantId.contains(query, ignoreCase = true) ||
                plant.healthStatus.contains(query, ignoreCase = true) ||
                plant.treatmentNote?.contains(query, ignoreCase = true) == true
            }
        }

    /** Physical observations only; one-frame boxes remain a live preview. */
    val confirmedTrackedPlants: List<TrackedPlant>
        get() = trackedPlants.filter { it.isConfirmed }

    val selectedPlant: TrackedPlant?
        get() = selectedPlantId?.let { id -> trackedPlants.find { it.plantId == id } }

    fun protectsActiveRunFromDeletion(runId: RunId): Boolean {
        return activeRunId == runId && runState in setOf(RunState.RECORDING, RunState.PAUSED)
    }
}

/**
 * Filter options for the plant data viewing screen.
 */
enum class PlantFilter {
    ALL,
    HEALTHY,
    SICK,
    UNCERTAIN,
    OCCLUDED,
}

data class FrontReviewUiState(
    val capturedFrameCount: Int,
    val candidateCount: Int,
    val decisionCount: Int,
    val reason: String,
    val detectorStatus: String,
    val reviewRequired: Boolean,
    val decisions: List<FrontReviewDecisionUiState> = emptyList(),
)

data class FrontReviewDecisionUiState(
    val index: Int,
    val positionLabel: String,
    val label: String,
    val confidenceLabel: String,
    val statusLabel: String,
    val actionLabel: String,
    val reason: String,
    val rowSide: RowSide,
)

data class SensorData(
    val tilt: Float = 0f,
    val roll: Float = 0f,
    val pitch: Float = 0f,
    val speedMps: Float? = null,
    val distanceM: Float? = null,
    val stepCount: Int? = null,
    val qualityPercent: Int = 0,
    val motionStatus: String = "Motion sensors idle",
    val gpsStatus: String = "GPS reference off",
    val gpsAccuracyM: Float? = null,
    val gpsFixAgeSeconds: Float? = null,
    val gpsDistanceM: Float? = null,
    val gpsProvider: String? = null,
    val gpsPathPoints: List<GpsPathPoint> = emptyList(),
    val gpsPathStatus: GpsPathStatus = GpsPathStatus.UNAVAILABLE,
    val fieldWidthM: Float? = null,
    val fieldLengthM: Float? = null,
    val stepSensorAvailable: Boolean = false,
    val motionEventsObserved: Boolean = false,
    val measurementSource: String = "No measurement yet",
    val distanceQuality: MeasurementQuality = MeasurementQuality.NOT_STARTED,
    val distanceSource: MeasurementSource = MeasurementSource.NONE,
    val distanceMessage: String = "Start a scan to measure distance",
    val stepLengthM: Float = 0.72f,
    val stepCalibrationStatus: StepCalibrationStatus = StepCalibrationStatus.DEFAULT,
)

data class WalkingCalibrationUiState(
    val active: Boolean = false,
    val knownDistanceM: Float? = null,
    val startStepCount: Int = 0,
    val currentSteps: Int = 0,
    val message: String? = null,
)
