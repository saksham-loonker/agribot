package com.sakshyam.agribot.domain.model

import java.time.Instant

@JvmInline
value class FieldId(val value: String) {
    init {
        require(value.isNotBlank()) { "FieldId cannot be blank" }
    }
}

@JvmInline
value class RowId(val value: String) {
    init {
        require(value.isNotBlank()) { "RowId cannot be blank" }
    }
}

@JvmInline
value class RunId(val value: String) {
    init {
        require(value.isNotBlank()) { "RunId cannot be blank" }
    }
}

@JvmInline
value class DecisionId(val value: String) {
    init {
        require(value.isNotBlank()) { "DecisionId cannot be blank" }
    }
}

enum class RecordingMode {
    SIDE_SCAN,
    FRONT_ROW_OVERVIEW,
}

enum class StepCalibrationStatus {
    DEFAULT,
    CALIBRATED,
}

enum class RunState {
    IDLE,
    PERMISSION_REQUIRED,
    READY,
    STARTING,
    RECORDING,
    PAUSED,
    STOPPING,
    COMPLETED,
    FAILED,
}

enum class DecisionStatus {
    OK,
    UNCERTAIN,
    MANUAL,
    SKIPPED,
}

enum class PlantHealthAction {
    NONE,
    INSPECT_OR_TREAT,
    RESCAN,
}

enum class TreatmentStatus {
    NOT_TREATED,
    TREATED,
    RESCAN_NEEDED,
    NOT_APPLICABLE,
}

data class TreatmentRecommendation(
    val diseaseLabel: String,
    val treatment: String,
    val chemical: String? = null,
    val organic: String? = null,
    val notes: String? = null,
)

enum class RowSide {
    LEFT,
    RIGHT,
    UNKNOWN,
}

data class ClassLabel(
    val index: Int,
    val displayName: String,
)

val AGRIBOT_LABEL_ORDER = listOf(
    ClassLabel(0, "Early_blight"),
    ClassLabel(1, "Healthy"),
    ClassLabel(2, "Late_blight"),
    ClassLabel(3, "Leaf Miner"),
    ClassLabel(4, "Magnesium Deficiency"),
    ClassLabel(5, "Nitrogen Deficiency"),
    ClassLabel(6, "Pottassium Deficiency"),
    ClassLabel(7, "Spotted Wilt Virus"),
)

data class FieldLayout(
    val id: FieldId,
    val name: String,
    val activeRowId: RowId,
    val rows: List<FieldRow>,
    val startPlant: Int,
    val plantStep: Int,
    val plantCooldownSec: Double,
    val updatedAt: Instant,
)

data class FieldRow(
    val id: RowId,
    val rowIndex: Int,
    val plantsPerRow: Int,
    val plantSpacingM: Double,
    val rowSpacingM: Double,
    val yM: Double,
)

data class ImportedFarmerConfig(
    val activeLayout: FieldLayout,
    val layouts: List<FieldLayout>,
    val recordingProfile: RecordingProfile,
)

data class RecordingProfile(
    val fps: Double,
    val threads: Int,
    val classifierSourceNote: String?,
    val cropMode: CropMode = CropMode.MASK,
    val confidenceThreshold: Float = ScanConstants.DEFAULT_CONFIDENCE_THRESHOLD,
    val highConfidenceThreshold: Float = ScanConstants.DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
    val cropScale: Float = 0.55f,
    val cropPad: Float = 0.05f,
    val maxEdge: Int = 256,
)

data class ScanSettings(
    val defaultMode: RecordingMode = RecordingMode.SIDE_SCAN,
    val targetFps: Int = 5,
    val cpuThreads: Int = 4,
    val confidenceThreshold: Float = ScanConstants.DEFAULT_CONFIDENCE_THRESHOLD,
    val highConfidenceThreshold: Float = ScanConstants.DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
    val cropMode: CropMode = CropMode.MASK,
    val maxEdge: Int = 256,
    val saveEvidenceFrames: Boolean = true,
    val operatorDisplayName: String? = null,
    val darkMode: Boolean = false,
    val sunMode: Boolean = false,
    val gpsReferenceEnabled: Boolean = true,
    val showOnboarding: Boolean = true,
    val cropType: String? = null,
    val modelBundleId: String = "agribot-model-bundle-v001",
    val stepLengthM: Float = 0.72f,
    val stepCalibrationStatus: StepCalibrationStatus = StepCalibrationStatus.DEFAULT,
)

enum class CropMode {
    MASK,
    CENTER,
    NONE,
}

data class FramePrediction(
    val label: String,
    val confidence: Float,
    val rawLabel: String,
    val isDisease: Boolean,
    val isUncertain: Boolean,
    val latencyMs: Double,
    val modelVersion: String,
    /** Difference between the top two class probabilities, when the model exposes them. */
    val top2Margin: Float? = null,
    /** Normalized entropy of the class distribution, when available. */
    val entropy: Float? = null,
    /** Probability mass on the "Healthy" class, when the model exposes it. */
    val healthyConfidence: Float? = null,
)

data class AnalysisFrame(
    val width: Int,
    val height: Int,
    val rotationDegrees: Int,
    val timestampNanos: Long,
    val rgbPixels: IntArray,
    val jpegBytes: ByteArray,
) {
    init {
        require(width > 0) { "width must be positive" }
        require(height > 0) { "height must be positive" }
        require(rgbPixels.size == width * height) { "rgbPixels size must equal width * height" }
    }
}

data class PlantDecision(
    val label: String,
    val confidence: Float,
    val status: DecisionStatus,
    val framesUsed: Int,
    val reason: String,
)

data class SideScanPosition(
    val fieldId: String,
    val rowId: String,
    val rowIndex: Int,
    val plantColumn: Int,
    val plantNumber: Int,
    val plantKey: String,
    val xM: Double,
    val yM: Double,
    val scanIndex: Int,
    val scanPass: Int,
    val fieldComplete: Boolean,
    val plannedTotalPlants: Int,
    val plantDisplay: String,
)

data class BoundingBox(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
) {
    val centerX: Float get() = (left + right) / 2f
    val centerY: Float get() = (top + bottom) / 2f
    val width: Float get() = right - left
    val height: Float get() = bottom - top
}

data class FrontCaptureCalibration(
    val cameraHeightM: Double,
    val cameraDistanceToNearestRowM: Double,
    val cameraTiltDegrees: Double,
    val rowSpacingM: Double,
    val plantSpacingM: Double,
    val leftRowId: String,
    val rightRowId: String,
    val nearestLeftPlantNumber: Int,
    val nearestRightPlantNumber: Int,
    val guideRailLeftPx: Float,
    val guideRailRightPx: Float,
    val frameWidthPx: Int,
    val frameHeightPx: Int,
    val calibrationQuality: Float,
)

data class FrontCorridorRows(
    val left: FieldRow,
    val right: FieldRow,
)

data class FrontCorridorSetup(
    val leftRowId: String,
    val rightRowId: String,
    val nearestLeftPlantNumber: Int,
    val nearestRightPlantNumber: Int,
)

data class FrontGeometryAssignment(
    val bboxPx: BoundingBox,
    val rowSide: RowSide,
    val rowId: String?,
    val plantColumnEstimate: Int?,
    val plantNumber: Int?,
    val rowAssignmentConfidence: Float,
    val plantPositionConfidence: Float,
    val geometryReason: String,
)

data class FrontOverviewCandidate(
    val bboxPx: BoundingBox,
    val detectorConfidence: Float,
    val label: String,
    val confidence: Float,
    val rawLabel: String = label,
    val top2Margin: Float? = null,
    val entropy: Float? = null,
)

data class FrontOverviewDecisionDraft(
    val bboxPx: BoundingBox,
    val rowSide: RowSide,
    val rowId: String?,
    val plantColumnEstimate: Int?,
    val plantNumber: Int?,
    val label: String,
    val rawLabel: String,
    val confidence: Float,
    val status: DecisionStatus,
    val action: PlantHealthAction,
    val rowAssignmentConfidence: Float,
    val plantPositionConfidence: Float,
    val geometryReason: String,
    val reason: String,
)

data class FrontBurstReview(
    val capturedFrameCount: Int,
    val decisions: List<FrontOverviewDecisionDraft>,
    val reviewRequired: Boolean,
    val reason: String,
)

data class RecordedDecision(
    val id: DecisionId,
    val runId: RunId,
    val sequence: Int,
    val timestamp: Instant,
    val epochTime: Double,
    val mode: RecordingMode,
    val fieldId: String,
    val rowId: String?,
    val rowSide: RowSide?,
    val rowIndex: Int?,
    val plantColumn: Int?,
    val plantNumber: Int?,
    val plantKey: String?,
    val xM: Double?,
    val yM: Double?,
    val bboxPx: BoundingBox?,
    val geometryConfidence: Float?,
    val plantPositionConfidence: Float?,
    val geometryReason: String?,
    val label: String,
    val confidence: Float,
    val rawLabel: String,
    val status: DecisionStatus,
    val action: PlantHealthAction,
    val framesUsed: Int,
    val reason: String,
    val lateFrames: Int,
    val gateAvgLatencyMs: Double,
    val gateMaxLatencyMs: Double,
    val modelVersion: String,
    val scanIndex: Int,
    val scanPass: Int?,
    val fieldComplete: Boolean?,
    val plannedTotalPlants: Int?,
    val plantDisplay: String?,
    val plantId: Int?,
    val threads: Int?,
    val fpsTarget: Double?,
    val manualOverride: Boolean = false,
    val notes: String? = null,
    val evidencePath: String? = null,
    val evidenceStatus: String? = null,
    val treatmentStatus: TreatmentStatus = TreatmentStatus.NOT_TREATED,
    val treatmentNote: String? = null,
    /** Snapshot of the measurement evidence available when this decision was recorded. */
    val measurementDistanceM: Double? = null,
    val measurementSource: String? = null,
    val measurementQuality: String? = null,
    val relativePlantWidth: Double? = null,
    val relativePlantHeight: Double? = null,
    val sizeSource: String? = null,
    val sizeQuality: String? = null,
    val gpsAccuracyM: Double? = null,
    val gpsFixAgeSeconds: Double? = null,
    val top2Margin: Double? = null,
    val predictionEntropy: Double? = null,
    /** Stable in-run tracker identity used to audit a decision's evidence chain. */
    val trackId: String? = null,
)

data class RunConfig(
    val runId: RunId,
    val mode: RecordingMode,
    val fieldLayoutId: FieldId,
    val targetFps: Int,
    val modelBundleId: String,
)

data class Run(
    val runId: RunId,
    val mode: RecordingMode,
    val fieldLayoutId: FieldId,
    val startedAt: Instant,
    val completedAt: Instant?,
    val state: RunState,
    val targetFps: Int,
    val modelBundleId: String,
)

data class RunSummary(
    val runId: RunId,
    val decisions: Int,
    val ok: Int,
    val sick: Int,
    val uncertain: Int,
    val uncertainRate: Double,
    val avgConfidence: Double,
    val maxConfidence: Float,
    val totalLateFrames: Int,
    val labels: Map<String, Int>,
    val statuses: Map<DecisionStatus, Int>,
    val latestDecision: RecordedDecision?,
)

data class RunHistoryItem(
    val runId: RunId,
    val startedAt: Instant,
    val fieldName: String,
    val modeLabel: String,
    val stateLabel: String,
    val sickCount: Int,
    val uncertainRateLabel: String,
    val decisionCountLabel: String,
)

data class DiagnosticsSnapshot(
    val appVersion: String,
    val modelBundleId: String,
    val cameraId: String,
    val supportedSizes: String,
    val analysisResolution: String,
    val cpuThreadProfile: String,
    val latestBenchmarkResult: String,
    val readinessSummary: String,
    val thermalBatteryWarnings: String,
    val storageUse: String,
    val permissionStatus: String,
    val runState: String,
    val lastExportPath: String?,
    val measurementSource: String = "none",
    val measurementQuality: String = "not_started",
    val gpsStatus: String = "not_started",
    val gpsProvider: String? = null,
    val gpsPathStatus: GpsPathStatus = GpsPathStatus.UNAVAILABLE,
    val gpsPathPointCount: Int = 0,
    val motionEventsObserved: Boolean = false,
    val captureQualityStatus: String = "waiting",
    val detectionStatus: String = "waiting",
    val latestDecisionReason: String? = null,
    val stepLengthM: Float = 0.72f,
    val stepCalibrationStatus: String = "default",
)

enum class DecisionFilter {
    ALL,
    SICK,
    UNCERTAIN,
    MANUAL,
    SKIPPED,
}

data class RunDecisionRow(
    val decisionId: DecisionId,
    val sequence: Int,
    val plantLabel: String,
    val label: String,
    val confidenceLabel: String,
    val statusLabel: String,
    val actionLabel: String,
    val reason: String,
)

data class RunFieldMap(
    val fieldLabel: String,
    val rows: List<RunFieldMapRow>,
)

data class RunFieldMapRow(
    val rowId: String,
    val rowLabel: String,
    val cells: List<RunFieldMapCell>,
)

data class RunFieldMapCell(
    val plantNumber: Int,
    val label: String,
    val status: String,
    val sequence: Int?,
)

data class RunEvent(
    val id: String,
    val runId: RunId,
    val timestamp: Instant,
    val type: String,
    val payloadJson: String,
)

data class EvidenceRetentionSnapshot(
    val bytesUsed: Long,
    val framesUsed: Int,
)

data class EvidenceCaptureResult(
    val path: String?,
    val status: String,
)

data class ExportedFile(
    val displayName: String,
    val contentType: String,
    val absolutePath: String,
)

data class ModelManifest(
    val bundleId: String,
    val createdAt: String,
    val classifier: ModelEntry,
    val detector: ModelEntry?,
    val labelsSha256: String?,
    val minAndroidSdk: Int,
    val cpuDefault: Boolean,
    val requiresNetwork: Boolean,
)

data class ModelEntry(
    val file: String,
    val source: String,
    val inputWidth: Int,
    val inputHeight: Int,
    val labelsFile: String?,
    val sha256: String?,
    val confidenceThreshold: Float?,
    val highConfidenceThreshold: Float?,
    /** Temperature used to temper the classifier's overconfident output. */
    val confidenceTemperature: Float = ScanConstants.DEFAULT_CONFIDENCE_TEMPERATURE,
    val outputCandidates: Int? = null,
    /** Explicit TFLite output contract; never infer this from dimension order at runtime. */
    val outputLayout: String? = null,
    /** Number of detector classes for raw YOLO outputs (4 + classes [+ objectness]). */
    val classCount: Int? = null,
    /** Whether raw detector rows contain an objectness field at index 4. */
    val hasObjectness: Boolean = false,
    /** Detector coordinate representation, for example model_input_pixels or normalized. */
    val coordinateSpace: String? = null,
    /** Input channel order and normalization contract recorded for diagnostics. */
    val colorSpace: String? = null,
    val normalization: String? = null,
)

data class ValidationResult(
    val isValid: Boolean,
    val errors: List<String>,
)

data class ModelBundleReadiness(
    val isReady: Boolean,
    val statusText: String,
    val errors: List<String>,
    val progress: Float = 0f,
)

data class PerformanceProfile(
    val targetFps: Int,
    val resolutionStep: Int,
)

enum class ThermalStatus {
    NONE,
    LIGHT,
    MODERATE,
    SEVERE,
    CRITICAL,
}

data class ThermalPolicyResult(
    val profile: PerformanceProfile,
    val pauseAfterCurrentGroup: Boolean,
    val reason: String,
)
