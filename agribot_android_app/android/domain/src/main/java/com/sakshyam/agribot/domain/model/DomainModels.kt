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
    /** Point at one plant; the app votes over several leaves and frames. */
    CHECK_PLANT,
    /** Walk along rows; each plant is counted once and gets its own verdict. */
    WALK_ROWS,
    ;

    companion object {
        /** Tolerant parse so runs saved by older app versions still open. */
        fun parse(name: String): RecordingMode = when (name) {
            "SIDE_SCAN" -> WALK_ROWS
            "FRONT_ROW_OVERVIEW" -> CHECK_PLANT
            else -> runCatching { valueOf(name) }.getOrDefault(CHECK_PLANT)
        }
    }
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

enum class RowSide {
    LEFT,
    RIGHT,
    UNKNOWN,
}

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

data class ScanSettings(
    val defaultMode: RecordingMode = RecordingMode.CHECK_PLANT,
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
    val modelBundleId: String = "",
    val stepLengthM: Float = 0.72f,
    val stepCalibrationStatus: StepCalibrationStatus = StepCalibrationStatus.DEFAULT,
)

enum class CropMode {
    MASK,
    CENTER,
    NONE,
}

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
    /** Multi-leaf evidence behind the verdict (null for runs saved by older versions). */
    val leavesSeen: Int? = null,
    val leavesAgreeing: Int? = null,
    /** Disease found on some leaves while most leaves looked healthy. */
    val partialFinding: Boolean? = null,
    val runnerUpLabel: String? = null,
    val runnerUpConfidence: Float? = null,
    /** WGS84 position of the farmer when the plant was recorded (Walk rows, if location allowed). */
    val latitude: Double? = null,
    val longitude: Double? = null,
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
    val gpsPathStatus: String = "unavailable",
    val gpsPathPointCount: Int = 0,
    val motionEventsObserved: Boolean = false,
    val captureQualityStatus: String = "waiting",
    val detectionStatus: String = "waiting",
    val latestDecisionReason: String? = null,
    val stepLengthM: Float = 0.72f,
    val stepCalibrationStatus: String = "default",
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
    /** Model label key of the latest decision, or null when the plant was not scanned. */
    val labelKey: String?,
    /** One of: empty, ok, sick, uncertain, manual, skipped. */
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

/** Localised strings for the shareable PDF report, supplied by the UI layer. */
data class ReportText(
    val title: String,
    val generatedAt: String,
    val fieldLine: String,
    val summaryLine: String,
    val columns: List<String>,
    /** Model label key -> display name. */
    val labelNames: Map<String, String>,
    val unsure: String,
    val notSeen: String,
    val partial: String,
    val disclaimer: String,
)

data class ExportedFile(
    val displayName: String,
    val contentType: String,
    val absolutePath: String,
)

