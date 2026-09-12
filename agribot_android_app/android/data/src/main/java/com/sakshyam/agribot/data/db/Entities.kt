package com.sakshyam.agribot.data.db

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(tableName = "field_layouts")
data class FieldLayoutEntity(
    @PrimaryKey val id: String,
    val name: String,
    val activeRowId: String,
    val startPlant: Int,
    val plantStep: Int,
    val plantCooldownSec: Double,
    val createdAt: Long,
    val updatedAt: Long,
    val isActive: Boolean,
)

@Entity(
    tableName = "field_rows",
    foreignKeys = [
        ForeignKey(
            entity = FieldLayoutEntity::class,
            parentColumns = ["id"],
            childColumns = ["fieldLayoutId"],
            onDelete = ForeignKey.CASCADE,
        ),
    ],
    indices = [Index("fieldLayoutId")],
)
data class FieldRowEntity(
    @PrimaryKey val id: String,
    val fieldLayoutId: String,
    val rowId: String,
    val rowIndex: Int,
    val plantsPerRow: Int,
    val plantSpacingM: Double,
    val rowSpacingM: Double,
    val yM: Double,
)

@Entity(tableName = "scan_runs")
data class ScanRunEntity(
    @PrimaryKey val runId: String,
    val mode: String,
    val fieldLayoutId: String,
    val startedAt: Long,
    val completedAt: Long?,
    val state: String,
    val note: String?,
    val modelBundleId: String,
    val cameraId: String?,
    val analysisWidth: Int?,
    val analysisHeight: Int?,
    val targetFps: Int,
    val appVersion: String,
)

@Entity(
    tableName = "decisions",
    foreignKeys = [
        ForeignKey(
            entity = ScanRunEntity::class,
            parentColumns = ["runId"],
            childColumns = ["runId"],
            onDelete = ForeignKey.CASCADE,
        ),
    ],
    indices = [Index("runId"), Index(value = ["runId", "sequence"], unique = true)],
)
data class DecisionEntity(
    @PrimaryKey val id: String,
    val runId: String,
    val sequence: Int,
    val timestamp: Long,
    val epochTime: Double,
    val mode: String,
    val fieldId: String,
    val rowId: String?,
    val rowSide: String?,
    val rowIndex: Int?,
    val plantColumn: Int?,
    val plantNumber: Int?,
    val plantKey: String?,
    val xM: Double?,
    val yM: Double?,
    val label: String,
    val confidence: Float,
    val rawLabel: String,
    val status: String,
    val action: String,
    val framesUsed: Int,
    val reason: String,
    val lateFrames: Int,
    val gateAvgLatencyMs: Double,
    val gateMaxLatencyMs: Double,
    val modelVersion: String,
    val manualOverride: Boolean,
    val notes: String?,
    val scanIndex: Int,
    val scanPass: Int?,
    val fieldComplete: Boolean?,
    val plannedTotalPlants: Int?,
    val plantDisplay: String?,
    val plantId: Int?,
    val trackId: String?,
    val threads: Int?,
    val fpsTarget: Double?,
    val evidencePath: String?,
    val evidenceStatus: String?,
    val treatmentStatus: String,
    val treatmentNote: String?,
    val measurementDistanceM: Double?,
    val measurementSource: String?,
    val measurementQuality: String?,
    val relativePlantWidth: Double?,
    val relativePlantHeight: Double?,
    val sizeSource: String?,
    val sizeQuality: String?,
    val gpsAccuracyM: Double?,
    val gpsFixAgeSeconds: Double?,
    val top2Margin: Double?,
    val predictionEntropy: Double?,
)

@Entity(
    tableName = "front_geometry",
    foreignKeys = [
        ForeignKey(
            entity = DecisionEntity::class,
            parentColumns = ["id"],
            childColumns = ["decisionId"],
            onDelete = ForeignKey.CASCADE,
        ),
    ],
    indices = [Index("decisionId")],
)
data class FrontGeometryEntity(
    @PrimaryKey val decisionId: String,
    val captureSegmentId: String?,
    val bboxLeft: Float?,
    val bboxTop: Float?,
    val bboxRight: Float?,
    val bboxBottom: Float?,
    val centerX: Float?,
    val centerY: Float?,
    val rowSide: String?,
    val rowAssignmentConfidence: Float?,
    val plantPositionConfidence: Float?,
    val geometryReason: String?,
    val cameraDistanceM: Double?,
    val cameraHeightM: Double?,
    val cameraTiltDegrees: Double?,
)

@Entity(tableName = "run_events")
data class RunEventEntity(
    @PrimaryKey val id: String,
    val runId: String,
    val timestamp: Long,
    val type: String,
    val payloadJson: String,
)

@Entity(tableName = "model_bundles")
data class ModelBundleEntity(
    @PrimaryKey val bundleId: String,
    val installedAt: Long,
    val classifierFile: String,
    val detectorFile: String?,
    val labelsHash: String?,
    val manifestJson: String,
    val signatureStatus: String,
    val active: Boolean,
)
