package com.sakshyam.agribot.data.repository

import com.sakshyam.agribot.data.db.DecisionEntity
import com.sakshyam.agribot.data.db.FieldLayoutEntity
import com.sakshyam.agribot.data.db.FieldRowEntity
import com.sakshyam.agribot.data.db.FrontGeometryEntity
import com.sakshyam.agribot.data.db.RunEventEntity
import com.sakshyam.agribot.data.db.ScanRunEntity
import com.sakshyam.agribot.domain.logic.ExportSerializer
import com.sakshyam.agribot.domain.model.BoundingBox
import com.sakshyam.agribot.domain.model.DecisionId
import com.sakshyam.agribot.domain.model.DecisionStatus
import com.sakshyam.agribot.domain.model.FieldId
import com.sakshyam.agribot.domain.model.FieldLayout
import com.sakshyam.agribot.domain.model.FieldRow
import com.sakshyam.agribot.domain.model.PlantHealthAction
import com.sakshyam.agribot.domain.model.RecordedDecision
import com.sakshyam.agribot.domain.model.RecordingMode
import com.sakshyam.agribot.domain.model.RowId
import com.sakshyam.agribot.domain.model.RowSide
import com.sakshyam.agribot.domain.model.Run
import com.sakshyam.agribot.domain.model.RunEvent
import com.sakshyam.agribot.domain.model.RunId
import com.sakshyam.agribot.domain.model.RunState
import com.sakshyam.agribot.domain.model.TreatmentStatus
import java.time.Instant

fun FieldLayout.toEntity(isActive: Boolean, now: Instant): FieldLayoutEntity =
    FieldLayoutEntity(
        id = id.value,
        name = name,
        activeRowId = activeRowId.value,
        startPlant = startPlant,
        plantStep = plantStep,
        plantCooldownSec = plantCooldownSec,
        createdAt = now.toEpochMilli(),
        updatedAt = updatedAt.toEpochMilli(),
        isActive = isActive,
    )

fun FieldLayout.toRowEntities(): List<FieldRowEntity> =
    rows.map { row ->
        FieldRowEntity(
            id = "${id.value}:${row.id.value}",
            fieldLayoutId = id.value,
            rowId = row.id.value,
            rowIndex = row.rowIndex,
            plantsPerRow = row.plantsPerRow,
            plantSpacingM = row.plantSpacingM,
            rowSpacingM = row.rowSpacingM,
            yM = row.yM,
        )
    }

fun FieldLayoutEntity.toDomain(rows: List<FieldRowEntity>): FieldLayout =
    FieldLayout(
        id = FieldId(id),
        name = name,
        activeRowId = RowId(activeRowId),
        rows = rows.map { it.toDomain() },
        startPlant = startPlant,
        plantStep = plantStep,
        plantCooldownSec = plantCooldownSec,
        updatedAt = Instant.ofEpochMilli(updatedAt),
    )

fun FieldRowEntity.toDomain(): FieldRow =
    FieldRow(
        id = RowId(rowId),
        rowIndex = rowIndex,
        plantsPerRow = plantsPerRow,
        plantSpacingM = plantSpacingM,
        rowSpacingM = rowSpacingM,
        yM = yM,
    )

fun ScanRunEntity.toDomain(): Run =
    Run(
        runId = RunId(runId),
        mode = RecordingMode.valueOf(mode),
        fieldLayoutId = FieldId(fieldLayoutId),
        startedAt = Instant.ofEpochMilli(startedAt),
        completedAt = completedAt?.let { Instant.ofEpochMilli(it) },
        state = RunState.valueOf(state),
        targetFps = targetFps,
        modelBundleId = modelBundleId,
    )

fun RecordedDecision.toEntity(): DecisionEntity =
    DecisionEntity(
        id = id.value,
        runId = runId.value,
        sequence = sequence,
        timestamp = timestamp.toEpochMilli(),
        epochTime = epochTime,
        mode = mode.name,
        fieldId = fieldId,
        rowId = rowId,
        rowSide = rowSide?.name,
        rowIndex = rowIndex,
        plantColumn = plantColumn,
        plantNumber = plantNumber,
        plantKey = plantKey,
        xM = xM,
        yM = yM,
        label = label,
        confidence = confidence,
        rawLabel = rawLabel,
        status = status.name,
        action = action.name,
        framesUsed = framesUsed,
        reason = reason,
        lateFrames = lateFrames,
        gateAvgLatencyMs = gateAvgLatencyMs,
        gateMaxLatencyMs = gateMaxLatencyMs,
        modelVersion = modelVersion,
        manualOverride = manualOverride,
        notes = notes,
        scanIndex = scanIndex,
        scanPass = scanPass,
        fieldComplete = fieldComplete,
        plannedTotalPlants = plannedTotalPlants,
        plantDisplay = plantDisplay,
        plantId = plantId,
        trackId = trackId,
        threads = threads,
        fpsTarget = fpsTarget,
        evidencePath = evidencePath,
        evidenceStatus = evidenceStatus,
        treatmentStatus = treatmentStatus.name,
        treatmentNote = treatmentNote,
        measurementDistanceM = measurementDistanceM,
        measurementSource = measurementSource,
        measurementQuality = measurementQuality,
        relativePlantWidth = relativePlantWidth,
        relativePlantHeight = relativePlantHeight,
        sizeSource = sizeSource,
        sizeQuality = sizeQuality,
        gpsAccuracyM = gpsAccuracyM,
        gpsFixAgeSeconds = gpsFixAgeSeconds,
        top2Margin = top2Margin,
        predictionEntropy = predictionEntropy,
    )

fun RecordedDecision.toGeometryEntity(): FrontGeometryEntity? =
    if (bboxPx == null && rowSide == null && geometryConfidence == null && geometryReason == null) {
        null
    } else {
        FrontGeometryEntity(
            decisionId = id.value,
            captureSegmentId = null,
            bboxLeft = bboxPx?.left,
            bboxTop = bboxPx?.top,
            bboxRight = bboxPx?.right,
            bboxBottom = bboxPx?.bottom,
            centerX = bboxPx?.centerX,
            centerY = bboxPx?.centerY,
            rowSide = rowSide?.name,
            rowAssignmentConfidence = geometryConfidence,
            plantPositionConfidence = plantPositionConfidence,
            geometryReason = geometryReason,
            cameraDistanceM = null,
            cameraHeightM = null,
            cameraTiltDegrees = null,
        )
    }

fun DecisionEntity.toDomain(geometry: FrontGeometryEntity?): RecordedDecision =
    RecordedDecision(
        id = DecisionId(id),
        runId = RunId(runId),
        sequence = sequence,
        timestamp = Instant.ofEpochMilli(timestamp),
        epochTime = epochTime,
        mode = RecordingMode.valueOf(mode),
        fieldId = fieldId,
        rowId = rowId,
        rowSide = rowSide?.let { RowSide.valueOf(it) },
        rowIndex = rowIndex,
        plantColumn = plantColumn,
        plantNumber = plantNumber,
        plantKey = plantKey,
        xM = xM,
        yM = yM,
        bboxPx = geometry?.let { geom ->
            if (geom.bboxLeft == null || geom.bboxTop == null || geom.bboxRight == null || geom.bboxBottom == null) {
                null
            } else {
                BoundingBox(geom.bboxLeft, geom.bboxTop, geom.bboxRight, geom.bboxBottom)
            }
        },
        geometryConfidence = geometry?.rowAssignmentConfidence,
        plantPositionConfidence = geometry?.plantPositionConfidence,
        geometryReason = geometry?.geometryReason,
        label = label,
        confidence = confidence,
        rawLabel = rawLabel,
        status = DecisionStatus.valueOf(status),
        action = PlantHealthAction.valueOf(action),
        framesUsed = framesUsed,
        reason = reason,
        lateFrames = lateFrames,
        gateAvgLatencyMs = gateAvgLatencyMs,
        gateMaxLatencyMs = gateMaxLatencyMs,
        modelVersion = modelVersion,
        scanIndex = scanIndex,
        scanPass = scanPass,
        fieldComplete = fieldComplete,
        plannedTotalPlants = plannedTotalPlants,
        plantDisplay = plantDisplay,
        plantId = plantId,
        trackId = trackId,
        threads = threads,
        fpsTarget = fpsTarget,
        manualOverride = manualOverride,
        notes = notes,
        evidencePath = evidencePath,
        evidenceStatus = evidenceStatus,
        treatmentStatus = treatmentStatus.toTreatmentStatus(),
        treatmentNote = treatmentNote,
        measurementDistanceM = measurementDistanceM,
        measurementSource = measurementSource,
        measurementQuality = measurementQuality,
        relativePlantWidth = relativePlantWidth,
        relativePlantHeight = relativePlantHeight,
        sizeSource = sizeSource,
        sizeQuality = sizeQuality,
        gpsAccuracyM = gpsAccuracyM,
        gpsFixAgeSeconds = gpsFixAgeSeconds,
        top2Margin = top2Margin,
        predictionEntropy = predictionEntropy,
    )

fun DecisionStatus.toPiValue(): String = ExportSerializer.statusValue(this)

fun RunEvent.toEntity(): RunEventEntity =
    RunEventEntity(
        id = id,
        runId = runId.value,
        timestamp = timestamp.toEpochMilli(),
        type = type,
        payloadJson = payloadJson,
    )

fun RunEventEntity.toDomain(): RunEvent =
    RunEvent(
        id = id,
        runId = RunId(runId),
        timestamp = Instant.ofEpochMilli(timestamp),
        type = type,
        payloadJson = payloadJson,
    )

private fun String.toTreatmentStatus(): TreatmentStatus =
    runCatching { TreatmentStatus.valueOf(this) }.getOrDefault(TreatmentStatus.NOT_TREATED)
