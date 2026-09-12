package com.sakshyam.agribot.domain

import com.sakshyam.agribot.domain.logic.RunFieldMapPresenter
import com.sakshyam.agribot.domain.model.DecisionId
import com.sakshyam.agribot.domain.model.DecisionStatus
import com.sakshyam.agribot.domain.model.FieldId
import com.sakshyam.agribot.domain.model.FieldLayout
import com.sakshyam.agribot.domain.model.FieldRow
import com.sakshyam.agribot.domain.model.PlantHealthAction
import com.sakshyam.agribot.domain.model.RecordedDecision
import com.sakshyam.agribot.domain.model.RecordingMode
import com.sakshyam.agribot.domain.model.RowId
import com.sakshyam.agribot.domain.model.RunId
import java.time.Instant
import kotlin.test.Test
import kotlin.test.assertEquals

class RunFieldMapPresenterTest {
    @Test
    fun buildsRowMajorFieldMapWithMissingPlantsAndActionStatus() {
        val map = RunFieldMapPresenter.map(
            layout = layout(),
            decisions = listOf(
                decision(sequence = 1, rowId = "A", plantColumn = 1, plantNumber = 10, label = "Healthy", status = DecisionStatus.OK, action = PlantHealthAction.NONE),
                decision(sequence = 2, rowId = "A", plantColumn = 2, plantNumber = 12, label = "Late_blight", status = DecisionStatus.OK, action = PlantHealthAction.INSPECT_OR_TREAT),
                decision(sequence = 3, rowId = "B", plantColumn = 1, plantNumber = 10, label = "Uncertain", status = DecisionStatus.UNCERTAIN, action = PlantHealthAction.RESCAN),
            ),
        )

        assertEquals("Field 2", map.fieldLabel)
        assertEquals(listOf("A", "B"), map.rows.map { it.rowId })
        assertEquals(listOf("ok", "sick", "empty"), map.rows[0].cells.map { it.status })
        assertEquals(listOf("uncertain", "empty", "empty"), map.rows[1].cells.map { it.status })
        assertEquals("Plant 12: Late_blight", map.rows[0].cells[1].label)
    }

    @Test
    fun latestDecisionWinsForRetakenPlant() {
        val map = RunFieldMapPresenter.map(
            layout = layout(),
            decisions = listOf(
                decision(sequence = 1, rowId = "A", plantColumn = 1, plantNumber = 10, label = "Uncertain", status = DecisionStatus.UNCERTAIN, action = PlantHealthAction.RESCAN),
                decision(sequence = 4, rowId = "A", plantColumn = 1, plantNumber = 10, label = "Healthy", status = DecisionStatus.MANUAL, action = PlantHealthAction.NONE, manual = true),
            ),
        )

        assertEquals("manual", map.rows[0].cells[0].status)
        assertEquals("Plant 10: Healthy", map.rows[0].cells[0].label)
        assertEquals(4, map.rows[0].cells[0].sequence)
    }

    private fun layout() = FieldLayout(
        id = FieldId("field_2"),
        name = "Field 2",
        activeRowId = RowId("A"),
        rows = listOf(
            FieldRow(RowId("A"), 1, 3, 0.4, 1.2, 0.0),
            FieldRow(RowId("B"), 2, 3, 0.4, 1.2, 1.2),
        ),
        startPlant = 10,
        plantStep = 2,
        plantCooldownSec = 2.0,
        updatedAt = Instant.parse("2026-06-08T09:25:00Z"),
    )

    private fun decision(
        sequence: Int,
        rowId: String,
        plantColumn: Int,
        plantNumber: Int,
        label: String,
        status: DecisionStatus,
        action: PlantHealthAction,
        manual: Boolean = false,
    ) = RecordedDecision(
        id = DecisionId("decision_$sequence"),
        runId = RunId("android_20260608_145500"),
        sequence = sequence,
        timestamp = Instant.parse("2026-06-08T09:25:00Z"),
        epochTime = 1780910700.0 + sequence,
        mode = RecordingMode.SIDE_SCAN,
        fieldId = "Field 2",
        rowId = rowId,
        rowSide = null,
        rowIndex = if (rowId == "A") 1 else 2,
        plantColumn = plantColumn,
        plantNumber = plantNumber,
        plantKey = "Field 2|$rowId|$plantNumber",
        xM = plantColumn * 0.4,
        yM = if (rowId == "A") 0.0 else 1.2,
        bboxPx = null,
        geometryConfidence = null,
        plantPositionConfidence = null,
        geometryReason = null,
        label = label,
        confidence = 0.9f,
        rawLabel = label,
        status = status,
        action = action,
        framesUsed = 2,
        reason = "primary_agreement",
        lateFrames = 0,
        gateAvgLatencyMs = 80.0,
        gateMaxLatencyMs = 90.0,
        modelVersion = "agribot-model-bundle-v001",
        scanIndex = sequence,
        scanPass = 1,
        fieldComplete = false,
        plannedTotalPlants = 6,
        plantDisplay = "Plant $plantNumber",
        plantId = plantNumber,
        threads = 4,
        fpsTarget = 5.0,
        manualOverride = manual,
    )
}
