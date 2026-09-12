package com.sakshyam.agribot.domain

import com.sakshyam.agribot.domain.logic.RunSummaryReducer
import com.sakshyam.agribot.domain.model.DecisionId
import com.sakshyam.agribot.domain.model.DecisionStatus
import com.sakshyam.agribot.domain.model.PlantHealthAction
import com.sakshyam.agribot.domain.model.RecordedDecision
import com.sakshyam.agribot.domain.model.RecordingMode
import com.sakshyam.agribot.domain.model.RunId
import java.time.Instant
import kotlin.test.Test
import kotlin.test.assertEquals

class RunSummaryReducerTest {
    @Test
    fun separatesHealthyLabelsFromDiseaseLabelsEvenWhenStatusIsOk() {
        val runId = RunId("run-summary")
        val summary = RunSummaryReducer.reduce(
            runId,
            listOf(
                decision(runId, 1, "Healthy", DecisionStatus.OK, PlantHealthAction.NONE),
                // A persisted import may contain an OK status with a disease
                // label; the label/action semantics must still report it sick.
                decision(runId, 2, "Late_blight", DecisionStatus.OK, PlantHealthAction.NONE),
                decision(runId, 3, "Early_blight", DecisionStatus.UNCERTAIN, PlantHealthAction.RESCAN),
            ),
        )

        assertEquals(1, summary.ok)
        assertEquals(1, summary.sick)
        assertEquals(1, summary.uncertain)
    }

    private fun decision(
        runId: RunId,
        sequence: Int,
        label: String,
        status: DecisionStatus,
        action: PlantHealthAction,
    ) = RecordedDecision(
        id = DecisionId("${runId.value}-$sequence"),
        runId = runId,
        sequence = sequence,
        timestamp = Instant.parse("2026-06-08T09:25:0${sequence}Z"),
        epochTime = sequence.toDouble(),
        mode = RecordingMode.SIDE_SCAN,
        fieldId = "field",
        rowId = "row",
        rowSide = null,
        rowIndex = 1,
        plantColumn = sequence,
        plantNumber = sequence,
        plantKey = "row:$sequence",
        xM = null,
        yM = null,
        bboxPx = null,
        geometryConfidence = null,
        plantPositionConfidence = null,
        geometryReason = null,
        label = label,
        confidence = 0.9f,
        rawLabel = label,
        status = status,
        action = action,
        framesUsed = 1,
        reason = "test",
        lateFrames = 0,
        gateAvgLatencyMs = 0.0,
        gateMaxLatencyMs = 0.0,
        modelVersion = "test",
        scanIndex = sequence,
        scanPass = null,
        fieldComplete = null,
        plannedTotalPlants = null,
        plantDisplay = null,
        plantId = sequence,
        threads = null,
        fpsTarget = null,
    )
}
