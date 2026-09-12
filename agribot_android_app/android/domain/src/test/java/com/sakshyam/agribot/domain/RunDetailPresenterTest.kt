package com.sakshyam.agribot.domain

import com.sakshyam.agribot.domain.logic.RunDetailPresenter
import com.sakshyam.agribot.domain.model.DecisionFilter
import com.sakshyam.agribot.domain.model.DecisionId
import com.sakshyam.agribot.domain.model.DecisionStatus
import com.sakshyam.agribot.domain.model.PlantHealthAction
import com.sakshyam.agribot.domain.model.RecordedDecision
import com.sakshyam.agribot.domain.model.RecordingMode
import com.sakshyam.agribot.domain.model.RunId
import java.time.Instant
import kotlin.test.Test
import kotlin.test.assertEquals

class RunDetailPresenterTest {
    @Test
    fun filtersDecisionRowsByPlanRequiredStatuses() {
        val decisions = listOf(
            decision(1, "Healthy", DecisionStatus.OK, PlantHealthAction.NONE, manual = false),
            decision(2, "Late_blight", DecisionStatus.OK, PlantHealthAction.INSPECT_OR_TREAT, manual = false),
            decision(3, "Uncertain", DecisionStatus.UNCERTAIN, PlantHealthAction.RESCAN, manual = false),
            decision(4, "Healthy", DecisionStatus.MANUAL, PlantHealthAction.NONE, manual = true),
            decision(5, "Skipped", DecisionStatus.SKIPPED, PlantHealthAction.RESCAN, manual = true),
        )

        assertEquals(listOf(2), RunDetailPresenter.rows(decisions, DecisionFilter.SICK).map { it.sequence })
        assertEquals(listOf(3), RunDetailPresenter.rows(decisions, DecisionFilter.UNCERTAIN).map { it.sequence })
        assertEquals(listOf(4, 5), RunDetailPresenter.rows(decisions, DecisionFilter.MANUAL).map { it.sequence })
        assertEquals(listOf(5), RunDetailPresenter.rows(decisions, DecisionFilter.SKIPPED).map { it.sequence })
    }

    @Test
    fun formatsDecisionRowsForCompactFieldReview() {
        val row = RunDetailPresenter.rows(
            listOf(decision(2, "Late_blight", DecisionStatus.OK, PlantHealthAction.INSPECT_OR_TREAT, manual = false)),
            DecisionFilter.ALL,
        ).single()

        assertEquals(2, row.sequence)
        assertEquals("Plant 2", row.plantLabel)
        assertEquals("Late_blight", row.label)
        assertEquals("90%", row.confidenceLabel)
        assertEquals("Inspect or treat", row.actionLabel)
    }

    private fun decision(
        sequence: Int,
        label: String,
        status: DecisionStatus,
        action: PlantHealthAction,
        manual: Boolean,
    ) = RecordedDecision(
        id = DecisionId("decision_$sequence"),
        runId = RunId("android_20260608_145500"),
        sequence = sequence,
        timestamp = Instant.parse("2026-06-08T09:25:0${sequence}Z"),
        epochTime = 1780910700.0 + sequence,
        mode = RecordingMode.SIDE_SCAN,
        fieldId = "Field 2",
        rowId = "B",
        rowSide = null,
        rowIndex = 2,
        plantColumn = sequence,
        plantNumber = sequence,
        plantKey = "Field 2|B|$sequence",
        xM = sequence * 0.35,
        yM = 1.4,
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
        plannedTotalPlants = 20,
        plantDisplay = "Plant $sequence",
        plantId = sequence,
        threads = 4,
        fpsTarget = 5.0,
        manualOverride = manual,
    )
}
