package com.sakshyam.agribot.domain

import com.sakshyam.agribot.domain.logic.RunRecoveryPlanner
import com.sakshyam.agribot.domain.model.DecisionId
import com.sakshyam.agribot.domain.model.DecisionStatus
import com.sakshyam.agribot.domain.model.FieldId
import com.sakshyam.agribot.domain.model.PlantHealthAction
import com.sakshyam.agribot.domain.model.RecordedDecision
import com.sakshyam.agribot.domain.model.RecordingMode
import com.sakshyam.agribot.domain.model.RowId
import com.sakshyam.agribot.domain.model.Run
import com.sakshyam.agribot.domain.model.RunId
import com.sakshyam.agribot.domain.model.RunState
import java.time.Instant
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class RunRecoveryPlannerTest {
    @Test
    fun recordingRunIsRecoveredAsPausedAtNextDecisionIndex() {
        val plan = RunRecoveryPlanner.plan(run(state = RunState.RECORDING), decisions = listOf(decision(1), decision(3)))

        assertTrue(plan.shouldRecover)
        assertTrue(plan.shouldMarkPaused)
        assertEquals(RunState.PAUSED, plan.recoveredState)
        assertEquals(4, plan.nextScanIndex)
        assertEquals(2, plan.decisionCount)
        assertEquals("Recovered interrupted run android_20260609_010203", plan.message)
    }

    @Test
    fun pausedRunIsRecoveredWithoutRewritingRepositoryState() {
        val plan = RunRecoveryPlanner.plan(run(state = RunState.PAUSED), decisions = listOf(decision(5)))

        assertTrue(plan.shouldRecover)
        assertFalse(plan.shouldMarkPaused)
        assertEquals(RunState.PAUSED, plan.recoveredState)
        assertEquals(6, plan.nextScanIndex)
        assertEquals(1, plan.decisionCount)
    }

    @Test
    fun completedRunIsNotRecovered() {
        val plan = RunRecoveryPlanner.plan(run(state = RunState.COMPLETED), decisions = listOf(decision(1)))

        assertFalse(plan.shouldRecover)
        assertFalse(plan.shouldMarkPaused)
        assertNull(plan.recoveredState)
    }

    private fun run(state: RunState) = Run(
        runId = RunId("android_20260609_010203"),
        mode = RecordingMode.SIDE_SCAN,
        fieldLayoutId = FieldId("field_2"),
        startedAt = Instant.parse("2026-06-09T01:02:03Z"),
        completedAt = null,
        state = state,
        targetFps = 5,
        modelBundleId = "agribot-model-bundle-v001",
    )

    private fun decision(scanIndex: Int) = RecordedDecision(
        id = DecisionId("decision_$scanIndex"),
        runId = RunId("android_20260609_010203"),
        sequence = scanIndex,
        timestamp = Instant.parse("2026-06-09T01:02:04Z"),
        epochTime = 1_780_000_000.0,
        mode = RecordingMode.SIDE_SCAN,
        fieldId = "Field 2",
        rowId = "B",
        rowSide = null,
        rowIndex = 2,
        plantColumn = scanIndex,
        plantNumber = scanIndex,
        plantKey = "Field 2|B|$scanIndex",
        xM = 0.35 * scanIndex,
        yM = 1.4,
        bboxPx = null,
        geometryConfidence = null,
        plantPositionConfidence = null,
        geometryReason = null,
        label = "Healthy",
        confidence = 0.91f,
        rawLabel = "Healthy",
        status = DecisionStatus.OK,
        action = PlantHealthAction.NONE,
        framesUsed = 2,
        reason = "primary_agreement",
        lateFrames = 0,
        gateAvgLatencyMs = 80.0,
        gateMaxLatencyMs = 90.0,
        modelVersion = "agribot-model-bundle-v001",
        scanIndex = scanIndex,
        scanPass = 1,
        fieldComplete = false,
        plannedTotalPlants = 20,
        plantDisplay = "Field 2 / Row B / Plant $scanIndex",
        plantId = scanIndex,
        threads = 4,
        fpsTarget = 5.0,
    )
}
