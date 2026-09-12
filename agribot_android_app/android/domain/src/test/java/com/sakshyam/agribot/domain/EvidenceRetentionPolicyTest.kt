package com.sakshyam.agribot.domain

import com.sakshyam.agribot.domain.logic.EvidenceRetentionPolicy
import com.sakshyam.agribot.domain.model.DecisionId
import com.sakshyam.agribot.domain.model.DecisionStatus
import com.sakshyam.agribot.domain.model.EvidenceRetentionSnapshot
import com.sakshyam.agribot.domain.model.PlantHealthAction
import com.sakshyam.agribot.domain.model.RecordedDecision
import com.sakshyam.agribot.domain.model.RecordingMode
import com.sakshyam.agribot.domain.model.RunId
import java.time.Instant
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class EvidenceRetentionPolicyTest {
    private val policy = EvidenceRetentionPolicy(maxBytesPerRun = 10, maxFramesPerRun = 2)

    @Test
    fun capturesSickUncertainAndManualDecisions() {
        assertTrue(policy.shouldCapture(decision(action = PlantHealthAction.INSPECT_OR_TREAT), emptySnapshot()).capture)
        assertTrue(policy.shouldCapture(decision(status = DecisionStatus.UNCERTAIN), emptySnapshot()).capture)
        assertTrue(policy.shouldCapture(decision(manual = true), emptySnapshot()).capture)
    }

    @Test
    fun skipsHealthyDecisionAndDisabledEvidence() {
        assertFalse(policy.shouldCapture(decision(), emptySnapshot()).capture)
        assertEquals("not_required", policy.shouldCapture(decision(), emptySnapshot()).status)
        assertEquals("disabled", policy.shouldCapture(decision(manual = true), emptySnapshot(), evidenceEnabled = false).status)
    }

    @Test
    fun stopsCapturingAtStorageCaps() {
        val cappedByBytes = policy.shouldCapture(decision(manual = true), EvidenceRetentionSnapshot(bytesUsed = 10, framesUsed = 0))
        val cappedByFrames = policy.shouldCapture(decision(manual = true), EvidenceRetentionSnapshot(bytesUsed = 0, framesUsed = 2))

        assertFalse(cappedByBytes.capture)
        assertEquals("storage_cap_reached", cappedByBytes.status)
        assertFalse(cappedByFrames.capture)
        assertEquals("storage_cap_reached", cappedByFrames.status)
    }

    @Test
    fun manualSnapshotUsesSameEvidenceCapsAndToggle() {
        val allowed = policy.shouldCaptureManualSnapshot(emptySnapshot())
        val disabled = policy.shouldCaptureManualSnapshot(emptySnapshot(), evidenceEnabled = false)
        val capped = policy.shouldCaptureManualSnapshot(EvidenceRetentionSnapshot(bytesUsed = 0, framesUsed = 2))

        assertTrue(allowed.capture)
        assertEquals("capture_pending", allowed.status)
        assertFalse(disabled.capture)
        assertEquals("disabled", disabled.status)
        assertFalse(capped.capture)
        assertEquals("storage_cap_reached", capped.status)
    }

    private fun emptySnapshot() = EvidenceRetentionSnapshot(bytesUsed = 0, framesUsed = 0)

    private fun decision(
        status: DecisionStatus = DecisionStatus.OK,
        action: PlantHealthAction = PlantHealthAction.NONE,
        manual: Boolean = false,
    ) = RecordedDecision(
        id = DecisionId("decision-1"),
        runId = RunId("run-1"),
        sequence = 1,
        timestamp = Instant.EPOCH,
        epochTime = 0.0,
        mode = RecordingMode.SIDE_SCAN,
        fieldId = "Field 2",
        rowId = "B",
        rowSide = null,
        rowIndex = 2,
        plantColumn = 1,
        plantNumber = 1,
        plantKey = "Field 2/B/001",
        xM = 0.0,
        yM = 1.4,
        bboxPx = null,
        geometryConfidence = null,
        plantPositionConfidence = null,
        geometryReason = null,
        label = if (status == DecisionStatus.UNCERTAIN) "Uncertain" else "Healthy",
        confidence = 0.9f,
        rawLabel = "Healthy",
        status = status,
        action = action,
        framesUsed = 2,
        reason = "test",
        lateFrames = 0,
        gateAvgLatencyMs = 20.0,
        gateMaxLatencyMs = 21.0,
        modelVersion = "test",
        scanIndex = 1,
        scanPass = 1,
        fieldComplete = false,
        plannedTotalPlants = 57,
        plantDisplay = "Row B plant 1",
        plantId = 1,
        threads = 4,
        fpsTarget = 10.0,
        manualOverride = manual,
    )
}
