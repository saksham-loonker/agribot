package com.sakshyam.agribot.domain

import com.sakshyam.agribot.domain.logic.RunHistoryPresenter
import com.sakshyam.agribot.domain.logic.RunSummaryReducer
import com.sakshyam.agribot.domain.model.DecisionId
import com.sakshyam.agribot.domain.model.DecisionStatus
import com.sakshyam.agribot.domain.model.FieldId
import com.sakshyam.agribot.domain.model.PlantHealthAction
import com.sakshyam.agribot.domain.model.RecordedDecision
import com.sakshyam.agribot.domain.model.RecordingMode
import com.sakshyam.agribot.domain.model.Run
import com.sakshyam.agribot.domain.model.RunId
import com.sakshyam.agribot.domain.model.RunState
import java.time.Instant
import kotlin.test.Test
import kotlin.test.assertEquals

class RunHistoryPresenterTest {
    @Test
    fun formatsRunHistoryItemWithFieldModeAndUncertainRate() {
        val run = Run(
            runId = RunId("android_20260608_145500"),
            mode = RecordingMode.SIDE_SCAN,
            fieldLayoutId = FieldId("field_2"),
            startedAt = Instant.parse("2026-06-08T09:25:00Z"),
            completedAt = Instant.parse("2026-06-08T09:30:00Z"),
            state = RunState.COMPLETED,
            targetFps = 5,
            modelBundleId = "agribot-model-bundle-v001",
        )
        val summary = RunSummaryReducer.reduce(
            run.runId,
            listOf(
                decision(run.runId, 1, "Healthy", DecisionStatus.OK, PlantHealthAction.NONE),
                decision(run.runId, 2, "Late_blight", DecisionStatus.OK, PlantHealthAction.INSPECT_OR_TREAT),
                decision(run.runId, 3, "Uncertain", DecisionStatus.UNCERTAIN, PlantHealthAction.RESCAN),
            ),
        )

        val item = RunHistoryPresenter.toItem(run, summary, fieldName = "Field 2")

        assertEquals("android_20260608_145500", item.runId.value)
        assertEquals("Field 2", item.fieldName)
        assertEquals("Side Scan", item.modeLabel)
        assertEquals(1, item.sickCount)
        assertEquals("33.3%", item.uncertainRateLabel)
        assertEquals("3 decisions", item.decisionCountLabel)
    }

    private fun decision(
        runId: RunId,
        sequence: Int,
        label: String,
        status: DecisionStatus,
        action: PlantHealthAction,
    ) = RecordedDecision(
        id = DecisionId("${runId.value}_$sequence"),
        runId = runId,
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
        plantDisplay = "Field 2 / Row B / Plant $sequence",
        plantId = sequence,
        threads = 4,
        fpsTarget = 5.0,
    )
}
