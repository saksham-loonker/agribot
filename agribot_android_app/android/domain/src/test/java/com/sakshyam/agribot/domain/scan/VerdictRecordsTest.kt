package com.sakshyam.agribot.domain.scan

import com.sakshyam.agribot.domain.logic.RunSummaryReducer
import com.sakshyam.agribot.domain.model.DecisionId
import com.sakshyam.agribot.domain.model.DecisionStatus
import com.sakshyam.agribot.domain.model.PlantHealthAction
import com.sakshyam.agribot.domain.model.RecordingMode
import com.sakshyam.agribot.domain.model.RunId
import java.time.Instant
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue
import org.junit.Test

class VerdictRecordsTest {
    private val labels = listOf("Early_blight", "Healthy", "Late_blight", "Other")
    private val run = RunId("r")

    private fun verdict(kind: VerdictKind, label: Int?, leaves: Int = 3, partial: Boolean = false) =
        PlantVerdict(kind, label, 0.8f, 1, 0.1f, leaves, 2, 4, partial)

    private fun record(v: PlantVerdict, seq: Int = 1, row: Int? = 1, plant: Int? = 1) =
        VerdictRecords.toDecision(DecisionId("d$seq"), run, seq, RecordingMode.WALK_ROWS, PlantPlace("f", row, plant), v, labels, "b", Instant.EPOCH)

    @Test fun `verdict kinds map to status and action`() {
        assertEquals(DecisionStatus.OK to PlantHealthAction.NONE, record(verdict(VerdictKind.HEALTHY, 1)).let { it.status to it.action })
        assertEquals(DecisionStatus.OK to PlantHealthAction.INSPECT_OR_TREAT, record(verdict(VerdictKind.DISEASE, 2)).let { it.status to it.action })
        assertEquals(DecisionStatus.OK to PlantHealthAction.INSPECT_OR_TREAT, record(verdict(VerdictKind.UNKNOWN_CONDITION, 3)).let { it.status to it.action })
        assertEquals(DecisionStatus.UNCERTAIN to PlantHealthAction.RESCAN, record(verdict(VerdictKind.UNSURE, 0)).let { it.status to it.action })
        val notSeen = record(verdict(VerdictKind.NEED_MORE_VIEWS, null, leaves = 0))
        assertEquals(DecisionStatus.SKIPPED, notSeen.status)
        assertEquals("", notSeen.label)
        assertEquals(0f, notSeen.confidence)
    }

    @Test fun `place, evidence and runner up are stored and round trip to a verdict`() {
        val d = record(verdict(VerdictKind.DISEASE, 2, partial = true), row = 3, plant = 7)
        assertEquals("3-7", d.plantKey)
        assertEquals("Late_blight", d.label)
        assertEquals("Healthy", d.runnerUpLabel)
        assertEquals(true, d.partialFinding)
        val (v, l) = d.toVerdict()
        assertEquals(VerdictKind.DISEASE, v.kind)
        assertEquals("Late_blight", l[v.labelIndex!!])
        assertTrue(v.partial)
        assertEquals(VerdictKind.UNKNOWN_CONDITION, record(verdict(VerdictKind.UNKNOWN_CONDITION, 3)).toVerdict().first.kind)
        assertNull(record(verdict(VerdictKind.NEED_MORE_VIEWS, null, leaves = 0)).toVerdict().first.labelIndex)
    }

    @Test fun `summary counts a re-checked plant once using its latest decision`() {
        val decisions = listOf(
            record(verdict(VerdictKind.DISEASE, 2), seq = 1, plant = 1),
            record(verdict(VerdictKind.HEALTHY, 1), seq = 2, plant = 2),
            record(verdict(VerdictKind.HEALTHY, 1), seq = 3, plant = 1),   // plant 1 re-checked: now healthy
        )
        val s = RunSummaryReducer.reduce(run, decisions)
        assertEquals(2, s.decisions)
        assertEquals(2, s.ok)
        assertEquals(0, s.sick)
    }
}
