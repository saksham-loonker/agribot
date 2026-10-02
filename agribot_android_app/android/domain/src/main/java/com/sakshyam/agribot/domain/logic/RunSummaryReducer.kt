package com.sakshyam.agribot.domain.logic

import com.sakshyam.agribot.domain.model.DecisionStatus
import com.sakshyam.agribot.domain.model.PlantHealthAction
import com.sakshyam.agribot.domain.model.RecordedDecision
import com.sakshyam.agribot.domain.model.RunId
import com.sakshyam.agribot.domain.model.RunSummary

object RunSummaryReducer {
    fun reduce(runId: RunId, all: List<RecordedDecision>): RunSummary {
        // A plant re-checked during the same run counts once: keep the latest decision per plant.
        val decisions = all.groupBy { it.plantKey ?: "seq-${it.sequence}" }.values.map { d -> d.maxBy { it.sequence } }
        val labels = decisions.groupingBy { it.label }.eachCount().toSortedMap()
        val statuses = decisions.groupingBy { it.status }.eachCount()
        val uncertain = statuses[DecisionStatus.UNCERTAIN] ?: 0
        val reportable = decisions.filter { it.isReportable() }
        val ok = reportable.count { it.isHealthyLabel() && it.action == PlantHealthAction.NONE }
        return RunSummary(
            runId = runId,
            decisions = decisions.size,
            ok = ok,
            sick = reportable.count { it.needsAttention() },
            uncertain = uncertain,
            uncertainRate = if (decisions.isEmpty()) 0.0 else uncertain.toDouble() / decisions.size,
            // Mean over actual verdicts only: unsure/skipped plants carry no meaningful confidence.
            avgConfidence = if (reportable.isEmpty()) 0.0 else reportable.sumOf { it.confidence.toDouble() } / reportable.size,
            maxConfidence = reportable.maxOfOrNull { it.confidence } ?: 0f,
            totalLateFrames = decisions.sumOf { it.lateFrames },
            labels = labels,
            statuses = statuses,
            latestDecision = all.maxByOrNull { it.sequence },
        )
    }

    private fun RecordedDecision.isReportable(): Boolean =
        status != DecisionStatus.UNCERTAIN &&
            status != DecisionStatus.SKIPPED &&
            action != PlantHealthAction.RESCAN

    /** Inspect-recommended, or any non-healthy diagnosis even if an older import stored action NONE. */
    private fun RecordedDecision.needsAttention(): Boolean {
        if (action == PlantHealthAction.INSPECT_OR_TREAT) return true
        val n = label.trim().lowercase().replace('_', ' ').replace('-', ' ')
        return n.isNotEmpty() && n !in setOf("healthy", "uncertain", "unsure", "unknown")
    }

    private fun RecordedDecision.isHealthyLabel(): Boolean =
        label.trim().lowercase().replace('_', ' ').replace('-', ' ') == "healthy"
}
