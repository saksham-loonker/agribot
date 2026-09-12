package com.sakshyam.agribot.domain.logic

import com.sakshyam.agribot.domain.model.DecisionStatus
import com.sakshyam.agribot.domain.model.PlantHealthAction
import com.sakshyam.agribot.domain.model.RecordedDecision
import com.sakshyam.agribot.domain.model.RunId
import com.sakshyam.agribot.domain.model.RunSummary

object RunSummaryReducer {
    fun reduce(runId: RunId, decisions: List<RecordedDecision>): RunSummary {
        val labels = decisions.groupingBy { it.label }.eachCount().toSortedMap()
        val statuses = decisions.groupingBy { it.status }.eachCount()
        val uncertain = statuses[DecisionStatus.UNCERTAIN] ?: 0
        val ok = decisions.count { it.isReportable() && it.isHealthyLabel() }
        val totalConfidence = decisions.sumOf { it.confidence.toDouble() }
        return RunSummary(
            runId = runId,
            decisions = decisions.size,
            ok = ok,
            sick = decisions.count { it.isReportable() && it.isDiseaseDecision() },
            uncertain = uncertain,
            uncertainRate = if (decisions.isEmpty()) 0.0 else uncertain.toDouble() / decisions.size,
            avgConfidence = if (decisions.isEmpty()) 0.0 else totalConfidence / decisions.size,
            maxConfidence = decisions.maxOfOrNull { it.confidence } ?: 0f,
            totalLateFrames = decisions.sumOf { it.lateFrames },
            labels = labels,
            statuses = statuses,
            latestDecision = decisions.maxByOrNull { it.sequence },
        )
    }

    private fun RecordedDecision.isReportable(): Boolean =
        status != DecisionStatus.UNCERTAIN &&
            status != DecisionStatus.SKIPPED &&
            action != PlantHealthAction.RESCAN

    private fun RecordedDecision.isHealthyLabel(): Boolean =
        label.trim().lowercase().replace('_', ' ').replace('-', ' ') == "healthy"

    private fun RecordedDecision.isDiseaseDecision(): Boolean {
        val normalized = label.trim().lowercase().replace('_', ' ').replace('-', ' ')
        if (normalized == "healthy" || normalized == "uncertain" || normalized == "unknown") return false
        return action == PlantHealthAction.INSPECT_OR_TREAT || TreatmentGuide.recommendationFor(label) != null
    }
}
