package com.sakshyam.agribot.domain.logic

import com.sakshyam.agribot.domain.model.DecisionFilter
import com.sakshyam.agribot.domain.model.DecisionStatus
import com.sakshyam.agribot.domain.model.PlantHealthAction
import com.sakshyam.agribot.domain.model.RecordedDecision
import com.sakshyam.agribot.domain.model.RunDecisionRow

object RunDetailPresenter {
    fun rows(decisions: List<RecordedDecision>, filter: DecisionFilter): List<RunDecisionRow> =
        decisions
            .filter { decision -> filter.matches(decision) }
            .sortedBy { decision -> decision.sequence }
            .map { decision ->
                RunDecisionRow(
                    decisionId = decision.id,
                    sequence = decision.sequence,
                    plantLabel = decision.plantDisplay ?: "Plant ${decision.plantNumber ?: decision.sequence}",
                    label = decision.label,
                    confidenceLabel = "${(decision.confidence * 100).toInt()}%",
                    statusLabel = ExportSerializer.statusValue(decision.status),
                    actionLabel = ExportSerializer.actionValue(decision.action),
                    reason = decision.reason,
                )
            }

    private fun DecisionFilter.matches(decision: RecordedDecision): Boolean = when (this) {
        DecisionFilter.ALL -> true
        DecisionFilter.SICK -> decision.action == PlantHealthAction.INSPECT_OR_TREAT
        DecisionFilter.UNCERTAIN -> decision.status == DecisionStatus.UNCERTAIN
        DecisionFilter.MANUAL -> decision.manualOverride || decision.status == DecisionStatus.MANUAL
        DecisionFilter.SKIPPED -> decision.status == DecisionStatus.SKIPPED
    }
}
