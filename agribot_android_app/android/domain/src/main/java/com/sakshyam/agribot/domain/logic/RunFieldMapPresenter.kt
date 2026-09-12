package com.sakshyam.agribot.domain.logic

import com.sakshyam.agribot.domain.model.DecisionStatus
import com.sakshyam.agribot.domain.model.FieldLayout
import com.sakshyam.agribot.domain.model.PlantHealthAction
import com.sakshyam.agribot.domain.model.RecordedDecision
import com.sakshyam.agribot.domain.model.RunFieldMap
import com.sakshyam.agribot.domain.model.RunFieldMapCell
import com.sakshyam.agribot.domain.model.RunFieldMapRow

object RunFieldMapPresenter {
    fun map(layout: FieldLayout?, decisions: List<RecordedDecision>): RunFieldMap {
        if (layout == null) {
            return RunFieldMap(
                fieldLabel = decisions.firstOrNull()?.fieldId ?: "Unknown field",
                rows = decisions
                    .filter { decision -> decision.rowId != null && decision.plantNumber != null }
                    .groupBy { decision -> decision.rowId.orEmpty() }
                    .toSortedMap()
                    .map { (rowId, rowDecisions) ->
                        RunFieldMapRow(
                            rowId = rowId,
                            rowLabel = "Row $rowId",
                            cells = rowDecisions
                                .groupBy { it.plantNumber ?: it.sequence }
                                .toSortedMap()
                                .map { (plantNumber, plantDecisions) -> cell(plantNumber, plantDecisions.latestDecision()) },
                        )
                    },
            )
        }

        val latestByPosition = decisions
            .filter { decision -> decision.rowId != null && decision.plantNumber != null }
            .groupBy { decision -> PositionKey(decision.rowId.orEmpty(), decision.plantNumber ?: 0) }
            .mapValues { (_, plantDecisions) -> plantDecisions.latestDecision() }

        return RunFieldMap(
            fieldLabel = layout.name,
            rows = layout.rows.sortedBy { row -> row.rowIndex }.map { row ->
                RunFieldMapRow(
                    rowId = row.id.value,
                    rowLabel = "Row ${row.id.value}",
                    cells = (1..row.plantsPerRow).map { column ->
                        val plantNumber = layout.startPlant + ((column - 1) * layout.plantStep)
                        cell(plantNumber, latestByPosition[PositionKey(row.id.value, plantNumber)])
                    },
                )
            },
        )
    }

    private fun cell(plantNumber: Int, decision: RecordedDecision?): RunFieldMapCell {
        if (decision == null) {
            return RunFieldMapCell(
                plantNumber = plantNumber,
                label = "Plant $plantNumber: Not scanned",
                status = "empty",
                sequence = null,
            )
        }
        return RunFieldMapCell(
            plantNumber = plantNumber,
            label = "Plant $plantNumber: ${decision.label}",
            status = status(decision),
            sequence = decision.sequence,
        )
    }

    private fun status(decision: RecordedDecision): String = when {
        decision.manualOverride || decision.status == DecisionStatus.MANUAL -> "manual"
        decision.status == DecisionStatus.SKIPPED -> "skipped"
        decision.status == DecisionStatus.UNCERTAIN -> "uncertain"
        decision.action == PlantHealthAction.INSPECT_OR_TREAT -> "sick"
        else -> "ok"
    }

    private fun List<RecordedDecision>.latestDecision(): RecordedDecision =
        maxBy { decision -> decision.sequence }

    private data class PositionKey(
        val rowId: String,
        val plantNumber: Int,
    )
}
