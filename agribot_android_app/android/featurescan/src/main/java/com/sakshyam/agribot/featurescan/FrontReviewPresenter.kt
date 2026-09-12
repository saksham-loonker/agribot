package com.sakshyam.agribot.featurescan

import com.sakshyam.agribot.domain.logic.ExportSerializer
import com.sakshyam.agribot.domain.model.DecisionStatus
import com.sakshyam.agribot.domain.model.FrontBurstReview
import com.sakshyam.agribot.domain.model.FrontOverviewDecisionDraft
import com.sakshyam.agribot.domain.model.PlantHealthAction
import com.sakshyam.agribot.domain.model.RowSide

object FrontReviewPresenter {
    fun toUiState(
        review: FrontBurstReview,
        detectorStatus: String,
        candidateCount: Int,
    ): FrontReviewUiState =
        FrontReviewUiState(
            capturedFrameCount = review.capturedFrameCount,
            candidateCount = candidateCount,
            decisionCount = review.decisions.size,
            reason = review.reason,
            detectorStatus = detectorStatus,
            reviewRequired = review.reviewRequired,
            decisions = review.decisions.mapIndexed { index, draft -> draft.toUi(index) },
        )

    fun correctDecision(
        review: FrontBurstReview,
        index: Int,
        label: String,
        status: DecisionStatus,
    ): FrontBurstReview =
        review.copy(
            decisions = review.decisions.mapIndexed { draftIndex, draft ->
                if (draftIndex == index) draft.corrected(label, status) else draft
            },
            reviewRequired = true,
            reason = "manual_front_review",
        )

    fun markAllUncertain(review: FrontBurstReview): FrontBurstReview =
        review.copy(
            decisions = review.decisions.map { draft -> draft.corrected("Uncertain", DecisionStatus.UNCERTAIN) },
            reviewRequired = true,
            reason = "manual_front_review_uncertain",
        )

    private fun FrontOverviewDecisionDraft.toUi(index: Int): FrontReviewDecisionUiState =
        FrontReviewDecisionUiState(
            index = index,
            positionLabel = positionLabel(),
            label = label,
            confidenceLabel = "${(confidence * 100).toInt()}%",
            statusLabel = ExportSerializer.statusValue(status),
            actionLabel = ExportSerializer.actionValue(action),
            reason = reason,
            rowSide = rowSide,
        )

    private fun FrontOverviewDecisionDraft.positionLabel(): String {
        val side = when (rowSide) {
            RowSide.LEFT -> "Left"
            RowSide.RIGHT -> "Right"
            RowSide.UNKNOWN -> "Unknown"
        }
        val row = rowId?.let { " row $it" } ?: " row unknown"
        val plant = plantNumber?.let { " plant $it" } ?: " plant unknown"
        return "$side$row$plant"
    }

    private fun FrontOverviewDecisionDraft.corrected(
        label: String,
        status: DecisionStatus,
    ): FrontOverviewDecisionDraft {
        val correctedAction = actionFor(label, status)
        val correctedReason = "manual_front_review_${label.lowercase().replace(' ', '_')}"
        return copy(
            label = label,
            status = status,
            action = correctedAction,
            reason = correctedReason,
        )
    }

    private fun actionFor(label: String, status: DecisionStatus): PlantHealthAction =
        when {
            status == DecisionStatus.UNCERTAIN -> PlantHealthAction.RESCAN
            label.equals("Healthy", ignoreCase = true) -> PlantHealthAction.NONE
            else -> PlantHealthAction.INSPECT_OR_TREAT
        }
}
