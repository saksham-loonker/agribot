package com.sakshyam.agribot.featurescan

import com.sakshyam.agribot.domain.model.BoundingBox
import com.sakshyam.agribot.domain.model.DecisionStatus
import com.sakshyam.agribot.domain.model.FrontBurstReview
import com.sakshyam.agribot.domain.model.FrontOverviewDecisionDraft
import com.sakshyam.agribot.domain.model.PlantHealthAction
import com.sakshyam.agribot.domain.model.RowSide
import kotlin.test.Test
import kotlin.test.assertEquals

class FrontReviewPresenterTest {
    @Test
    fun exposesReviewableMappedDecisionRows() {
        val ui = FrontReviewPresenter.toUiState(
            review = review(),
            detectorStatus = "detector_candidates",
            candidateCount = 1,
        )

        assertEquals(1, ui.decisions.size)
        assertEquals("Left row A plant 12", ui.decisions.single().positionLabel)
        assertEquals("Late_blight", ui.decisions.single().label)
        assertEquals("88%", ui.decisions.single().confidenceLabel)
    }

    @Test
    fun correctsSingleFrontDecisionWithoutChangingOtherDrafts() {
        val corrected = FrontReviewPresenter.correctDecision(
            review = review(
                listOf(
                    draft(label = "Late_blight", rowId = "A", plantNumber = 12),
                    draft(label = "Healthy", rowId = "B", plantNumber = 20),
                ),
            ),
            index = 0,
            label = "Healthy",
            status = DecisionStatus.MANUAL,
        )

        assertEquals("Healthy", corrected.decisions[0].label)
        assertEquals(PlantHealthAction.NONE, corrected.decisions[0].action)
        assertEquals("manual_front_review_healthy", corrected.decisions[0].reason)
        assertEquals("Healthy", corrected.decisions[1].label)
        assertEquals("manual_front_review", corrected.reason)
    }

    @Test
    fun marksAllDraftsUncertainForAmbiguousReview() {
        val corrected = FrontReviewPresenter.markAllUncertain(review())

        assertEquals("Uncertain", corrected.decisions.single().label)
        assertEquals(DecisionStatus.UNCERTAIN, corrected.decisions.single().status)
        assertEquals(PlantHealthAction.RESCAN, corrected.decisions.single().action)
        assertEquals("manual_front_review_uncertain", corrected.reason)
    }

    private fun review(decisions: List<FrontOverviewDecisionDraft> = listOf(draft())) = FrontBurstReview(
        capturedFrameCount = 7,
        decisions = decisions,
        reviewRequired = false,
        reason = "ready_for_review",
    )

    private fun draft(
        label: String = "Late_blight",
        rowId: String? = "A",
        plantNumber: Int? = 12,
    ) = FrontOverviewDecisionDraft(
        bboxPx = BoundingBox(120f, 220f, 260f, 420f),
        rowSide = RowSide.LEFT,
        rowId = rowId,
        plantColumnEstimate = 3,
        plantNumber = plantNumber,
        label = label,
        rawLabel = label,
        confidence = 0.88f,
        status = DecisionStatus.OK,
        action = PlantHealthAction.INSPECT_OR_TREAT,
        rowAssignmentConfidence = 0.82f,
        plantPositionConfidence = 0.79f,
        geometryReason = "clear",
        reason = "front_burst_candidate",
    )
}
