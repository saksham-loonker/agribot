package com.sakshyam.agribot.domain

import com.sakshyam.agribot.domain.logic.FrontBurstProcessor
import com.sakshyam.agribot.domain.model.BoundingBox
import com.sakshyam.agribot.domain.model.DecisionStatus
import com.sakshyam.agribot.domain.model.FrontCaptureCalibration
import com.sakshyam.agribot.domain.model.FrontOverviewCandidate
import com.sakshyam.agribot.domain.model.PlantHealthAction
import com.sakshyam.agribot.domain.model.RowSide
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class FrontBurstProcessorTest {
    @Test
    fun noDetectorCandidatesRequiresReviewWithoutGuessingPlants() {
        val review = FrontBurstProcessor.process(
            capturedFrameCount = 7,
            calibration = calibration(),
            candidates = emptyList(),
        )

        assertTrue(review.reviewRequired)
        assertEquals("no_detector_candidates", review.reason)
        assertEquals(emptyList(), review.decisions)
    }

    @Test
    fun clearLeftAndRightCandidatesAreMappedForReview() {
        val review = FrontBurstProcessor.process(
            capturedFrameCount = 7,
            calibration = calibration(),
            candidates = listOf(
                FrontOverviewCandidate(
                    bboxPx = BoundingBox(96f, 260f, 150f, 360f),
                    detectorConfidence = 0.86f,
                    label = "Healthy",
                    confidence = 0.91f,
                ),
                FrontOverviewCandidate(
                    bboxPx = BoundingBox(500f, 250f, 560f, 360f),
                    detectorConfidence = 0.84f,
                    label = "Late_blight",
                    confidence = 0.88f,
                ),
            ),
        )

        assertEquals(false, review.reviewRequired)
        assertEquals("ready_for_review", review.reason)
        assertEquals(RowSide.LEFT, review.decisions[0].rowSide)
        assertEquals("A", review.decisions[0].rowId)
        assertEquals(PlantHealthAction.NONE, review.decisions[0].action)
        assertEquals(RowSide.RIGHT, review.decisions[1].rowSide)
        assertEquals("B", review.decisions[1].rowId)
        assertEquals(PlantHealthAction.INSPECT_OR_TREAT, review.decisions[1].action)
    }

    @Test
    fun ambiguousGeometryProducesUncertainDraftInsteadOfRowGuess() {
        val review = FrontBurstProcessor.process(
            capturedFrameCount = 7,
            calibration = calibration(calibrationQuality = 0.55f),
            candidates = listOf(
                FrontOverviewCandidate(
                    bboxPx = BoundingBox(260f, 240f, 360f, 360f),
                    detectorConfidence = 0.90f,
                    label = "Healthy",
                    confidence = 0.93f,
                ),
            ),
        )

        val decision = review.decisions.single()
        assertTrue(review.reviewRequired)
        assertEquals(RowSide.UNKNOWN, decision.rowSide)
        assertEquals(null, decision.rowId)
        assertEquals(null, decision.plantNumber)
        assertEquals("Uncertain", decision.label)
        assertEquals(DecisionStatus.UNCERTAIN, decision.status)
        assertEquals("ambiguous_geometry", decision.reason)
    }

    @Test
    fun burstCountOutsidePlanRangeRequiresReview() {
        val review = FrontBurstProcessor.process(
            capturedFrameCount = 3,
            calibration = calibration(),
            candidates = listOf(
                FrontOverviewCandidate(
                    bboxPx = BoundingBox(96f, 260f, 150f, 360f),
                    detectorConfidence = 0.86f,
                    label = "Healthy",
                    confidence = 0.91f,
                ),
            ),
        )

        assertTrue(review.reviewRequired)
        assertEquals("burst_count_out_of_range", review.reason)
    }

    @Test
    fun usesTheConfiguredClassifierThresholdForFrontDecisions() {
        val review = FrontBurstProcessor.process(
            capturedFrameCount = 7,
            calibration = calibration(),
            classifierConfidenceThreshold = 0.65f,
            candidates = listOf(
                FrontOverviewCandidate(
                    bboxPx = BoundingBox(96f, 260f, 150f, 360f),
                    detectorConfidence = 0.86f,
                    label = "Healthy",
                    confidence = 0.70f,
                ),
            ),
        )

        assertEquals(false, review.reviewRequired)
        assertEquals("Healthy", review.decisions.single().label)
        assertEquals(DecisionStatus.OK, review.decisions.single().status)
    }

    @Test
    fun duplicateCandidatesForSamePlantMergeIntoOneDecision() {
        val review = FrontBurstProcessor.process(
            capturedFrameCount = 7,
            calibration = calibration(),
            candidates = listOf(
                FrontOverviewCandidate(
                    bboxPx = BoundingBox(92f, 260f, 150f, 360f),
                    detectorConfidence = 0.88f,
                    label = "Healthy",
                    confidence = 0.91f,
                ),
                FrontOverviewCandidate(
                    bboxPx = BoundingBox(100f, 252f, 158f, 366f),
                    detectorConfidence = 0.86f,
                    label = "Healthy",
                    confidence = 0.87f,
                ),
            ),
        )

        assertEquals(false, review.reviewRequired)
        assertEquals(1, review.decisions.size)
        val decision = review.decisions.single()
        assertEquals(RowSide.LEFT, decision.rowSide)
        assertEquals("A", decision.rowId)
        assertEquals(11, decision.plantNumber)
        assertEquals("Healthy", decision.label)
        assertEquals("front_burst_grouped_candidate", decision.reason)
        assertEquals(92f, decision.bboxPx.left)
        assertEquals(252f, decision.bboxPx.top)
        assertEquals(158f, decision.bboxPx.right)
        assertEquals(366f, decision.bboxPx.bottom)
    }

    @Test
    fun conflictingHighConfidenceLabelsForSamePlantBecomeUncertain() {
        val review = FrontBurstProcessor.process(
            capturedFrameCount = 7,
            calibration = calibration(),
            candidates = listOf(
                FrontOverviewCandidate(
                    bboxPx = BoundingBox(92f, 260f, 150f, 360f),
                    detectorConfidence = 0.88f,
                    label = "Healthy",
                    confidence = 0.91f,
                ),
                FrontOverviewCandidate(
                    bboxPx = BoundingBox(100f, 252f, 158f, 366f),
                    detectorConfidence = 0.86f,
                    label = "Late_blight",
                    confidence = 0.87f,
                ),
            ),
        )

        assertTrue(review.reviewRequired)
        assertEquals("uncertain_front_burst", review.reason)
        assertEquals(1, review.decisions.size)
        val decision = review.decisions.single()
        assertEquals(RowSide.LEFT, decision.rowSide)
        assertEquals("A", decision.rowId)
        assertEquals(11, decision.plantNumber)
        assertEquals("Uncertain", decision.label)
        assertEquals(DecisionStatus.UNCERTAIN, decision.status)
        assertEquals("conflicting_front_burst_labels", decision.reason)
    }

    @Test
    fun highConfidenceAbstentionLabelStillRequiresReview() {
        val review = FrontBurstProcessor.process(
            capturedFrameCount = 7,
            calibration = calibration(),
            candidates = listOf(
                FrontOverviewCandidate(
                    bboxPx = BoundingBox(96f, 260f, 150f, 360f),
                    detectorConfidence = 0.86f,
                    label = "Uncertain",
                    confidence = 0.87f,
                ),
            ),
        )

        val decision = review.decisions.single()
        assertTrue(review.reviewRequired)
        assertEquals("Uncertain", decision.label)
        assertEquals(DecisionStatus.UNCERTAIN, decision.status)
        assertEquals("classifier_abstention", decision.reason)
    }

    private fun calibration(calibrationQuality: Float = 0.9f) = FrontCaptureCalibration(
        cameraHeightM = 1.1,
        cameraDistanceToNearestRowM = 1.0,
        cameraTiltDegrees = 15.0,
        rowSpacingM = 1.4,
        plantSpacingM = 0.45,
        leftRowId = "A",
        rightRowId = "B",
        nearestLeftPlantNumber = 10,
        nearestRightPlantNumber = 20,
        guideRailLeftPx = 120f,
        guideRailRightPx = 520f,
        frameWidthPx = 640,
        frameHeightPx = 480,
        calibrationQuality = calibrationQuality,
    )
}
