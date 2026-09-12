package com.sakshyam.agribot.domain.logic

import com.sakshyam.agribot.domain.model.BoundingBox
import com.sakshyam.agribot.domain.model.DecisionStatus
import com.sakshyam.agribot.domain.model.FrontBurstReview
import com.sakshyam.agribot.domain.model.FrontCaptureCalibration
import com.sakshyam.agribot.domain.model.FrontGeometryAssignment
import com.sakshyam.agribot.domain.model.FrontOverviewCandidate
import com.sakshyam.agribot.domain.model.FrontOverviewDecisionDraft
import com.sakshyam.agribot.domain.model.PlantHealthAction
import com.sakshyam.agribot.domain.model.RowSide
import com.sakshyam.agribot.domain.model.ScanConstants

object FrontBurstProcessor {
    private const val MIN_BURST_FRAMES = 5
    private const val MAX_BURST_FRAMES = 9
    private const val MIN_CLASSIFIER_CONFIDENCE = ScanConstants.DEFAULT_CONFIDENCE_THRESHOLD

    fun process(
        capturedFrameCount: Int,
        calibration: FrontCaptureCalibration,
        candidates: List<FrontOverviewCandidate>,
        classifierConfidenceThreshold: Float = MIN_CLASSIFIER_CONFIDENCE,
    ): FrontBurstReview {
        if (candidates.isEmpty()) {
            return FrontBurstReview(
                capturedFrameCount = capturedFrameCount,
                decisions = emptyList(),
                reviewRequired = true,
                reason = "no_detector_candidates",
            )
        }

        val decisions = mapAndMergeCandidates(
            calibration = calibration,
            candidates = candidates,
            classifierConfidenceThreshold = classifierConfidenceThreshold.coerceIn(0f, 1f),
        )

        val burstCountOutOfRange = capturedFrameCount !in MIN_BURST_FRAMES..MAX_BURST_FRAMES
        val reviewRequired = burstCountOutOfRange || decisions.any { it.status == DecisionStatus.UNCERTAIN }
        return FrontBurstReview(
            capturedFrameCount = capturedFrameCount,
            decisions = decisions,
            reviewRequired = reviewRequired,
            reason = when {
                burstCountOutOfRange -> "burst_count_out_of_range"
                reviewRequired -> "uncertain_front_burst"
                else -> "ready_for_review"
            },
        )
    }

    /**
     * Preserve detector evidence without inventing row/plant coordinates when
     * the camera has not been geometrically calibrated. Every candidate is
     * deliberately non-actionable and must be reviewed or rescanned.
     */
    fun processWithoutCalibration(
        capturedFrameCount: Int,
        candidates: List<FrontOverviewCandidate>,
    ): FrontBurstReview {
        if (candidates.isEmpty()) {
            return FrontBurstReview(
                capturedFrameCount = capturedFrameCount,
                decisions = emptyList(),
                reviewRequired = true,
                reason = "no_detector_candidates",
            )
        }
        val decisions = candidates.map { candidate ->
            FrontOverviewDecisionDraft(
                bboxPx = candidate.bboxPx,
                rowSide = RowSide.UNKNOWN,
                rowId = null,
                plantColumnEstimate = null,
                plantNumber = null,
                label = "Uncertain",
                rawLabel = candidate.rawLabel.ifBlank { "unknown" },
                confidence = candidate.confidence.takeIf { it.isFinite() }?.coerceIn(0f, 1f) ?: 0f,
                status = DecisionStatus.UNCERTAIN,
                action = PlantHealthAction.RESCAN,
                rowAssignmentConfidence = 0f,
                plantPositionConfidence = 0f,
                geometryReason = "front_calibration_missing",
                reason = "front_calibration_missing",
            )
        }
        return FrontBurstReview(
            capturedFrameCount = capturedFrameCount,
            decisions = decisions,
            reviewRequired = true,
            reason = "front_calibration_missing",
        )
    }

    private fun actionFor(label: String, status: DecisionStatus): PlantHealthAction =
        when {
            status == DecisionStatus.UNCERTAIN -> PlantHealthAction.RESCAN
            label.equals("Healthy", ignoreCase = true) -> PlantHealthAction.NONE
            else -> PlantHealthAction.INSPECT_OR_TREAT
        }

    private fun mapAndMergeCandidates(
        calibration: FrontCaptureCalibration,
        candidates: List<FrontOverviewCandidate>,
        classifierConfidenceThreshold: Float,
    ): List<FrontOverviewDecisionDraft> {
        val mapper = FrontGeometryMapper(calibration)
        val mapped = candidates.mapIndexed { index, candidate ->
            MappedFrontCandidate(
                sourceIndex = index,
                candidate = candidate,
                assignment = mapper.assign(candidate.bboxPx, candidate.detectorConfidence),
            )
        }
        val grouped = mapped.groupBy { it.groupKey() }
        return grouped.values
            .sortedBy { group -> group.minOf { it.sourceIndex } }
            .map { group -> mergeGroup(group, classifierConfidenceThreshold) }
    }

    private fun MappedFrontCandidate.groupKey(): String =
        if (assignment.rowSide == RowSide.UNKNOWN || assignment.rowId == null || assignment.plantNumber == null) {
            "candidate:$sourceIndex"
        } else {
            "${assignment.rowSide}:${assignment.rowId}:${assignment.plantNumber}"
        }

    private fun mergeGroup(
        group: List<MappedFrontCandidate>,
        classifierConfidenceThreshold: Float,
    ): FrontOverviewDecisionDraft {
        val first = group.first()
        val assignment = first.assignment
        val uncertainGeometry = assignment.rowSide == RowSide.UNKNOWN
        val lowClassifier = group.any { it.candidate.confidence < classifierConfidenceThreshold }
        val confidentLabels = group
            .filter { it.candidate.confidence >= classifierConfidenceThreshold }
            .filter { it.candidate.label.isActionableLabel() }
            .map { it.candidate.label }
            .distinct()
        val hasUncertainLabel = group.any { candidate ->
            candidate.candidate.label.isUncertainLabel() &&
                candidate.candidate.confidence >= classifierConfidenceThreshold
        }
        val conflictingLabels = confidentLabels.size > 1
        val status = if (uncertainGeometry || lowClassifier || conflictingLabels || hasUncertainLabel) {
            DecisionStatus.UNCERTAIN
        } else {
            DecisionStatus.OK
        }
        val label = if (status == DecisionStatus.UNCERTAIN) "Uncertain" else confidentLabels.single()
        val reason = when {
            uncertainGeometry -> assignment.geometryReason
            hasUncertainLabel -> "classifier_abstention"
            conflictingLabels -> "conflicting_front_burst_labels"
            lowClassifier -> "low_classifier_confidence"
            group.size > 1 -> "front_burst_grouped_candidate"
            else -> "front_burst_candidate"
        }
        return FrontOverviewDecisionDraft(
            bboxPx = group.mergeBoundingBox(),
            rowSide = assignment.rowSide,
            rowId = assignment.rowId,
            plantColumnEstimate = assignment.plantColumnEstimate,
            plantNumber = assignment.plantNumber,
            label = label,
            rawLabel = group.joinRawLabels(),
            confidence = group.map { it.candidate.confidence }.average().toFloat(),
            status = status,
            action = actionFor(label, status),
            rowAssignmentConfidence = group.minOf { it.assignment.rowAssignmentConfidence },
            plantPositionConfidence = group.minOf { it.assignment.plantPositionConfidence },
            geometryReason = assignment.geometryReason,
            reason = reason,
        )
    }

    private fun List<MappedFrontCandidate>.mergeBoundingBox(): BoundingBox =
        BoundingBox(
            left = minOf { it.candidate.bboxPx.left },
            top = minOf { it.candidate.bboxPx.top },
            right = maxOf { it.candidate.bboxPx.right },
            bottom = maxOf { it.candidate.bboxPx.bottom },
        )

    private fun List<MappedFrontCandidate>.joinRawLabels(): String =
        map { it.candidate.rawLabel }
            .distinct()
            .joinToString(separator = "|")
}

private fun String.isUncertainLabel(): Boolean =
    equals("uncertain", ignoreCase = true) || equals("unknown", ignoreCase = true)

private fun String.isActionableLabel(): Boolean =
    isNotBlank() && !isUncertainLabel()

private data class MappedFrontCandidate(
    val sourceIndex: Int,
    val candidate: FrontOverviewCandidate,
    val assignment: FrontGeometryAssignment,
)
