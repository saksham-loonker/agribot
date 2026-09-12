package com.sakshyam.agribot.domain.logic

import com.sakshyam.agribot.domain.model.FrontOverviewCandidate
import com.sakshyam.agribot.domain.model.ScanConstants

/**
 * Decides whether a frame should be allowed to produce plant detections.
 *
 * The detector model alone is not enough. It is a YOLO classifier trained on
 * plant images, so it will happily suggest boxes on any sufficiently
 * textured scene (lab equipment, hands on a table, a laptop). A non-plant
 * scene gate rejects those frames before the detector output ever becomes
 * a tracked plant and definitely before it becomes a recorded disease
 * decision.
 *
 * The gate is intentionally simple and deterministic:
 *  1. If vegetation coverage (leaf-like green pixels) is below a small
 *     threshold, only return detections if the detector is extremely
 *     confident on at least [ScanConstants.MIN_PLANT_CANDIDATES_WHEN_BARE]
 *     boxes. Otherwise we drop everything and treat the frame as empty.
 *  2. Otherwise return the candidates unchanged.
 */
object NonPlantSceneGate {
    data class Decision(
        val candidates: List<FrontOverviewCandidate>,
        val dropped: Boolean,
        val reason: String,
    )

    fun apply(
        candidates: List<FrontOverviewCandidate>,
        vegetation: VegetationCoverageAnalyzer.Assessment,
        bareMinScore: Float = 0.65f,
        bareMinCandidates: Int = ScanConstants.MIN_PLANT_CANDIDATES_WHEN_BARE,
        minCoverage: Float = ScanConstants.MIN_VEGETATION_COVERAGE,
    ): Decision {
        if (candidates.isEmpty()) {
            return Decision(candidates = emptyList(), dropped = false, reason = "no_detector_output")
        }
        if (vegetation.coverage >= minCoverage) {
            return Decision(candidates = candidates, dropped = false, reason = "vegetation_present")
        }
        val strongCandidates = candidates.filter { it.detectorConfidence >= bareMinScore }
        if (strongCandidates.size >= bareMinCandidates) {
            return Decision(
                candidates = strongCandidates,
                dropped = false,
                reason = "bare_scene_strong_boxes",
            )
        }
        return Decision(
            candidates = emptyList(),
            dropped = true,
            reason = "bare_scene_no_strong_boxes",
        )
    }
}