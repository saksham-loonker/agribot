package com.sakshyam.agribot.featurescan

import com.sakshyam.agribot.domain.model.FrontOverviewCandidate
import kotlin.math.hypot

/**
 * Turns per-frame detector output into a bounded temporal consensus. A box
 * observed in only one frame is retained as explicit uncertainty for review;
 * it is never allowed to become an automatic plant decision.
 */
object FrontBurstConsensus {
    const val MIN_DISTINCT_FRAMES = 2
    const val MAX_CLUSTERS = 32

    data class FrameCandidates(
        val frameIndex: Int,
        val candidates: List<FrontOverviewCandidate>,
    )

    fun requireDistinctFrameConsensus(
        frames: List<FrameCandidates>,
        minDistinctFrames: Int = MIN_DISTINCT_FRAMES,
    ): List<FrontOverviewCandidate> {
        if (frames.isEmpty()) return emptyList()
        val clusters = mutableListOf<MutableList<Pair<Int, FrontOverviewCandidate>>>()
        frames.take(FrontBurstCapturePolicy.TARGET_FRAME_COUNT).forEach { frame ->
            frame.candidates
                .sortedByDescending { it.detectorConfidence }
                .take(FrontBurstCapturePolicy.MAX_DETECTOR_CANDIDATES_PER_FRAME)
                .forEach { candidate ->
                    val cluster = clusters.firstOrNull { existing ->
                        existing.any { (_, previous) -> matches(previous, candidate) }
                    }
                    if (cluster == null && clusters.size < MAX_CLUSTERS) {
                        clusters += mutableListOf(frame.frameIndex to candidate)
                    } else if (cluster != null) {
                        // At most one candidate per frame per cluster prevents
                        // duplicate decoder boxes from faking temporal support.
                        val existingIndex = cluster.indexOfFirst { (index, _) -> index == frame.frameIndex }
                        if (existingIndex < 0) {
                            cluster += frame.frameIndex to candidate
                        } else if (candidate.detectorConfidence > cluster[existingIndex].second.detectorConfidence) {
                            cluster[existingIndex] = frame.frameIndex to candidate
                        }
                    }
                }
        }

        return clusters
            .sortedByDescending { cluster -> cluster.maxOf { it.second.detectorConfidence } }
            .flatMap { cluster ->
                val distinctFrames = cluster.map { it.first }.distinct().size
                if (distinctFrames >= minDistinctFrames.coerceAtLeast(1)) {
                    cluster.sortedBy { it.first }.map { it.second }
                } else {
                    listOf(
                        cluster.maxBy { it.second.detectorConfidence }.second.copy(
                            label = "Uncertain",
                            confidence = 0f,
                            rawLabel = "insufficient_frame_consensus",
                        ),
                    )
                }
            }
    }

    private fun matches(first: FrontOverviewCandidate, second: FrontOverviewCandidate): Boolean {
        val left = maxOf(first.bboxPx.left, second.bboxPx.left)
        val top = maxOf(first.bboxPx.top, second.bboxPx.top)
        val right = minOf(first.bboxPx.right, second.bboxPx.right)
        val bottom = minOf(first.bboxPx.bottom, second.bboxPx.bottom)
        val intersection = (right - left).coerceAtLeast(0f) * (bottom - top).coerceAtLeast(0f)
        val firstArea = first.bboxPx.width.coerceAtLeast(0f) * first.bboxPx.height.coerceAtLeast(0f)
        val secondArea = second.bboxPx.width.coerceAtLeast(0f) * second.bboxPx.height.coerceAtLeast(0f)
        val union = firstArea + secondArea - intersection
        val iou = if (union <= 0f) 0f else intersection / union
        if (iou >= 0.20f) return true
        val distance = hypot(
            first.bboxPx.centerX - second.bboxPx.centerX,
            first.bboxPx.centerY - second.bboxPx.centerY,
        )
        return distance <= maxOf(first.bboxPx.width, first.bboxPx.height, second.bboxPx.width, second.bboxPx.height)
    }
}
