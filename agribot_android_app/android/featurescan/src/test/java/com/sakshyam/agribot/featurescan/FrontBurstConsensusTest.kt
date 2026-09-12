package com.sakshyam.agribot.featurescan

import com.sakshyam.agribot.domain.model.BoundingBox
import com.sakshyam.agribot.domain.model.FrontOverviewCandidate
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class FrontBurstConsensusTest {
    @Test
    fun singleFrameCandidateIsExplicitlyUncertain() {
        val result = FrontBurstConsensus.requireDistinctFrameConsensus(
            listOf(FrontBurstConsensus.FrameCandidates(0, listOf(candidate(100f, 0.9f)))),
        )

        assertEquals(1, result.size)
        assertEquals("insufficient_frame_consensus", result.single().rawLabel)
        assertEquals(0f, result.single().confidence)
    }

    @Test
    fun samePlantAcrossTwoFramesIsRetainedForClassification() {
        val result = FrontBurstConsensus.requireDistinctFrameConsensus(
            listOf(
                FrontBurstConsensus.FrameCandidates(0, listOf(candidate(100f, 0.80f))),
                FrontBurstConsensus.FrameCandidates(1, listOf(candidate(103f, 0.90f))),
            ),
        )

        assertEquals(2, result.size)
        assertTrue(result.all { it.rawLabel == "crop" })
    }

    private fun candidate(left: Float, confidence: Float) = FrontOverviewCandidate(
        bboxPx = BoundingBox(left, 100f, left + 60f, 180f),
        detectorConfidence = confidence,
        label = "Uncertain",
        confidence = 0f,
        rawLabel = "crop",
    )
}
