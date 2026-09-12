package com.sakshyam.agribot.domain

import com.sakshyam.agribot.domain.logic.NonPlantSceneGate
import com.sakshyam.agribot.domain.logic.VegetationCoverageAnalyzer
import com.sakshyam.agribot.domain.model.BoundingBox
import com.sakshyam.agribot.domain.model.FrontOverviewCandidate
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class NonPlantSceneGateTest {
    private fun candidate(confidence: Float): FrontOverviewCandidate =
        FrontOverviewCandidate(
            bboxPx = BoundingBox(10f, 10f, 80f, 80f),
            detectorConfidence = confidence,
            label = "crop",
            confidence = 0f,
            rawLabel = "crop",
        )

    @Test
    fun emptyDetectorOutputIsAlwaysReturnedAsEmpty() {
        val decision = NonPlantSceneGate.apply(
            candidates = emptyList(),
            vegetation = VegetationCoverageAnalyzer.Assessment(0f, 0f, 0f, 100),
        )
        assertEquals(0, decision.candidates.size)
        assertEquals("no_detector_output", decision.reason)
    }

    @Test
    fun indoorLabWithLowConfidenceBoxesIsSuppressed() {
        val candidates = listOf(candidate(0.40f), candidate(0.42f), candidate(0.55f))
        val decision = NonPlantSceneGate.apply(
            candidates = candidates,
            vegetation = VegetationCoverageAnalyzer.Assessment(0.001f, 0.02f, 0.001f, 1000),
            bareMinScore = 0.65f,
            bareMinCandidates = 2,
            minCoverage = 0.012f,
        )
        assertTrue(decision.dropped, "bare scene with no strong boxes must be suppressed")
        assertEquals(0, decision.candidates.size)
    }

    @Test
    fun realFieldWithGreenCoverageKeepsAllCandidates() {
        val candidates = listOf(candidate(0.42f), candidate(0.51f), candidate(0.66f))
        val decision = NonPlantSceneGate.apply(
            candidates = candidates,
            vegetation = VegetationCoverageAnalyzer.Assessment(0.34f, 0.45f, 0.34f, 2000),
        )
        assertEquals(3, decision.candidates.size)
        assertEquals("vegetation_present", decision.reason)
    }

    @Test
    fun bareSceneWithVeryStrongBoxesSurvives() {
        val candidates = listOf(candidate(0.95f), candidate(0.92f), candidate(0.40f))
        val decision = NonPlantSceneGate.apply(
            candidates = candidates,
            vegetation = VegetationCoverageAnalyzer.Assessment(0.002f, 0.01f, 0.002f, 1000),
            bareMinScore = 0.65f,
            bareMinCandidates = 2,
        )
        assertEquals(2, decision.candidates.size)
        assertTrue(decision.candidates.all { it.detectorConfidence >= 0.65f })
    }
}