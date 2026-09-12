package com.sakshyam.agribot.domain

import com.sakshyam.agribot.domain.logic.VegetationCoverageAnalyzer
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class VegetationCoverageAnalyzerTest {
    @Test
    fun emptyFrameHasZeroCoverage() {
        val assessment = VegetationCoverageAnalyzer.evaluate(0, 0, IntArray(0))
        assertEquals(0f, assessment.coverage)
    }

    @Test
    fun indoorGreySceneHasVeryLowCoverage() {
        // 64x64 all mid-grey pixels
        val pixels = IntArray(64 * 64) { 0x808080 }
        val assessment = VegetationCoverageAnalyzer.evaluate(64, 64, pixels)
        assertTrue(assessment.coverage < 0.01f, "grey indoor scene should not be classified as vegetation: ${assessment.coverage}")
    }

    @Test
    fun leafyFrameHasMeaningfulCoverage() {
        val pixels = IntArray(64 * 64) { index ->
            // dark green-ish leaves with some variation
            val r = 40 + (index % 30)
            val g = 110 + (index % 25)
            val b = 50 + (index % 20)
            (r shl 16) or (g shl 8) or b
        }
        val assessment = VegetationCoverageAnalyzer.evaluate(64, 64, pixels)
        assertTrue(assessment.coverage > 0.5f, "leafy frame should have high vegetation coverage: ${assessment.coverage}")
    }

    @Test
    fun blueSkyHasLowCoverageEvenThoughItIsNotGrey() {
        // Mostly blue sky pixels (no leaves)
        val pixels = IntArray(64 * 64) { _ ->
            (60 shl 16) or (110 shl 8) or 200
        }
        val assessment = VegetationCoverageAnalyzer.evaluate(64, 64, pixels)
        assertTrue(assessment.coverage < 0.05f, "blue sky should not be classified as vegetation: ${assessment.coverage}")
    }
}