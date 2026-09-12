package com.sakshyam.agribot.ml

import com.sakshyam.agribot.ml.preprocessing.PlantInputEvidence
import com.sakshyam.agribot.ml.preprocessing.RgbImage
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class PlantInputEvidenceTest {
    @Test fun rejectsAchromaticLabBackground() {
        assertFalse(PlantInputEvidence.hasUsableColor(RgbImage(32, 32, IntArray(1024) { i ->
            val v = i % 256
            (v shl 16) or (v shl 8) or v
        })))
    }
    @Test fun allowsGreenAndBrownLeafEvidence() {
        for (color in listOf(0x398B35, 0x986B39)) {
            assertTrue(PlantInputEvidence.hasUsableColor(RgbImage(32, 32, IntArray(1024) { color })))
        }
    }
    @Test fun tinyColoredNoiseDoesNotValidateBackground() {
        assertFalse(PlantInputEvidence.hasUsableColor(RgbImage(32, 32, IntArray(1024) { if (it == 0) 0x398B35 else 0x888888 })))
    }
}
