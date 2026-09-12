package com.sakshyam.agribot.ml

import com.sakshyam.agribot.ml.inference.ClassifierProbabilityCalibrator
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ClassifierProbabilityCalibratorTest {
    @Test
    fun temperatureScalingReducesAnOverconfidentTopScoreWithoutChangingTheWinner() {
        val calibrated = ClassifierProbabilityCalibrator.temperatureScale(
            floatArrayOf(0.01f, 0.97f, 0.01f, 0.01f),
            temperature = 3.0f,
        )

        assertEquals(1, calibrated.indices.maxBy { calibrated[it] })
        assertTrue(calibrated[1] < 0.70f)
        assertEquals(1.0f, calibrated.sum(), absoluteTolerance = 0.0001f)
    }

    @Test
    fun invalidTemperatureFallsBackToTheConservativeDefault() {
        val calibrated = ClassifierProbabilityCalibrator.temperatureScale(
            floatArrayOf(0.1f, 0.8f, 0.1f),
            temperature = Float.NaN,
        )

        assertEquals(1, calibrated.indices.maxBy { calibrated[it] })
        assertTrue(calibrated[1] < 0.90f)
    }
}
