package com.sakshyam.agribot.ml

import com.sakshyam.agribot.ml.inference.ClassifierDecision
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ClassifierDecisionTest {
    @Test
    fun confidenceAboveThresholdIsNeverReportedAsUncertain() {
        val prediction = ClassifierDecision.toPrediction(
            rawLabel = "Late_blight",
            confidence = 0.95f,
            threshold = 0.62f,
            latencyMs = 12.0,
            modelVersion = "test",
            healthyConfidence = 0.05f,
        )

        assertEquals("Late_blight", prediction.label)
        assertFalse(prediction.isUncertain)
        assertTrue(prediction.isDisease)
        assertEquals(0.95f, prediction.confidence)
    }

    @Test
    fun confidenceBelowThresholdRetainsRawLabelForDiagnostics() {
        val prediction = ClassifierDecision.toPrediction(
            rawLabel = "Late_blight",
            confidence = 0.54f,
            threshold = 0.62f,
            latencyMs = 12.0,
            modelVersion = "test",
        )

        assertEquals("Uncertain", prediction.label)
        assertEquals("Late_blight", prediction.rawLabel)
        assertTrue(prediction.isUncertain)
        assertFalse(prediction.isDisease)
    }

    @Test
    fun healthyUsesTheLowerBaseGateWhileDiseaseKeepsTheHigherBar() {
        val healthy = ClassifierDecision.toPrediction(
            rawLabel = "Healthy",
            confidence = 0.65f,
            threshold = 0.62f,
            highConfidenceThreshold = 0.90f,
            latencyMs = 12.0,
            modelVersion = "test",
        )
        val disease = ClassifierDecision.toPrediction(
            rawLabel = "Late_blight",
            confidence = 0.70f,
            threshold = 0.62f,
            highConfidenceThreshold = 0.90f,
            latencyMs = 12.0,
            modelVersion = "test",
        )

        assertEquals("Healthy", healthy.label)
        assertFalse(healthy.isUncertain)
        assertEquals("Uncertain", disease.label)
        assertEquals("Late_blight", disease.rawLabel)
        assertTrue(disease.isUncertain)
    }

    @Test
    fun unknownLabelNeverBecomesAnAutomaticDisease() {
        val prediction = ClassifierDecision.toPrediction(
            rawLabel = "Unknown",
            confidence = 0.99f,
            threshold = 0.765f,
            latencyMs = 12.0,
            modelVersion = "test",
        )

        assertEquals("Uncertain", prediction.label)
        assertTrue(prediction.isUncertain)
        assertFalse(prediction.isDisease)
        assertEquals("Unknown", prediction.rawLabel)
    }

    @Test
    fun diseaseWithHealthyRunnerUpIsForcedToUncertain() {
        // Even with a high confidence, if Healthy is close, we cannot be sure.
        val prediction = ClassifierDecision.toPrediction(
            rawLabel = "Late_blight",
            confidence = 0.91f,
            threshold = 0.62f,
            highConfidenceThreshold = 0.90f,
            latencyMs = 12.0,
            modelVersion = "test",
            healthyConfidence = 0.84f,
        )

        assertEquals("Uncertain", prediction.label)
        assertTrue(prediction.isUncertain)
        assertFalse(prediction.isDisease)
    }

    @Test
    fun diseaseWithHealthyFarBehindIsAccepted() {
        val prediction = ClassifierDecision.toPrediction(
            rawLabel = "Late_blight",
            confidence = 0.95f,
            threshold = 0.62f,
            highConfidenceThreshold = 0.90f,
            latencyMs = 12.0,
            modelVersion = "test",
            healthyConfidence = 0.20f,
        )

        assertEquals("Late_blight", prediction.label)
        assertFalse(prediction.isUncertain)
        assertTrue(prediction.isDisease)
    }
}
