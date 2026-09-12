package com.sakshyam.agribot.domain

import com.sakshyam.agribot.domain.logic.ClassifierDecisionPolicy
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ClassifierDecisionPolicyTest {
    @Test
    fun healthyUsesBaseThresholdAndDiseaseUsesHighThreshold() {
        assertEquals(
            0.62f,
            ClassifierDecisionPolicy.acceptanceThreshold("Healthy", 0.62f, 0.90f),
        )
        assertEquals(
            0.90f,
            ClassifierDecisionPolicy.acceptanceThreshold("Late_blight", 0.62f, 0.90f),
        )
    }

    @Test
    fun userRaisedBaseThresholdStillCannotLowerDiseaseBar() {
        assertEquals(
            0.80f,
            ClassifierDecisionPolicy.acceptanceThreshold("Healthy", 0.80f, 0.72f),
        )
        assertEquals(
            0.80f,
            ClassifierDecisionPolicy.acceptanceThreshold("Early_blight", 0.80f, 0.72f),
        )
    }

    @Test
    fun diseaseIsNotClearlyDominantWhenHealthyIsClose() {
        val close = ClassifierDecisionPolicy.isDiseaseClearlyDominant(
            top1Label = "Late_blight",
            top1Confidence = 0.71f,
            healthyConfidence = 0.65f,
        )
        assertFalse(close, "A disease score within the Healthy challenge margin must not be treated as dominant")
    }

    @Test
    fun diseaseIsClearlyDominantWhenHealthyIsFarBehind() {
        val far = ClassifierDecisionPolicy.isDiseaseClearlyDominant(
            top1Label = "Late_blight",
            top1Confidence = 0.92f,
            healthyConfidence = 0.40f,
        )
        assertTrue(far)
    }

    @Test
    fun healthyIsAlwaysDominantOverItself() {
        val ok = ClassifierDecisionPolicy.isDiseaseClearlyDominant(
            top1Label = "Healthy",
            top1Confidence = 0.7f,
            healthyConfidence = 0.7f,
        )
        assertTrue(ok)
    }
}
