package com.sakshyam.agribot.domain

import com.sakshyam.agribot.domain.logic.PlantDecisionGate
import com.sakshyam.agribot.domain.model.DecisionStatus
import com.sakshyam.agribot.domain.model.FramePrediction
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class PlantDecisionGateTest {
    @Test
    fun primaryHighConfidenceAgreementEmitsAfterTwoFrames() {
        val gate = PlantDecisionGate()

        assertNull(gate.add(prediction("Healthy", 0.95f)))
        val decision = gate.add(prediction("Healthy", 0.92f))

        assertEquals("Healthy", decision?.label)
        assertEquals(0.92f, decision?.confidence)
        assertEquals(DecisionStatus.OK, decision?.status)
        assertEquals(2, decision?.framesUsed)
        assertEquals("primary_agreement", decision?.reason)
    }

    @Test
    fun primaryDisagreementUsesBackupMajority() {
        val gate = PlantDecisionGate(highConfidence = 0.765f)

        assertNull(gate.add(prediction("Healthy", 0.91f)))
        assertNull(gate.add(prediction("Late_blight", 0.92f)))
        val decision = gate.add(prediction("Healthy", 0.89f))

        assertEquals("Healthy", decision?.label)
        assertEquals(DecisionStatus.OK, decision?.status)
        assertEquals(3, decision?.framesUsed)
        assertEquals("backup_majority", decision?.reason)
    }

    @Test
    fun lowConfidenceFramesEmitUncertainAfterBackup() {
        val gate = PlantDecisionGate(highConfidence = 0.765f)

        assertNull(gate.add(prediction("Late_blight", 0.74f)))
        assertNull(gate.add(prediction("Late_blight", 0.73f)))
        val decision = gate.add(prediction("Late_blight", 0.72f))

        assertEquals("Uncertain", decision?.label)
        assertEquals(DecisionStatus.UNCERTAIN, decision?.status)
        assertEquals(3, decision?.framesUsed)
        assertEquals("low_confidence", decision?.reason)
        assertEquals(0.72f, decision?.confidence)
    }

    @Test
    fun tiedLabelsEmitUncertain() {
        val gate = PlantDecisionGate(highConfidence = 0.765f, maxFrames = 4)

        assertNull(gate.add(prediction("Healthy", 0.91f)))
        assertNull(gate.add(prediction("Late_blight", 0.92f)))
        assertNull(gate.add(prediction("Leaf Miner", 0.93f)))
        val decision = gate.add(prediction("Magnesium Deficiency", 0.94f))

        assertEquals("Uncertain", decision?.label)
        assertEquals("disagreement", decision?.reason)
    }

    @Test
    fun highScoreWithAmbiguousClassMarginStillRequiresReview() {
        val gate = PlantDecisionGate()

        assertNull(gate.add(prediction("Late_blight", 0.97f, top2Margin = 0.02f)))
        assertNull(gate.add(prediction("Late_blight", 0.96f, top2Margin = 0.03f)))
        val decision = gate.add(prediction("Late_blight", 0.95f, top2Margin = 0.04f))

        assertEquals("Uncertain", decision?.label)
        assertEquals("ambiguous_class", decision?.reason)
        assertEquals(0.95f, decision?.confidence)
    }

    @Test
    fun sameLabelWithDifferentCapitalizationStillAgrees() {
        val gate = PlantDecisionGate()

        assertNull(gate.add(prediction("healthy", 0.87f)))
        val decision = gate.add(prediction("Healthy", 0.86f))

        assertEquals("healthy", decision?.label)
        assertEquals(DecisionStatus.OK, decision?.status)
        assertEquals("primary_agreement", decision?.reason)
    }

    @Test
    fun uncertainLabelNeverAgreesWithItselfAsAnActionableResult() {
        val gate = PlantDecisionGate()

        assertNull(gate.add(prediction("uncertain", 0.99f)))
        assertNull(gate.add(prediction("Uncertain", 0.98f)))
        val decision = gate.add(prediction("Uncertain", 0.97f))

        assertEquals(DecisionStatus.UNCERTAIN, decision?.status)
        assertEquals("Uncertain", decision?.label)
        assertEquals("no_actionable_label", decision?.reason)
    }

    @Test
    fun explicitUncertainFlagBlocksAnOtherwiseHighConfidenceFrame() {
        val gate = PlantDecisionGate()

        assertNull(gate.add(prediction("Healthy", 0.99f, isUncertain = true)))
        assertNull(gate.add(prediction("Healthy", 0.98f, isUncertain = true)))
        val decision = gate.add(prediction("Healthy", 0.97f, isUncertain = true))

        assertEquals(DecisionStatus.UNCERTAIN, decision?.status)
        assertEquals("model_abstained", decision?.reason)
    }

    private fun prediction(
        label: String,
        confidence: Float,
        top2Margin: Float? = null,
        isUncertain: Boolean = label.equals("Uncertain", ignoreCase = true),
    ) = FramePrediction(
        label = label,
        confidence = confidence,
        rawLabel = label,
        isDisease = label != "Healthy" && label != "Uncertain",
        isUncertain = isUncertain,
        latencyMs = 80.0,
        modelVersion = "test-model",
        top2Margin = top2Margin,
    )
}
