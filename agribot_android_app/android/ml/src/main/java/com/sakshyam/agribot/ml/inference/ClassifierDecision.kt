package com.sakshyam.agribot.ml.inference

import com.sakshyam.agribot.domain.logic.ClassifierDecisionPolicy
import com.sakshyam.agribot.domain.model.FramePrediction

object ClassifierDecision {
    fun toPrediction(
        rawLabel: String,
        confidence: Float,
        threshold: Float,
        latencyMs: Double,
        modelVersion: String,
        highConfidenceThreshold: Float = ClassifierDecisionPolicy.DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
        top2Margin: Float? = null,
        entropy: Float? = null,
        healthyConfidence: Float? = null,
    ): FramePrediction {
        val safeConfidence = confidence.takeIf { it.isFinite() }?.coerceIn(0f, 1f) ?: 0f
        val normalizedRawLabel = rawLabel.trim().take(MAX_LABEL_LENGTH)
        val hasActionableLabel = normalizedRawLabel.isNotBlank() &&
            !normalizedRawLabel.equals("Unknown", ignoreCase = true) &&
            !normalizedRawLabel.equals("Uncertain", ignoreCase = true)
        val isHealthy = normalizedRawLabel.equals("Healthy", ignoreCase = true)
        val safeThreshold = ClassifierDecisionPolicy.acceptanceThreshold(
            rawLabel = normalizedRawLabel,
            baseThreshold = threshold,
            highConfidenceThreshold = highConfidenceThreshold,
        )
        val healthyChallengePassed = ClassifierDecisionPolicy.isDiseaseClearlyDominant(
            top1Label = normalizedRawLabel,
            top1Confidence = safeConfidence,
            healthyConfidence = healthyConfidence,
        )
        val isUncertain = safeConfidence < safeThreshold ||
            !hasActionableLabel ||
            (!isHealthy && !healthyChallengePassed)
        val label = when {
            isUncertain -> "Uncertain"
            isHealthy -> "Healthy"
            else -> normalizedRawLabel.ifBlank { "Uncertain" }
        }
        return FramePrediction(
            label = label,
            confidence = safeConfidence,
            rawLabel = normalizedRawLabel.ifBlank { "Unknown" },
            isDisease = !isHealthy && !isUncertain,
            isUncertain = isUncertain,
            latencyMs = latencyMs.takeIf { it.isFinite() }?.coerceAtLeast(0.0) ?: 0.0,
            modelVersion = modelVersion.trim().take(MAX_MODEL_VERSION_LENGTH).ifBlank { "unknown" },
            top2Margin = top2Margin?.takeIf { it.isFinite() }?.coerceIn(0f, 1f),
            entropy = entropy?.takeIf { it.isFinite() }?.coerceIn(0f, 1f),
            healthyConfidence = healthyConfidence?.takeIf { it.isFinite() }?.coerceIn(0f, 1f),
        )
    }

    private const val MAX_LABEL_LENGTH = 128
    private const val MAX_MODEL_VERSION_LENGTH = 128
}
