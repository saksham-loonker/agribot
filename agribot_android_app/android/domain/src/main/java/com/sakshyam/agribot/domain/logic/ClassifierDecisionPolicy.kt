package com.sakshyam.agribot.domain.logic

import kotlin.math.max

/**
 * Shared confidence policy for both Side Scan and Front Row Overview.
 *
 * The classifier's softmax score is not a calibrated probability. Healthy and
 * disease labels also have different operational costs, so one global gate
 * makes the UI either abstain too often or surface weak disease guesses.
 */
object ClassifierDecisionPolicy {
    const val DEFAULT_UNCERTAIN_THRESHOLD = 0.62f
    const val DEFAULT_HIGH_CONFIDENCE_THRESHOLD = 0.90f

    /**
     * When the top-1 prediction is a disease label, the runner-up "Healthy"
     * score is treated as evidence that the model is hedging. If Healthy is
     * within this margin of the disease top-1, we treat the prediction as
     * uncertain rather than as a confirmed disease.
     *
     * The margin exists because a real "Late Blight" plant still receives
     * some non-zero probability mass on the Healthy channel. We want to
     * accept disease only when Healthy is clearly smaller.
     */
    const val HEALTHY_CHALLENGE_MARGIN = 0.18f

    /** A clear Healthy top-1 result may be accepted at the base threshold. */
    fun acceptanceThreshold(
        rawLabel: String,
        baseThreshold: Float,
        highConfidenceThreshold: Float = DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
    ): Float {
        val safeBase = baseThreshold.takeIf { it.isFinite() }?.coerceIn(0f, 1f) ?: DEFAULT_UNCERTAIN_THRESHOLD
        val safeHigh = highConfidenceThreshold.takeIf { it.isFinite() }?.coerceIn(0f, 1f)
            ?: DEFAULT_HIGH_CONFIDENCE_THRESHOLD
        return if (rawLabel.trim().equals("Healthy", ignoreCase = true)) {
            safeBase
        } else {
            max(safeBase, safeHigh)
        }
    }

    /** Healthy uses the lower acceptance bar; disease labels require stability and a higher bar. */
    fun decisionThreshold(
        label: String,
        highConfidenceThreshold: Float,
    ): Float = acceptanceThreshold(
        rawLabel = label,
        baseThreshold = DEFAULT_UNCERTAIN_THRESHOLD,
        highConfidenceThreshold = highConfidenceThreshold,
    )

    /**
     * True when a disease top-1 score is clearly dominant over the runner-up
     * Healthy probability. Returns true if the top-1 is Healthy (Healthy never
     * needs to defend itself against itself).
     */
    fun isDiseaseClearlyDominant(
        top1Label: String,
        top1Confidence: Float,
        healthyConfidence: Float?,
        margin: Float = HEALTHY_CHALLENGE_MARGIN,
    ): Boolean {
        if (top1Label.trim().equals("Healthy", ignoreCase = true)) return true
        if (!top1Confidence.isFinite()) return false
        if (healthyConfidence == null || !healthyConfidence.isFinite()) return true
        return top1Confidence - healthyConfidence >= margin
    }
}
