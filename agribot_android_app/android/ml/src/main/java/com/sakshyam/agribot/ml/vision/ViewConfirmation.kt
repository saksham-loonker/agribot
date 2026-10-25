package com.sakshyam.agribot.ml.vision

import com.sakshyam.agribot.domain.scan.VerdictKind
import com.sakshyam.agribot.domain.scan.VerdictPolicy

/** Corroborate a newly recovered diagnosis without disturbing an existing baseline decision. */
internal class ViewConfirmation(
    private val decision: EnsembleDecision,
    private val healthyIndex: Int,
    private val otherIndex: Int,
) {
    private val policy = VerdictPolicy()

    private fun key(logits: FloatArray): Pair<VerdictKind, Int> {
        val p = decision.combine(listOf(logits)).probabilities
        val order = p.indices.sortedByDescending { p[it] }
        val top = order[0]
        val confidence = p[top]
        val margin = confidence - p[order[1]]
        val kind = when (top) {
            healthyIndex -> if (confidence >= policy.healthyMinProb && margin >= policy.healthyMinMargin) VerdictKind.HEALTHY else VerdictKind.UNSURE
            otherIndex -> if (confidence >= policy.otherMinProb) VerdictKind.UNKNOWN_CONDITION else VerdictKind.UNSURE
            else -> if (confidence >= policy.diseaseMinProb && margin >= policy.diseaseMinMargin) VerdictKind.DISEASE else VerdictKind.UNSURE
        }
        return kind to top
    }

    fun needsConfirmation(reference: FloatArray, candidate: FloatArray): Boolean {
        val proposed = key(candidate)
        return proposed.first in listOf(VerdictKind.HEALTHY, VerdictKind.DISEASE) && proposed != key(reference)
    }

    fun confirms(candidate: FloatArray, fullRoi: FloatArray): Boolean = key(candidate) == key(fullRoi)
}
