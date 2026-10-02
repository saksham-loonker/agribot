package com.sakshyam.agribot.ml.vision

import kotlin.math.exp
import kotlin.math.ln

/**
 * Turns per-member classifier logits into one calibrated probability vector.
 *
 * combined = log(mean_i softmax(logits_i)); combined[otherIndex] += otherBias;
 * probabilities = softmax(combined / temperature). The label is argmax (identical for any T > 0).
 * Optional energy gate: if T * logsumexp(combined[known] / T) < threshold the leaf is treated as Other.
 * All constants come from the model manifest and were chosen on validation data only.
 */
class EnsembleDecision(
    private val numClasses: Int,
    private val otherIndex: Int,
    private val otherBias: Double,
    private val temperature: Double,
    private val energyRejectBelow: Double?,
) {
    init {
        require(otherIndex in 0 until numClasses)
        require(temperature > 0.0)
    }

    data class Result(val probabilities: FloatArray, val labelIndex: Int, val energy: Double, val rejectedByEnergy: Boolean)

    fun combine(memberLogits: List<FloatArray>): Result {
        require(memberLogits.isNotEmpty())
        val mean = DoubleArray(numClasses)
        for (z in memberLogits) {
            require(z.size == numClasses) { "member output ${z.size} != $numClasses classes" }
            val p = softmax(DoubleArray(numClasses) { z[it].toDouble() })
            for (k in 0 until numClasses) mean[k] += p[k] / memberLogits.size
        }
        val combined = DoubleArray(numClasses) { ln(mean[it] + 1e-12) }
        combined[otherIndex] += otherBias
        val scaled = DoubleArray(numClasses) { combined[it] / temperature }
        val probs = softmax(scaled)
        var maxKnown = Double.NEGATIVE_INFINITY
        for (k in 0 until numClasses) if (k != otherIndex) maxKnown = maxOf(maxKnown, scaled[k])
        var sum = 0.0
        for (k in 0 until numClasses) if (k != otherIndex) sum += exp(scaled[k] - maxKnown)
        val energy = temperature * (ln(sum) + maxKnown)
        var arg = 0
        for (k in 1 until numClasses) if (probs[k] > probs[arg]) arg = k
        val rejected = energyRejectBelow != null && energy < energyRejectBelow
        return Result(FloatArray(numClasses) { probs[it].toFloat() }, if (rejected) otherIndex else arg, energy, rejected)
    }

    companion object {
        fun softmax(z: DoubleArray): DoubleArray {
            val m = z.max()
            val e = DoubleArray(z.size) { exp(z[it] - m) }
            val s = e.sum()
            return DoubleArray(z.size) { e[it] / s }
        }
    }
}
