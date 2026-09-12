package com.sakshyam.agribot.ml.inference

import kotlin.math.exp
import kotlin.math.ln

/**
 * Converts the model's overconfident softmax-like output into a conservative
 * decision score.  The exported classifier returns probabilities, but those
 * values are not calibrated probabilities on camera/screen inputs.  Applying
 * temperature scaling preserves the top class while preventing a single
 * unsupported crop from being presented as near-certain evidence.
 */
object ClassifierProbabilityCalibrator {
    const val DEFAULT_TEMPERATURE = 3.0f

    fun temperatureScale(values: FloatArray, temperature: Float = DEFAULT_TEMPERATURE): FloatArray {
        if (values.isEmpty()) return FloatArray(0)

        val safeTemperature = temperature
            .takeIf { it.isFinite() && it > 0f }
            ?: DEFAULT_TEMPERATURE
        val finite = values.map { value ->
            value.takeIf { it.isFinite() && it >= 0f } ?: 0f
        }
        val total = finite.sum().takeIf { it.isFinite() && it > 0f } ?: return uniform(values.size)
        val logScores = finite.map { value ->
            ln((value / total).coerceAtLeast(MIN_PROBABILITY).toDouble()) / safeTemperature
        }
        val maximum = logScores.maxOrNull() ?: return uniform(values.size)
        val exponentials = logScores.map { score -> exp((score - maximum).coerceIn(-80.0, 80.0)) }
        val exponentialTotal = exponentials.sum()
            .takeIf { it.isFinite() && it > 0.0 }
            ?: return uniform(values.size)
        return FloatArray(values.size) { index ->
            (exponentials[index] / exponentialTotal).toFloat().coerceIn(0f, 1f)
        }
    }

    private fun uniform(size: Int): FloatArray = FloatArray(size) { 1f / size }

    private const val MIN_PROBABILITY = 1.0e-12f
}
