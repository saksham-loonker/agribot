package com.sakshyam.agribot.domain.logic

import com.sakshyam.agribot.domain.model.DecisionStatus
import com.sakshyam.agribot.domain.model.FramePrediction
import com.sakshyam.agribot.domain.model.PlantDecision
import java.util.Locale

class PlantDecisionGate(
    private val highConfidence: Float = ClassifierDecisionPolicy.DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
    private val maxFrames: Int = PRIMARY_FRAMES + BACKUP_FRAMES,
    private val minimumTop2Margin: Float = 0.05f,
    private val maximumEntropy: Float = 0.92f,
) {
    private val frames = mutableListOf<FramePrediction>()

    val snapshot: List<FramePrediction>
        get() = frames.toList()

    fun reset() {
        frames.clear()
    }

    fun add(prediction: FramePrediction): PlantDecision? {
        frames.add(prediction)
        if (frames.size >= PRIMARY_FRAMES) {
            val first = frames[0]
            val second = frames[1]
            if (agreeHighConfidence(first, second)) {
                return decision(first.label, listOf(first, second), "primary_agreement")
            }
        }

        if (frames.size >= maxFrames) {
            return fallbackDecision()
        }

        return null
    }

    private fun agreeHighConfidence(left: FramePrediction, right: FramePrediction): Boolean =
        hasActionableLabel(left) &&
            hasActionableLabel(right) &&
            !left.isUncertain &&
            !right.isUncertain &&
            left.label.trim().equals(right.label.trim(), ignoreCase = true) &&
            meetsDecisionConfidence(left) &&
            meetsDecisionConfidence(right) &&
            !isAmbiguous(left) &&
            !isAmbiguous(right)

    private fun fallbackDecision(): PlantDecision {
        val actionable = frames.filter(::hasActionableLabel)
        val usable = frames.filter { frame ->
            hasActionableLabel(frame) &&
            !frame.isUncertain &&
            meetsDecisionConfidence(frame) &&
            !isAmbiguous(frame)
        }

        if (usable.isEmpty()) {
            val reason = when {
                actionable.isEmpty() -> "no_actionable_label"
                actionable.none(::meetsDecisionConfidence) -> "low_confidence"
                actionable.filter(::meetsDecisionConfidence).all { it.isUncertain } -> "model_abstained"
                else -> "ambiguous_class"
            }
            return decision(UNCERTAIN_LABEL, frames, reason)
        }

        val counts = usable.groupingBy { it.label.trim().lowercase(Locale.ROOT) }.eachCount()
        val maxVotes = counts.values.maxOrNull() ?: 0
        val tied = counts.values.count { it == maxVotes } > 1
        if (maxVotes >= 2 && !tied) {
            val normalizedLabel = counts.entries.first { it.value == maxVotes }.key
            val label = usable.first { it.label.trim().lowercase(Locale.ROOT) == normalizedLabel }.label
            return decision(
                label,
                usable.filter { it.label.trim().lowercase(Locale.ROOT) == normalizedLabel },
                "backup_majority",
            )
        }

        return decision(UNCERTAIN_LABEL, frames, "disagreement")
    }

    private fun isAmbiguous(frame: FramePrediction): Boolean =
        frame.top2Margin?.let { it < minimumTop2Margin } == true ||
            frame.entropy?.let { it > maximumEntropy } == true

    private fun hasActionableLabel(frame: FramePrediction): Boolean =
            frame.label.isNotBlank() &&
            !frame.label.equals(UNCERTAIN_LABEL, ignoreCase = true) &&
            !frame.label.equals("Unknown", ignoreCase = true)

    private fun meetsDecisionConfidence(frame: FramePrediction): Boolean =
        frame.confidence >= ClassifierDecisionPolicy.decisionThreshold(
            label = frame.label,
            highConfidenceThreshold = highConfidence,
        )

    private fun decision(label: String, confidenceFrames: List<FramePrediction>, reason: String): PlantDecision {
        val confidence = confidenceFrames.minOfOrNull { it.confidence } ?: 0f
        return PlantDecision(
            label = label,
            confidence = confidence,
            status = if (label.equals(UNCERTAIN_LABEL, ignoreCase = true)) DecisionStatus.UNCERTAIN else DecisionStatus.OK,
            framesUsed = frames.size,
            reason = reason,
        )
    }

    companion object {
        const val PRIMARY_FRAMES = 2
        const val BACKUP_FRAMES = 1
        const val UNCERTAIN_LABEL = "Uncertain"
    }
}
