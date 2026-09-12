package com.sakshyam.agribot.ml.inference

import com.sakshyam.agribot.domain.model.FramePrediction
import com.sakshyam.agribot.domain.model.FrontOverviewCandidate
import com.sakshyam.agribot.ml.preprocessing.RgbImage
import com.sakshyam.agribot.ml.preprocessing.PlantInputEvidence
import kotlin.math.hypot

object FrontCandidateClassifier {
    fun classifyCandidates(
        image: RgbImage,
        candidates: List<FrontOverviewCandidate>,
        maxClassifiedCandidates: Int = Int.MAX_VALUE,
        focusX: Float? = null,
        focusY: Float? = null,
        classifyCrop: (RgbImage) -> FramePrediction,
    ): List<FrontOverviewCandidate> {
        val limit = maxClassifiedCandidates.coerceAtLeast(0)
        val focus = if (focusX != null && focusY != null) focusX to focusY else null
        val usableCandidates = candidates.filter { candidate ->
            PlantInputEvidence.hasUsableColor(FrontCandidateCropper.crop(image, candidate.bboxPx))
        }
        val targets = usableCandidates.withIndex()
            .sortedWith(
                compareBy<IndexedValue<FrontOverviewCandidate>> { indexed ->
                    focus?.let { (x, y) ->
                        hypot(
                            indexed.value.bboxPx.centerX - x,
                            indexed.value.bboxPx.centerY - y,
                        )
                    } ?: 0f
                }.thenByDescending { indexed -> indexed.value.detectorConfidence },
            )
            .take(limit)
            .mapTo(mutableSetOf()) { it.index }

        return usableCandidates.mapIndexed { index, candidate ->
            if (index !in targets) {
                // Preserve every detector box for the overlay/tracker, but do
                // not spend a classifier invocation on distant plants during
                // Side Scan. A detector-only box is explicitly non-actionable.
                candidate.copy(label = "Uncertain", confidence = 0f, rawLabel = "detector_only")
            } else {
                val prediction = classifyCrop(FrontCandidateCropper.crop(image, candidate.bboxPx))
                candidate.copy(
                    label = prediction.label,
                    confidence = prediction.confidence,
                    rawLabel = prediction.rawLabel,
                    top2Margin = prediction.top2Margin,
                    entropy = prediction.entropy,
                )
            }
        }
    }
}
