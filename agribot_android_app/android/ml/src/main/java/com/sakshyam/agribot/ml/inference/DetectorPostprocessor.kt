package com.sakshyam.agribot.ml.inference

import com.sakshyam.agribot.domain.model.BoundingBox
import kotlin.math.max
import kotlin.math.min

data class Detection(
    val bbox: BoundingBox,
    val confidence: Float,
    val classIndex: Int = 0,
)

object DetectorPostprocessor {
    fun nms(detections: List<Detection>, iouThreshold: Float = 0.45f): List<Detection> {
        val selected = mutableListOf<Detection>()
        for (candidate in detections.sortedByDescending { it.confidence }) {
            if (selected.none { iou(it.bbox, candidate.bbox) > iouThreshold }) {
                selected.add(candidate)
            }
        }
        return selected
    }

    fun filterCandidate(
        detection: Detection,
        frameWidth: Int,
        frameHeight: Int,
        minAreaFraction: Float = 0.002f,
        minShortestSidePx: Float = 12f,
    ): Boolean {
        val frameArea = (frameWidth * frameHeight).toFloat().coerceAtLeast(1f)
        val area = detection.bbox.width * detection.bbox.height
        val shortest = min(detection.bbox.width, detection.bbox.height)
        val ratio = detection.bbox.width / detection.bbox.height.coerceAtLeast(1f)
        return area >= frameArea * minAreaFraction &&
            shortest >= minShortestSidePx &&
            ratio in 0.2f..5.0f
    }

    private fun iou(a: BoundingBox, b: BoundingBox): Float {
        val left = max(a.left, b.left)
        val top = max(a.top, b.top)
        val right = min(a.right, b.right)
        val bottom = min(a.bottom, b.bottom)
        val intersection = max(0f, right - left) * max(0f, bottom - top)
        val union = a.width * a.height + b.width * b.height - intersection
        return if (union <= 0f) 0f else intersection / union
    }
}
