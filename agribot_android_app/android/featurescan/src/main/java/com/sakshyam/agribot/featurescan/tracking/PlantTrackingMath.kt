package com.sakshyam.agribot.featurescan.tracking

import com.sakshyam.agribot.domain.model.BoundingBox
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Pure utility functions for plant tracking mathematics.
 * These functions have no Android dependencies and can be unit tested directly.
 */
object PlantTrackingMath {

    /**
     * Calculate Intersection over Union (IoU) between two bounding boxes.
     * Returns a value between 0 and 1, where 1 means perfect overlap.
     */
    fun calculateIoU(box1: BoundingBox, box2: BoundingBox): Float {
        val intersectionWidth = max(0f, min(box1.right, box2.right) - max(box1.left, box2.left))
        val intersectionHeight = max(0f, min(box1.bottom, box2.bottom) - max(box1.top, box2.top))
        val intersectionArea = intersectionWidth * intersectionHeight

        val box1Area = box1.width * box1.height
        val box2Area = box2.width * box2.height

        return if (box1Area + box2Area - intersectionArea > 0) {
            intersectionArea / (box1Area + box2Area - intersectionArea)
        } else {
            0f
        }
    }

    /**
     * Calculate cosine similarity between two feature vectors.
     * Returns a value between 0 and 1, where 1 means identical features.
     */
    fun calculateCosineSimilarity(feat1: FloatArray, feat2: FloatArray): Float {
        if (feat1.isEmpty() || feat2.isEmpty() || feat1.size != feat2.size) return 0f

        var dotProduct = 0f
        var norm1 = 0f
        var norm2 = 0f

        for (i in feat1.indices) {
            dotProduct += feat1[i] * feat2[i]
            norm1 += feat1[i] * feat1[i]
            norm2 += feat2[i] * feat2[i]
        }

        if (norm1 == 0f || norm2 == 0f) return 0f

        val similarity = dotProduct / (sqrt(norm1) * sqrt(norm2))
        return max(0f, min(1f, similarity))
    }

    /**
     * Calculate spatial distance score between a tracked plant and a new bounding box.
     * Returns a normalized score where 1.0 means very close, 0.0 means far away.
     */
    fun calculateSpatialDistanceScore(
        plantBbox: BoundingBox,
        newBbox: BoundingBox
    ): Float {
        val centerXDiff = abs(plantBbox.centerX - newBbox.centerX)
        val centerYDiff = abs(plantBbox.centerY - newBbox.centerY)
        val maxDistance = sqrt(plantBbox.width * plantBbox.width + plantBbox.height * plantBbox.height)

        if (maxDistance == 0f) return 0f

        val normalizedDistance = sqrt(centerXDiff * centerXDiff + centerYDiff * centerYDiff) / maxDistance
        return 1f - min(1f, normalizedDistance.toFloat())
    }

    /**
     * Calculate combined match score between an existing plant and a new detection.
     * Uses weighted combination of IoU, spatial distance, and cosine similarity.
     *
     * @return Score between 0 and 1, where 1 means perfect match
     */
    fun calculateMatchScore(
        existingBbox: BoundingBox,
        newBbox: BoundingBox,
        existingFeatures: FloatArray,
        newFeatures: FloatArray
    ): Float {
        val iouScore = calculateIoU(existingBbox, newBbox)
        val distanceScore = calculateMotionTolerantSpatialScore(existingBbox, newBbox)
        val cosineScore = calculateCosineSimilarity(existingFeatures, newFeatures)
        val scaleScore = calculateScaleSimilarity(existingBbox, newBbox)

        // IoU is excellent while the phone is steady, but it collapses as the
        // operator walks and the same plant shifts outside the previous box.
        // A bounded motion-tolerant centre score keeps the stable identity
        // without allowing distant, similarly sized plants to merge.
        return (0.45f * iouScore +
            0.35f * distanceScore +
            0.15f * scaleScore +
            0.05f * cosineScore).coerceIn(0f, 1f)
    }

    private fun calculateMotionTolerantSpatialScore(
        existingBbox: BoundingBox,
        newBbox: BoundingBox,
    ): Float {
        val centerDistance = kotlin.math.hypot(
            existingBbox.centerX - newBbox.centerX,
            existingBbox.centerY - newBbox.centerY,
        )
        val tolerance = 4f * max(
            max(existingBbox.width, existingBbox.height),
            max(newBbox.width, newBbox.height),
        ).coerceAtLeast(1f)
        return (1f - centerDistance / tolerance).coerceIn(0f, 1f)
    }

    private fun calculateScaleSimilarity(first: BoundingBox, second: BoundingBox): Float {
        val widthSimilarity = min(first.width, second.width) /
            max(first.width, second.width).coerceAtLeast(1f)
        val heightSimilarity = min(first.height, second.height) /
            max(first.height, second.height).coerceAtLeast(1f)
        return ((widthSimilarity + heightSimilarity) / 2f).coerceIn(0f, 1f)
    }

    /**
     * Extract features from a bounding box and confidence for similarity matching.
     * Features include:
     * - Center X position
     * - Center Y position
     * - Width
     * - Height
     * - Confidence
     * - Aspect ratio (width/height)
     * - Area ratio (width * height)
     */
    fun extractFeatures(
        bbox: BoundingBox,
        confidence: Float
    ): FloatArray {
        val aspectRatio = if (bbox.height > 0) bbox.width / bbox.height else 0f
        val areaRatio = bbox.width * bbox.height

        return floatArrayOf(
            bbox.centerX,
            bbox.centerY,
            bbox.width,
            bbox.height,
            confidence,
            aspectRatio,
            areaRatio,
        )
    }
}
