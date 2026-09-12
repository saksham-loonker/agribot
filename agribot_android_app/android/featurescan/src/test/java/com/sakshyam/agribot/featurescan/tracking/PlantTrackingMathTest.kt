package com.sakshyam.agribot.featurescan.tracking

import com.sakshyam.agribot.domain.model.BoundingBox
import com.sakshyam.agribot.domain.model.FrontOverviewCandidate
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class PlantTrackingMathTest {

    // ==================== IoU Tests ====================

    @Test
    fun iouReturnsOneForIdenticalBoxes() {
        val box = BoundingBox(10f, 20f, 50f, 80f)
        val iou = PlantTrackingMath.calculateIoU(box, box)
        assertEquals(1.0f, iou, 0.001f)
    }

    @Test
    fun iouReturnsZeroForNonOverlappingBoxes() {
        val box1 = BoundingBox(0f, 0f, 10f, 10f)
        val box2 = BoundingBox(20f, 20f, 30f, 30f)
        val iou = PlantTrackingMath.calculateIoU(box1, box2)
        assertEquals(0.0f, iou, 0.001f)
    }

    @Test
    fun iouReturnsCorrectValueForPartialOverlap() {
        // box1: (0,0) to (10,10) -> area = 100
        // box2: (5,5) to (15,15) -> area = 100
        // intersection: (5,5) to (10,10) -> area = 25
        // union = 100 + 100 - 25 = 175
        // IoU = 25/175 = 0.1428...
        val box1 = BoundingBox(0f, 0f, 10f, 10f)
        val box2 = BoundingBox(5f, 5f, 15f, 15f)
        val iou = PlantTrackingMath.calculateIoU(box1, box2)
        assertEquals(25f / 175f, iou, 0.001f)
    }

    @Test
    fun iouReturnsZeroForZeroAreaBoxes() {
        val box1 = BoundingBox(5f, 5f, 5f, 5f) // zero width and height
        val box2 = BoundingBox(0f, 0f, 10f, 10f)
        val iou = PlantTrackingMath.calculateIoU(box1, box2)
        assertEquals(0.0f, iou, 0.001f)
    }

    // ==================== Cosine Similarity Tests ====================

    @Test
    fun cosineSimilarityReturnsOneForIdenticalVectors() {
        val feat = floatArrayOf(1f, 2f, 3f, 4f)
        val similarity = PlantTrackingMath.calculateCosineSimilarity(feat, feat)
        assertEquals(1.0f, similarity, 0.001f)
    }

    @Test
    fun cosineSimilarityReturnsZeroForOrthogonalVectors() {
        val feat1 = floatArrayOf(1f, 0f, 0f)
        val feat2 = floatArrayOf(0f, 1f, 0f)
        val similarity = PlantTrackingMath.calculateCosineSimilarity(feat1, feat2)
        assertEquals(0.0f, similarity, 0.001f)
    }

    @Test
    fun cosineSimilarityReturnsZeroForEmptyVectors() {
        val similarity = PlantTrackingMath.calculateCosineSimilarity(floatArrayOf(), floatArrayOf())
        assertEquals(0.0f, similarity, 0.001f)
    }

    @Test
    fun cosineSimilarityReturnsZeroForMismatchedSizes() {
        val feat1 = floatArrayOf(1f, 2f, 3f)
        val feat2 = floatArrayOf(1f, 2f)
        val similarity = PlantTrackingMath.calculateCosineSimilarity(feat1, feat2)
        assertEquals(0.0f, similarity, 0.001f)
    }

    @Test
    fun cosineSimilarityReturnsZeroForZeroNormVectors() {
        val feat1 = floatArrayOf(0f, 0f, 0f)
        val feat2 = floatArrayOf(1f, 2f, 3f)
        val similarity = PlantTrackingMath.calculateCosineSimilarity(feat1, feat2)
        assertEquals(0.0f, similarity, 0.001f)
    }

    @Test
    fun cosineSimilarityClampsNegativeToZero() {
        // Vectors pointing in opposite directions -> negative cosine similarity
        // Should be clamped to 0
        val feat1 = floatArrayOf(1f, 0f)
        val feat2 = floatArrayOf(-1f, 0f)
        val similarity = PlantTrackingMath.calculateCosineSimilarity(feat1, feat2)
        assertEquals(0.0f, similarity, 0.001f)
    }

    // ==================== Spatial Distance Score Tests ====================

    @Test
    fun spatialDistanceScoreReturnsOneForSameCenter() {
        val bbox1 = BoundingBox(10f, 10f, 20f, 20f)
        val bbox2 = BoundingBox(10f, 10f, 20f, 20f)
        val score = PlantTrackingMath.calculateSpatialDistanceScore(bbox1, bbox2)
        assertEquals(1.0f, score, 0.001f)
    }

    @Test
    fun spatialDistanceScoreReturnsZeroForFarAwayBoxes() {
        val bbox1 = BoundingBox(0f, 0f, 10f, 10f)
        val bbox2 = BoundingBox(100f, 100f, 110f, 110f)
        val score = PlantTrackingMath.calculateSpatialDistanceScore(bbox1, bbox2)
        assertEquals(0.0f, score, 0.001f)
    }

    @Test
    fun spatialDistanceScoreReturnsZeroForZeroSizeBox() {
        val bbox1 = BoundingBox(5f, 5f, 5f, 5f) // zero size
        val bbox2 = BoundingBox(10f, 10f, 20f, 20f)
        val score = PlantTrackingMath.calculateSpatialDistanceScore(bbox1, bbox2)
        assertEquals(0.0f, score, 0.001f)
    }

    // ==================== Match Score Tests ====================

    @Test
    fun matchScoreReturnsOneForIdenticalBoxesAndFeatures() {
        val bbox = BoundingBox(10f, 10f, 50f, 50f)
        val features = PlantTrackingMath.extractFeatures(bbox, 0.9f)
        val score = PlantTrackingMath.calculateMatchScore(bbox, bbox, features, features)
        assertEquals(1.0f, score, 0.001f)
    }

    @Test
    fun matchScoreReturnsZeroForNonOverlappingBoxes() {
        val bbox1 = BoundingBox(0f, 0f, 10f, 10f)
        val bbox2 = BoundingBox(100f, 100f, 110f, 110f)
        val features1 = PlantTrackingMath.extractFeatures(bbox1, 0.9f)
        val features2 = PlantTrackingMath.extractFeatures(bbox2, 0.9f)
        val score = PlantTrackingMath.calculateMatchScore(bbox1, bbox2, features1, features2)
        // IoU=0 and distance=0, so score is only from cosine similarity (dominated by area feature)
        // Should be well below the 0.35f matching threshold
        assertTrue(score < 0.3f, "Score should be below matching threshold for non-overlapping boxes, got $score")
    }

    // ==================== Feature Extraction Tests ====================

    @Test
    fun extractFeaturesReturnsCorrectSize() {
        val bbox = BoundingBox(10f, 20f, 50f, 80f)
        val features = PlantTrackingMath.extractFeatures(bbox, 0.85f)
        assertEquals(7, features.size)
    }

    @Test
    fun extractFeaturesContainsCorrectValues() {
        val bbox = BoundingBox(10f, 20f, 50f, 80f)
        val features = PlantTrackingMath.extractFeatures(bbox, 0.85f)

        assertEquals(bbox.centerX, features[0], 0.001f)
        assertEquals(bbox.centerY, features[1], 0.001f)
        assertEquals(bbox.width, features[2], 0.001f)
        assertEquals(bbox.height, features[3], 0.001f)
        assertEquals(0.85f, features[4], 0.001f)
        // aspect ratio = width/height = 40/60 = 0.666...
        assertEquals(40f / 60f, features[5], 0.001f)
        // area = 40 * 60 = 2400
        assertEquals(2400f, features[6], 0.001f)
    }

    @Test
    fun extractFeaturesHandlesZeroHeight() {
        val bbox = BoundingBox(10f, 20f, 50f, 20f) // zero height
        val features = PlantTrackingMath.extractFeatures(bbox, 0.5f)
        assertEquals(0f, features[5], 0.001f) // aspect ratio should be 0
    }

    // ==================== Integration: Tracking Scenario Tests ====================

    @Test
    fun trackingScenarioSamePlantAcrossFrames() {
        // Simulate a plant detected in two consecutive frames with slight movement
        val bbox1 = BoundingBox(100f, 100f, 150f, 150f)
        val bbox2 = BoundingBox(102f, 101f, 152f, 151f) // slight movement

        val features1 = PlantTrackingMath.extractFeatures(bbox1, 0.9f)
        val features2 = PlantTrackingMath.extractFeatures(bbox2, 0.88f)

        val score = PlantTrackingMath.calculateMatchScore(bbox1, bbox2, features1, features2)

        // Should have a high match score since boxes are very similar
        assertTrue(score > 0.7f, "Score should be high for similar boxes, got $score")
    }

    @Test
    fun trackingScenarioKeepsSamePlantWhenWalkingShiftsItsBox() {
        val bbox1 = BoundingBox(100f, 100f, 150f, 150f)
        val bbox2 = BoundingBox(190f, 100f, 240f, 150f)
        val score = PlantTrackingMath.calculateMatchScore(
            bbox1,
            bbox2,
            PlantTrackingMath.extractFeatures(bbox1, 0.9f),
            PlantTrackingMath.extractFeatures(bbox2, 0.88f),
        )

        assertTrue(score > 0.35f, "Walking shift should remain matchable, got $score")
    }

    @Test
    fun trackingScenarioDifferentPlants() {
        // Simulate two different plants in the same frame
        val bbox1 = BoundingBox(100f, 100f, 150f, 150f)
        val bbox2 = BoundingBox(300f, 300f, 350f, 350f)

        val features1 = PlantTrackingMath.extractFeatures(bbox1, 0.9f)
        val features2 = PlantTrackingMath.extractFeatures(bbox2, 0.85f)

        val score = PlantTrackingMath.calculateMatchScore(bbox1, bbox2, features1, features2)

        // Should have a low match score since boxes are far apart
        assertTrue(score < 0.4f, "Score should be low for distant boxes, got $score")
    }

    @Test
    fun trackingScenarioReidentificationAfterOcclusion() {
        // Simulate a plant that was occluded and then re-detected
        // The features should still match well even after occlusion
        val bbox1 = BoundingBox(100f, 100f, 150f, 150f)
        val bbox2 = BoundingBox(105f, 105f, 155f, 155f) // slight movement after occlusion

        val features1 = PlantTrackingMath.extractFeatures(bbox1, 0.9f)
        val features2 = PlantTrackingMath.extractFeatures(bbox2, 0.87f)

        val cosineSim = PlantTrackingMath.calculateCosineSimilarity(features1, features2)
        val iou = PlantTrackingMath.calculateIoU(bbox1, bbox2)

        // Both cosine similarity and IoU should be reasonably high
        assertTrue(cosineSim > 0.9f, "Cosine similarity should be high for similar plants, got $cosineSim")
        assertTrue(iou > 0.5f, "IoU should be high for overlapping boxes, got $iou")
    }

    @Test
    fun trackConfidenceAlwaysBelongsToTheCurrentLabel() {
        val initial = TrackedPlant(
            plantId = "plant_1",
            createdAt = 1L,
            lastSeen = 1L,
            currentBbox = BoundingBox(100f, 100f, 150f, 150f),
            healthStatus = "Late_blight",
            confidence = 0.97f,
            positionHistory = emptyList(),
            detectionCount = 2,
            features = FloatArray(7),
        )

        val updated = initial.updateWithDetection(
            candidate = FrontOverviewCandidate(
                bboxPx = BoundingBox(101f, 100f, 151f, 150f),
                detectorConfidence = 0.91f,
                label = "Uncertain",
                confidence = 0.87f,
            ),
            timestamp = 2L,
            tilt = 0f,
            roll = 0f,
            pitch = 0f,
        )

        assertEquals("Late_blight", updated.healthStatus)
        assertEquals(0.97f, updated.confidence, 0.001f)
        assertEquals(null, updated.pendingHealthLabel)
        assertEquals(0, updated.pendingHealthLabelStreak)
    }

    @Test
    fun detectorOnlyTrackIsNotPresentedAsAHealthDiagnosis() {
        val initial = TrackedPlant(
            plantId = "plant_1",
            createdAt = 1L,
            lastSeen = 1L,
            currentBbox = BoundingBox(100f, 100f, 150f, 150f),
            healthStatus = "Uncertain",
            classificationPending = true,
            confidence = 0f,
            positionHistory = emptyList(),
            detectionCount = 1,
            features = FloatArray(7),
        )

        val classified = initial.updateWithDetection(
            candidate = FrontOverviewCandidate(
                bboxPx = BoundingBox(101f, 100f, 151f, 150f),
                detectorConfidence = 0.91f,
                label = "Healthy",
                confidence = 0.87f,
                rawLabel = "Healthy",
            ),
            timestamp = 2L,
            tilt = 0f,
            roll = 0f,
            pitch = 0f,
        )

        assertFalse(classified.classificationPending)
        assertEquals("Uncertain", classified.healthStatus)
        val confirmed = classified.updateWithDetection(
            FrontOverviewCandidate(BoundingBox(101f, 100f, 151f, 150f), 0.91f, "Healthy", 0.87f, "Healthy"),
            3L, 0f, 0f, 0f,
        )
        assertEquals("Healthy", confirmed.healthStatus)
        assertEquals(0.87f, confirmed.confidence, 0.001f)
    }

    @Test
    fun competingLabelNeedsTwoConsecutiveObservationsBeforeItReplacesStableLabel() {
        val initial = TrackedPlant(
            plantId = "plant_2",
            createdAt = 1L,
            lastSeen = 1L,
            currentBbox = BoundingBox(100f, 100f, 150f, 150f),
            healthStatus = "Late_blight",
            confidence = 0.90f,
            positionHistory = emptyList(),
            detectionCount = 2,
            features = FloatArray(7),
        )
        val healthy = FrontOverviewCandidate(
            bboxPx = BoundingBox(101f, 100f, 151f, 150f),
            detectorConfidence = 0.91f,
            label = "Healthy",
            confidence = 0.88f,
            rawLabel = "Healthy",
        )

        val first = initial.updateWithDetection(healthy, 2L, 0f, 0f, 0f)
        val second = first.updateWithDetection(healthy, 3L, 0f, 0f, 0f)

        assertEquals("Late_blight", first.healthStatus)
        assertEquals("Healthy", second.healthStatus)
        assertEquals(1, first.pendingHealthLabelStreak)
        assertEquals(0, second.pendingHealthLabelStreak)
    }
}
