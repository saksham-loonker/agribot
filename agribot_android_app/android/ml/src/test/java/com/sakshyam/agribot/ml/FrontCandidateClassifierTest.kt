package com.sakshyam.agribot.ml

import com.sakshyam.agribot.domain.model.BoundingBox
import com.sakshyam.agribot.domain.model.FramePrediction
import com.sakshyam.agribot.domain.model.FrontOverviewCandidate
import com.sakshyam.agribot.ml.inference.FrontCandidateClassifier
import com.sakshyam.agribot.ml.preprocessing.RgbImage
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class FrontCandidateClassifierTest {
    @Test
    fun replacesDetectorPlaceholderWithClassifierPredictionFromCrop() {
        val capturedCrops = mutableListOf<RgbImage>()
        val classified = FrontCandidateClassifier.classifyCandidates(
            image = image(width = 20, height = 20),
            candidates = listOf(
                FrontOverviewCandidate(
                    bboxPx = BoundingBox(left = 5f, top = 6f, right = 15f, bottom = 16f),
                    detectorConfidence = 0.88f,
                    label = "Uncertain",
                    confidence = 0f,
                    rawLabel = "crop",
                ),
            ),
        ) { crop ->
            capturedCrops += crop
            FramePrediction(
                label = "Late_blight",
                confidence = 0.91f,
                rawLabel = "Late_blight",
                isDisease = true,
                isUncertain = false,
                latencyMs = 12.0,
                modelVersion = "test",
            )
        }

        assertEquals(1, capturedCrops.size)
        assertEquals(12, capturedCrops.single().width)
        assertEquals(12, capturedCrops.single().height)
        assertEquals("Late_blight", classified.single().label)
        assertEquals("Late_blight", classified.single().rawLabel)
        assertEquals(0.91f, classified.single().confidence)
        assertEquals(0.88f, classified.single().detectorConfidence)
    }

    @Test
    fun limitsSideScanClassificationToCandidatesNearestThePhoneCentre() {
        var classifierCalls = 0
        val candidates = listOf(
            candidate(left = 1f, top = 1f, right = 5f, bottom = 5f, confidence = 0.99f),
            candidate(left = 8f, top = 8f, right = 12f, bottom = 12f, confidence = 0.60f),
            candidate(left = 16f, top = 16f, right = 19f, bottom = 19f, confidence = 0.98f),
        )

        val classified = FrontCandidateClassifier.classifyCandidates(
            image = image(width = 20, height = 20),
            candidates = candidates,
            maxClassifiedCandidates = 1,
            focusX = 10f,
            focusY = 10f,
        ) {
            classifierCalls += 1
            FramePrediction(
                label = "Healthy",
                confidence = 0.9f,
                rawLabel = "Healthy",
                isDisease = false,
                isUncertain = false,
                latencyMs = 10.0,
                modelVersion = "test",
            )
        }

        assertEquals(1, classifierCalls)
        assertEquals("Healthy", classified[1].label)
        assertEquals("detector_only", classified[0].rawLabel)
        assertEquals("detector_only", classified[2].rawLabel)
        assertTrue(classified[0].confidence == 0f)
    }

    private fun candidate(left: Float, top: Float, right: Float, bottom: Float, confidence: Float) =
        FrontOverviewCandidate(
            bboxPx = BoundingBox(left, top, right, bottom),
            detectorConfidence = confidence,
            label = "Uncertain",
            confidence = 0f,
            rawLabel = "crop",
        )

    private fun image(width: Int, height: Int): RgbImage =
        RgbImage(
            width = width,
            height = height,
            pixels = IntArray(width * height) { index -> 0x398B35 + (index % 16) },
        )
}
