package com.sakshyam.agribot.ml

import com.sakshyam.agribot.ml.inference.DetectorOutputParser
import com.sakshyam.agribot.ml.inference.DetectorCoordinateSpace
import com.sakshyam.agribot.ml.inference.DetectorInputTransform
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DetectorOutputParserTest {
    @Test
    fun parsesNmsRowsAndScalesInputCoordinatesToFramePixels() {
        val rows = arrayOf(
            floatArrayOf(25.6f, 51.2f, 128f, 204.8f, 0.86f, 0f),
            floatArrayOf(0f, 0f, 20f, 20f, 0.10f, 0f),
        )

        val candidates = DetectorOutputParser.parseNmsRows(
            rows = rows,
            frameWidth = 640,
            frameHeight = 480,
            inputWidth = 256,
            inputHeight = 256,
            minScore = 0.25f,
        )

        assertEquals(1, candidates.size)
        assertEquals(64f, candidates.single().bboxPx.left, absoluteTolerance = 0.01f)
        assertEquals(96f, candidates.single().bboxPx.top, absoluteTolerance = 0.01f)
        assertEquals(320f, candidates.single().bboxPx.right, absoluteTolerance = 0.01f)
        assertEquals(384f, candidates.single().bboxPx.bottom, absoluteTolerance = 0.01f)
        assertEquals(0.86f, candidates.single().detectorConfidence, absoluteTolerance = 0.0001f)
        assertEquals("crop", candidates.single().rawLabel)
        assertEquals("Uncertain", candidates.single().label)
        assertEquals(0f, candidates.single().confidence)
    }

    @Test
    fun parsesNormalizedCoordinates() {
        val rows = arrayOf(floatArrayOf(0.1f, 0.2f, 0.5f, 0.8f, 0.76f, 0f))

        val candidates = DetectorOutputParser.parseNmsRows(
            rows = rows,
            frameWidth = 640,
            frameHeight = 480,
            inputWidth = 256,
            inputHeight = 256,
            minScore = 0.25f,
        )

        assertEquals(1, candidates.size)
        assertEquals(64f, candidates.single().bboxPx.left, absoluteTolerance = 0.01f)
        assertEquals(96f, candidates.single().bboxPx.top, absoluteTolerance = 0.01f)
        assertEquals(320f, candidates.single().bboxPx.right, absoluteTolerance = 0.01f)
        assertEquals(384f, candidates.single().bboxPx.bottom, absoluteTolerance = 0.01f)
    }

    @Test
    fun capsCandidatesAfterSortingByDetectorConfidence() {
        val rows = arrayOf(
            floatArrayOf(0f, 0f, 80f, 80f, 0.40f, 0f),
            floatArrayOf(160f, 160f, 240f, 240f, 0.90f, 0f),
        )

        val candidates = DetectorOutputParser.parseNmsRows(
            rows = rows,
            frameWidth = 640,
            frameHeight = 480,
            inputWidth = 256,
            inputHeight = 256,
            maxCandidates = 1,
        )

        assertEquals(1, candidates.size)
        assertEquals(0.90f, candidates.single().detectorConfidence, absoluteTolerance = 0.0001f)
    }

    @Test
    fun rejectsMalformedLowScoreAndEmptyBoxes() {
        val rows = arrayOf(
            floatArrayOf(0.1f, 0.2f, 0.5f),
            floatArrayOf(0.1f, 0.2f, 0.5f, 0.8f, 0.10f, 0f),
            floatArrayOf(0.5f, 0.8f, 0.1f, 0.2f, 0.90f, 0f),
        )

        val candidates = DetectorOutputParser.parseNmsRows(
            rows = rows,
            frameWidth = 640,
            frameHeight = 480,
            inputWidth = 256,
            inputHeight = 256,
            minScore = 0.25f,
        )

        assertTrue(candidates.isEmpty())
    }

    @Test
    fun rejectsNonFiniteAndNonProbabilityScores() {
        val rows = arrayOf(
            floatArrayOf(10f, 10f, 80f, 80f, Float.NaN, 0f),
            floatArrayOf(10f, 10f, 80f, 80f, Float.POSITIVE_INFINITY, 0f),
            floatArrayOf(10f, 10f, 80f, 80f, 1.2f, 0f),
        )

        val candidates = DetectorOutputParser.parseNmsRows(
            rows = rows,
            frameWidth = 640,
            frameHeight = 480,
            inputWidth = 256,
            inputHeight = 256,
        )

        assertTrue(candidates.isEmpty())
    }

    @Test
    fun rejectsNmsRowsWithTinyOrImplausiblePlantBoxes() {
        val rows = arrayOf(
            // 10x15 px after scaling: below the model/frame minimum.
            floatArrayOf(40f, 40f, 44f, 48f, 0.90f, 0f),
            // A very tall box is rejected by the aspect-ratio guard.
            floatArrayOf(80f, 20f, 88f, 126.67f, 0.89f, 0f),
            // 350x40 px after scaling: implausibly wide crop candidate.
            floatArrayOf(20f, 20f, 160f, 41.33f, 0.88f, 0f),
            // Valid 120x120 px candidate.
            floatArrayOf(100f, 80f, 148f, 144f, 0.87f, 0f),
        )

        val candidates = DetectorOutputParser.parseNmsRows(
            rows = rows,
            frameWidth = 640,
            frameHeight = 480,
            inputWidth = 256,
            inputHeight = 256,
            minScore = 0.25f,
        )

        assertEquals(1, candidates.size)
        assertEquals(250f, candidates.single().bboxPx.left, absoluteTolerance = 0.01f)
        assertEquals(150f, candidates.single().bboxPx.top, absoluteTolerance = 0.01f)
        assertEquals(370f, candidates.single().bboxPx.right, absoluteTolerance = 0.01f)
        assertEquals(270f, candidates.single().bboxPx.bottom, absoluteTolerance = 0.01f)
    }

    @Test
    fun parsesRawYoloOutputAndAppliesNms() {
        val raw = Array(5) { FloatArray(3) }
        raw[0][0] = 0.50f
        raw[1][0] = 0.50f
        raw[2][0] = 0.40f
        raw[3][0] = 0.20f
        raw[4][0] = 0.90f
        raw[0][1] = 0.51f
        raw[1][1] = 0.50f
        raw[2][1] = 0.40f
        raw[3][1] = 0.20f
        raw[4][1] = 0.70f
        raw[0][2] = 0.10f
        raw[1][2] = 0.10f
        raw[2][2] = 0.10f
        raw[3][2] = 0.10f
        raw[4][2] = 0.20f

        val candidates = DetectorOutputParser.parseRawYoloOutput(
            rows = raw,
            frameWidth = 640,
            frameHeight = 480,
            inputWidth = 256,
            inputHeight = 256,
            minScore = 0.25f,
            iouThreshold = 0.45f,
        )

        assertEquals(1, candidates.size)
        assertEquals(192f, candidates.single().bboxPx.left, absoluteTolerance = 0.01f)
        assertEquals(192f, candidates.single().bboxPx.top, absoluteTolerance = 0.01f)
        assertEquals(448f, candidates.single().bboxPx.right, absoluteTolerance = 0.01f)
        assertEquals(288f, candidates.single().bboxPx.bottom, absoluteTolerance = 0.01f)
        assertEquals(0.90f, candidates.single().detectorConfidence, absoluteTolerance = 0.0001f)
    }

    @Test
    fun rejectsRawYoloCandidatesWithTinyOrImplausiblePlantBoxesBeforeNms() {
        val raw = Array(5) { FloatArray(4) }
        // 10x15 px after scaling: below the model/frame minimum.
        raw[0][0] = 55f
        raw[1][0] = 80f
        raw[2][0] = 4f
        raw[3][0] = 8f
        raw[4][0] = 0.90f
        // A very tall box is rejected by the aspect-ratio guard.
        raw[0][1] = 120f
        raw[1][1] = 100f
        raw[2][1] = 8f
        raw[3][1] = 106.67f
        raw[4][1] = 0.89f
        // 350x40 px after scaling: implausibly wide crop candidate.
        raw[0][2] = 110f
        raw[1][2] = 60f
        raw[2][2] = 140f
        raw[3][2] = 21.33f
        raw[4][2] = 0.88f
        // Valid 120x120 px candidate.
        raw[0][3] = 124f
        raw[1][3] = 112f
        raw[2][3] = 48f
        raw[3][3] = 64f
        raw[4][3] = 0.87f

        val candidates = DetectorOutputParser.parseRawYoloOutput(
            rows = raw,
            frameWidth = 640,
            frameHeight = 480,
            inputWidth = 256,
            inputHeight = 256,
            minScore = 0.25f,
            iouThreshold = 0.45f,
        )

        assertEquals(1, candidates.size)
        assertEquals(250f, candidates.single().bboxPx.left, absoluteTolerance = 0.01f)
        assertEquals(150f, candidates.single().bboxPx.top, absoluteTolerance = 0.01f)
        assertEquals(370f, candidates.single().bboxPx.right, absoluteTolerance = 0.01f)
        assertEquals(270f, candidates.single().bboxPx.bottom, absoluteTolerance = 0.01f)
    }

    @Test
    fun mapsLetterboxedRawCoordinatesBackToTheUnpaddedCameraFrame() {
        val raw = Array(5) { FloatArray(1) }
        raw[0][0] = 96f
        raw[1][0] = 128f
        raw[2][0] = 64f
        raw[3][0] = 128f
        raw[4][0] = 0.90f

        val candidates = DetectorOutputParser.parseRawYoloOutput(
            rows = raw,
            frameWidth = 640,
            frameHeight = 480,
            inputWidth = 256,
            inputHeight = 256,
            minScore = 0.25f,
            coordinateSpace = DetectorCoordinateSpace.MODEL_INPUT_PIXELS,
            transform = DetectorInputTransform(
                inputWidth = 256,
                inputHeight = 256,
                frameWidth = 640,
                frameHeight = 480,
                scaleX = 0.4f,
                scaleY = 0.4f,
                padX = 0f,
                padY = 32f,
            ),
        )

        assertEquals(1, candidates.size)
        assertEquals(160f, candidates.single().bboxPx.left, absoluteTolerance = 0.01f)
        assertEquals(80f, candidates.single().bboxPx.top, absoluteTolerance = 0.01f)
        assertEquals(320f, candidates.single().bboxPx.right, absoluteTolerance = 0.01f)
        assertEquals(400f, candidates.single().bboxPx.bottom, absoluteTolerance = 0.01f)
    }

    @Test
    fun mapsNormalizedRawCoordinatesBeforeApplyingLetterboxTransform() {
        val raw = Array(5) { FloatArray(1) }
        raw[0][0] = 0.50f
        raw[1][0] = 0.50f
        raw[2][0] = 0.25f
        raw[3][0] = 0.50f
        raw[4][0] = 0.90f

        val candidates = DetectorOutputParser.parseRawYoloOutput(
            rows = raw,
            frameWidth = 640,
            frameHeight = 480,
            inputWidth = 256,
            inputHeight = 256,
            minScore = 0.25f,
            coordinateSpace = DetectorCoordinateSpace.NORMALIZED,
            transform = DetectorInputTransform(
                inputWidth = 256,
                inputHeight = 256,
                frameWidth = 640,
                frameHeight = 480,
                scaleX = 0.4f,
                scaleY = 0.4f,
                padX = 0f,
                padY = 32f,
            ),
        )

        assertEquals(1, candidates.size)
        assertEquals(240f, candidates.single().bboxPx.left, absoluteTolerance = 0.01f)
        assertEquals(80f, candidates.single().bboxPx.top, absoluteTolerance = 0.01f)
        assertEquals(400f, candidates.single().bboxPx.right, absoluteTolerance = 0.01f)
        assertEquals(400f, candidates.single().bboxPx.bottom, absoluteTolerance = 0.01f)
    }

    @Test
    fun rawYoloUsesBestClassScoreInsteadOfOnlyClassZero() {
        val raw = Array(12) { FloatArray(1) }
        raw[0][0] = 128f
        raw[1][0] = 128f
        raw[2][0] = 96f
        raw[3][0] = 96f
        raw[4][0] = 0.04f // Early_blight
        raw[5][0] = 0.02f // Healthy
        raw[6][0] = 0.91f // Late_blight

        val candidates = DetectorOutputParser.parseRawYoloOutput(
            rows = raw,
            frameWidth = 256,
            frameHeight = 256,
            inputWidth = 256,
            inputHeight = 256,
            classCount = 8,
            labels = listOf("Early_blight", "Healthy", "Late_blight"),
        )

        assertEquals(1, candidates.size)
        assertEquals(0.91f, candidates.single().detectorConfidence, absoluteTolerance = 0.0001f)
        assertEquals("Late_blight", candidates.single().label)
    }

    @Test
    fun rawYoloCombinesObjectnessWithBestClassScore() {
        val raw = Array(6) { FloatArray(1) }
        raw[0][0] = 128f
        raw[1][0] = 128f
        raw[2][0] = 96f
        raw[3][0] = 96f
        raw[4][0] = 0.5f
        raw[5][0] = 0.8f

        val candidates = DetectorOutputParser.parseRawYoloOutput(
            rows = raw,
            frameWidth = 256,
            frameHeight = 256,
            inputWidth = 256,
            inputHeight = 256,
            minScore = 0.25f,
            classCount = 1,
            hasObjectness = true,
            labels = listOf("Healthy"),
        )

        assertEquals(1, candidates.size)
        assertEquals(0.4f, candidates.single().detectorConfidence, absoluteTolerance = 0.0001f)
    }
}
