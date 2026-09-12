package com.sakshyam.agribot.ml

import com.sakshyam.agribot.ml.inference.DetectorTensorLayout
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse

class DetectorTensorLayoutTest {
    @Test
    fun rawUltralyticsTensorUsesFeaturesFirstAxis() {
        val layout = DetectorTensorLayout.from(intArrayOf(1, 5, 1344))

        assertEquals(DetectorTensorLayout.Format.RAW_YOLO_FEATURES_FIRST, layout.format)
        assertEquals(1344, layout.candidateCount)
        assertEquals(5, layout.fieldCount)
    }

    @Test
    fun nmsTensorUsesCandidatesFirstAxis() {
        val layout = DetectorTensorLayout.from(intArrayOf(1, 300, 6))

        assertEquals(DetectorTensorLayout.Format.NMS_XYXY_SCORE_CLASS, layout.format)
        assertEquals(300, layout.candidateCount)
        assertEquals(6, layout.fieldCount)
    }

    @Test
    fun ambiguousSingleCandidateTensorIsRejectedInsteadOfMisparsed() {
        assertFalse(DetectorTensorLayout.from(intArrayOf(1, 5, 1)).isSupported)
    }

    @Test
    fun manifestHintDisambiguatesRawCandidatesFirstShape() {
        val layout = DetectorTensorLayout.from(
            shape = intArrayOf(1, 1344, 12),
            outputLayoutHint = "raw_yolo_candidates_first",
        )

        assertEquals(DetectorTensorLayout.Format.RAW_YOLO_CANDIDATES_FIRST, layout.format)
        assertEquals(1344, layout.candidateCount)
        assertEquals(12, layout.fieldCount)
    }

    @Test
    fun manifestHintSupportsSingleNmsOutput() {
        val layout = DetectorTensorLayout.from(
            shape = intArrayOf(1, 1, 6),
            outputLayoutHint = "builtin_nms_xyxy_score_class",
        )

        assertEquals(DetectorTensorLayout.Format.NMS_XYXY_SCORE_CLASS, layout.format)
        assertEquals(1, layout.candidateCount)
        assertEquals(6, layout.fieldCount)
    }
}
