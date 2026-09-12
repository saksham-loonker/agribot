package com.sakshyam.agribot.ml.inference

/**
 * Describes the only detector output layouts supported by the Android runtime.
 *
 * Ultralytics exports a raw detector as [1, fields, candidates] while an NMS
 * export is [1, candidates, fields].  Treating those dimensions as
 * interchangeable silently produces plausible-looking but incorrect boxes.
 */
data class DetectorTensorLayout(
    val format: Format,
    val candidateCount: Int,
    val fieldCount: Int,
) {
    enum class Format {
        NMS_XYXY_SCORE_CLASS,
        RAW_YOLO_FEATURES_FIRST,
        RAW_YOLO_CANDIDATES_FIRST,
        UNSUPPORTED,
    }

    val isSupported: Boolean
        get() = format != Format.UNSUPPORTED && candidateCount > 0 && fieldCount > 0

    companion object {
        fun from(
            shape: IntArray,
            outputLayoutHint: String? = null,
            expectedCandidates: Int? = null,
        ): DetectorTensorLayout {
            if (shape.size != 3 || shape[0] != 1) {
                return unsupported()
            }

            val first = shape[1]
            val last = shape[2]
            val hinted = outputLayoutHint?.trim()?.lowercase()
            if (hinted != null) {
                val format = when (hinted) {
                    "builtin_nms_xyxy_score_class", "nms_xyxy_score_class", "nms" ->
                        Format.NMS_XYXY_SCORE_CLASS
                    "raw_yolo_features_first", "raw_features_first", "raw_yolo" ->
                        Format.RAW_YOLO_FEATURES_FIRST
                    "raw_yolo_candidates_first", "raw_candidates_first" ->
                        Format.RAW_YOLO_CANDIDATES_FIRST
                    else -> null
                } ?: return unsupported()
                val layout = when (format) {
                    Format.NMS_XYXY_SCORE_CLASS -> DetectorTensorLayout(format, first, last)
                    Format.RAW_YOLO_FEATURES_FIRST -> DetectorTensorLayout(format, last, first)
                    Format.RAW_YOLO_CANDIDATES_FIRST -> DetectorTensorLayout(format, first, last)
                    Format.UNSUPPORTED -> unsupported()
                }
                if (!layout.isSupported) return unsupported()
                if (expectedCandidates != null && layout.candidateCount != expectedCandidates) {
                    return unsupported()
                }
                return layout
            }

            // A raw output with [1, 1344, 12] is dimensionally ambiguous with
            // an NMS output. Never silently choose NMS in that case; a manifest
            // hint is required for production bundles.
            if (first >= 1000 && last >= 8 && last <= 64 && expectedCandidates == null) {
                return unsupported()
            }
            return when {
                // NMS output: [batch, candidate_count, 6+].
                last >= 6 && first > last -> DetectorTensorLayout(
                    format = Format.NMS_XYXY_SCORE_CLASS,
                    candidateCount = first,
                    fieldCount = last,
                )

                // Raw YOLO output: [batch, 5+, candidate_count].
                first >= 5 && last > first -> DetectorTensorLayout(
                    format = Format.RAW_YOLO_FEATURES_FIRST,
                    candidateCount = last,
                    fieldCount = first,
                )

                // Some converters transpose raw YOLO output to
                // [batch, candidate_count, 5+].
                last >= 5 && first > last -> DetectorTensorLayout(
                    format = Format.RAW_YOLO_CANDIDATES_FIRST,
                    candidateCount = first,
                    fieldCount = last,
                )

                else -> unsupported()
            }
        }

        private fun unsupported(): DetectorTensorLayout = DetectorTensorLayout(
            format = Format.UNSUPPORTED,
            candidateCount = 0,
            fieldCount = 0,
        )
    }
}
