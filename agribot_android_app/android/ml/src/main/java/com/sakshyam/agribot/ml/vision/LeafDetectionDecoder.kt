package com.sakshyam.agribot.ml.vision

/** A detected leaf in source-frame pixels. */
data class LeafDetection(val box: PixelBox, val score: Float)

/**
 * Decodes the YOLOv9 leaf detector output `[1, 5, N]` whose rows are x1, y1, x2, y2 (letterboxed input
 * pixels) and score (already in 0..1). Greedy class-agnostic NMS. Mirrors `decode_detections` in
 * golden/canon_reference.py exactly (stable score sort, IoU <= threshold keeps, max detections).
 */
object LeafDetectionDecoder {
    fun decode(
        output: FloatArray,
        candidates: Int,
        letterbox: CanonicalImageOps.Letterbox,
        frameW: Int,
        frameH: Int,
        scoreThreshold: Double,
        nmsIou: Double,
        maxDetections: Int,
    ): List<LeafDetection> {
        require(output.size >= 5 * candidates) { "detector output too small: ${output.size} < ${5 * candidates}" }
        val n = candidates
        val thr = scoreThreshold
        // Scores compared in double precision like numpy (float32 score vs float64 threshold).
        val keep = (0 until n).filter { output[4 * n + it].toDouble() >= thr }
            .sortedByDescending { output[4 * n + it] }   // stable: ties keep index order
        val sel = ArrayList<DoubleArray>(maxDetections)
        val selected = ArrayList<LeafDetection>(maxDetections)
        for (i in keep) {
            val x1 = letterbox.toFrameX(output[i]).coerceIn(0.0, frameW.toDouble())
            val y1 = letterbox.toFrameY(output[n + i]).coerceIn(0.0, frameH.toDouble())
            val x2 = letterbox.toFrameX(output[2 * n + i]).coerceIn(0.0, frameW.toDouble())
            val y2 = letterbox.toFrameY(output[3 * n + i]).coerceIn(0.0, frameH.toDouble())
            if (x2 - x1 < 2.0 || y2 - y1 < 2.0) continue
            val box = doubleArrayOf(x1, y1, x2, y2)
            if (sel.all { iou(it, box) <= nmsIou }) {
                sel += box
                selected += LeafDetection(PixelBox(x1.toFloat(), y1.toFloat(), x2.toFloat(), y2.toFloat()), output[4 * n + i])
            }
            if (selected.size >= maxDetections) break
        }
        return selected
    }

    private fun iou(a: DoubleArray, b: DoubleArray): Double {
        val ix = maxOf(0.0, minOf(a[2], b[2]) - maxOf(a[0], b[0]))
        val iy = maxOf(0.0, minOf(a[3], b[3]) - maxOf(a[1], b[1]))
        val inter = ix * iy
        val union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return if (union > 0) inter / union else 0.0
    }
}
