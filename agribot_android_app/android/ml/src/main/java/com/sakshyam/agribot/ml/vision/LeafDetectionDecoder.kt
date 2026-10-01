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
        scoreThreshold: Float,
        nmsIou: Float,
        maxDetections: Int,
    ): List<LeafDetection> {
        require(output.size >= 5 * candidates) { "detector output too small: ${output.size} < ${5 * candidates}" }
        val n = candidates
        val keep = (0 until n).filter { output[4 * n + it] >= scoreThreshold }
            .sortedByDescending { output[4 * n + it] }   // stable: ties keep index order
        val selected = ArrayList<LeafDetection>(maxDetections)
        for (i in keep) {
            val (fx1, fy1) = letterbox.toFrame(output[i], output[n + i])
            val (fx2, fy2) = letterbox.toFrame(output[2 * n + i], output[3 * n + i])
            val box = PixelBox(
                fx1.coerceIn(0f, frameW.toFloat()), fy1.coerceIn(0f, frameH.toFloat()),
                fx2.coerceIn(0f, frameW.toFloat()), fy2.coerceIn(0f, frameH.toFloat()),
            )
            if (box.width < 2f || box.height < 2f) continue
            if (selected.all { it.box.iou(box) <= nmsIou }) selected += LeafDetection(box, output[4 * n + i])
            if (selected.size >= maxDetections) break
        }
        return selected
    }
}
