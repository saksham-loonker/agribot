package com.sakshyam.agribot.ml.inference

import com.sakshyam.agribot.domain.model.AGRIBOT_LABEL_ORDER
import com.sakshyam.agribot.domain.model.BoundingBox
import com.sakshyam.agribot.domain.model.FrontOverviewCandidate
import com.sakshyam.agribot.domain.model.ScanConstants
import kotlin.math.absoluteValue
import kotlin.math.exp
import kotlin.math.roundToInt

object DetectorOutputParser {
    private const val MIN_BOX_AREA_RATIO = 0.002f
    private const val MIN_BOX_SIDE_PX = 12f
    private const val MIN_ASPECT_RATIO = 0.25f
    private const val MAX_ASPECT_RATIO = 4.0f

    data class ParseDiagnostics(
        val rawCandidateCount: Int,
        val scoreValidCount: Int,
        val acceptedBeforeNms: Int,
        val acceptedAfterNms: Int,
        val maxRawConfidence: Float?,
        val rejectionCounts: Map<String, Int>,
    )

    data class ParseResult(
        val candidates: List<FrontOverviewCandidate>,
        val diagnostics: ParseDiagnostics,
    )

    fun parseNmsRows(
        rows: Array<FloatArray>, frameWidth: Int, frameHeight: Int,
        inputWidth: Int, inputHeight: Int, minScore: Float = 0.25f,
        coordinateSpace: DetectorCoordinateSpace = DetectorCoordinateSpace.AUTO,
        transform: DetectorInputTransform? = null,
        maxCandidates: Int = ScanConstants.MAX_DETECTOR_CANDIDATES,
    ): List<FrontOverviewCandidate> = parseNmsRowsDetailed(
        rows, frameWidth, frameHeight, inputWidth, inputHeight, minScore,
        coordinateSpace, transform, maxCandidates,
    ).candidates

    fun parseNmsRowsDetailed(
        rows: Array<FloatArray>, frameWidth: Int, frameHeight: Int,
        inputWidth: Int, inputHeight: Int, minScore: Float = 0.25f,
        coordinateSpace: DetectorCoordinateSpace = DetectorCoordinateSpace.AUTO,
        transform: DetectorInputTransform? = null,
        maxCandidates: Int = ScanConstants.MAX_DETECTOR_CANDIDATES,
        labels: List<String> = emptyList(),
    ): ParseResult {
        val reasons = mutableMapOf<String, Int>()
        fun reject(reason: String) { reasons[reason] = (reasons[reason] ?: 0) + 1 }
        var scoreValid = 0
        var maxRaw: Float? = null
        val parsed = rows.mapNotNull { row ->
            if (row.size < 6) { reject("malformed_row"); return@mapNotNull null }
            if (row.take(6).any { !it.isFinite() }) { reject("non_finite"); return@mapNotNull null }
            val score = row[4]
            if (score !in 0f..1f || score < minScore) { reject("score_below_threshold"); return@mapNotNull null }
            scoreValid += 1
            maxRaw = maxOf(maxRaw ?: score, score)
            val left = mapX(row[0], inputWidth, frameWidth, coordinateSpace, transform)
            val top = mapY(row[1], inputHeight, frameHeight, coordinateSpace, transform)
            val right = mapX(row[2], inputWidth, frameWidth, coordinateSpace, transform)
            val bottom = mapY(row[3], inputHeight, frameHeight, coordinateSpace, transform)
            if (right <= left || bottom <= top) { reject("invalid_box"); return@mapNotNull null }
            val bbox = BoundingBox(
                left.coerceIn(0f, frameWidth.toFloat()), top.coerceIn(0f, frameHeight.toFloat()),
                right.coerceIn(0f, frameWidth.toFloat()), bottom.coerceIn(0f, frameHeight.toFloat()),
            )
            if (!bbox.isValidPlantCandidate(frameWidth, frameHeight)) { reject("geometry_filter"); return@mapNotNull null }
            val classIndex = row[5].roundToIntOrNull()
            val detectorLabel = classIndex?.let { labels.getOrNull(it) }
            FrontOverviewCandidate(bbox, score, detectorLabel ?: "Uncertain", 0f, detectorLabel ?: "crop")
        }.sortedByDescending { it.detectorConfidence }
        val kept = parsed.take(maxCandidates.coerceAtLeast(0))
        return ParseResult(kept, ParseDiagnostics(rows.size, scoreValid, parsed.size, kept.size, maxRaw, reasons.toMap()))
    }

    fun parseRawYoloOutput(
        rows: Array<FloatArray>, frameWidth: Int, frameHeight: Int,
        inputWidth: Int, inputHeight: Int, minScore: Float = 0.25f,
        iouThreshold: Float = 0.45f, maxCandidates: Int = 300,
        coordinateSpace: DetectorCoordinateSpace = DetectorCoordinateSpace.AUTO,
        transform: DetectorInputTransform? = null, classCount: Int? = null,
        hasObjectness: Boolean = false, labels: List<String> = emptyList(),
    ): List<FrontOverviewCandidate> = parseRawYoloOutputDetailed(
        rows, frameWidth, frameHeight, inputWidth, inputHeight, minScore,
        iouThreshold, maxCandidates, coordinateSpace, transform, classCount,
        hasObjectness, labels,
    ).candidates

    fun parseRawYoloOutputDetailed(
        rows: Array<FloatArray>, frameWidth: Int, frameHeight: Int,
        inputWidth: Int, inputHeight: Int, minScore: Float = 0.25f,
        iouThreshold: Float = 0.45f, maxCandidates: Int = 300,
        coordinateSpace: DetectorCoordinateSpace = DetectorCoordinateSpace.AUTO,
        transform: DetectorInputTransform? = null, classCount: Int? = null,
        hasObjectness: Boolean = false, labels: List<String> = emptyList(),
    ): ParseResult {
        if (rows.size < 5 || rows.any { it.isEmpty() }) {
            return ParseResult(emptyList(), ParseDiagnostics(0, 0, 0, 0, null, mapOf("malformed_tensor" to 1)))
        }
        val candidateCount = rows.minOf { it.size }
        val reasons = mutableMapOf<String, Int>()
        fun reject(reason: String) { reasons[reason] = (reasons[reason] ?: 0) + 1 }
        var scoreValid = 0
        var maxRaw: Float? = null
        val candidates = (0 until candidateCount).mapNotNull { index ->
            if ((0..3).any { field -> !rows[field][index].isFinite() }) {
                reject("non_finite_coordinates"); return@mapNotNull null
            }
            val scoreInfo = scoreFor(rows, index, classCount, hasObjectness, labels)
            val score = scoreInfo?.score
            if (score == null || !score.isFinite() || score !in 0f..1f || score < minScore) {
                reject(if (scoreInfo == null) "malformed_score" else "score_below_threshold")
                return@mapNotNull null
            }
            scoreValid += 1
            maxRaw = maxOf(maxRaw ?: score, score)
            val centerX = rows[0][index]
            val centerY = rows[1][index]
            val width = rows[2][index]
            val height = rows[3][index]
            val left = mapX(centerX - width / 2f, inputWidth, frameWidth, coordinateSpace, transform)
            val top = mapY(centerY - height / 2f, inputHeight, frameHeight, coordinateSpace, transform)
            val right = mapX(centerX + width / 2f, inputWidth, frameWidth, coordinateSpace, transform)
            val bottom = mapY(centerY + height / 2f, inputHeight, frameHeight, coordinateSpace, transform)
            if (right <= left || bottom <= top) { reject("invalid_box"); return@mapNotNull null }
            val bbox = BoundingBox(
                left.coerceIn(0f, frameWidth.toFloat()), top.coerceIn(0f, frameHeight.toFloat()),
                right.coerceIn(0f, frameWidth.toFloat()), bottom.coerceIn(0f, frameHeight.toFloat()),
            )
            if (!bbox.isValidPlantCandidate(frameWidth, frameHeight)) { reject("geometry_filter"); return@mapNotNull null }
            FrontOverviewCandidate(
                bboxPx = bbox, detectorConfidence = score, label = scoreInfo.label,
                confidence = 0f, rawLabel = scoreInfo.label.takeIf { it != "Uncertain" } ?: "crop",
            )
        }.sortedByDescending { it.detectorConfidence }
        val kept = mutableListOf<FrontOverviewCandidate>()
        for (candidate in candidates) {
            if (kept.none { overlap(candidate, it) > iouThreshold }) kept += candidate
            if (kept.size >= maxCandidates.coerceAtLeast(0)) break
        }
        return ParseResult(
            kept,
            ParseDiagnostics(candidateCount, scoreValid, candidates.size, kept.size, maxRaw, reasons.toMap()),
        )
    }

    private data class ScoreInfo(val score: Float, val label: String)

    private fun scoreFor(
        rows: Array<FloatArray>, index: Int, classCount: Int?, hasObjectness: Boolean,
        labels: List<String>,
    ): ScoreInfo? {
        // Compatibility for the historical five-field test fixture. Production
        // bundles declare classCount and therefore score all class channels.
        if (classCount == null) {
            val score = rows.getOrNull(4)?.getOrNull(index) ?: return null
            if (!score.isFinite()) return null
            return ScoreInfo(toProbability(score), "Uncertain")
        }
        val classStart = if (hasObjectness) 5 else 4
        val available = minOf(classCount.coerceAtLeast(0), rows.size - classStart)
        if (available <= 0) return null
        val objectness = if (hasObjectness) {
            rows[4].getOrNull(index)?.takeIf { it.isFinite() }?.let(::toProbability) ?: return null
        } else 1f
        var bestIndex = -1
        var bestClassScore = Float.NEGATIVE_INFINITY
        for (classOffset in 0 until available) {
            val value = rows[classStart + classOffset].getOrNull(index) ?: return null
            if (!value.isFinite()) continue
            val probability = toProbability(value)
            if (probability > bestClassScore) { bestClassScore = probability; bestIndex = classOffset }
        }
        if (bestIndex < 0 || !bestClassScore.isFinite()) return null
        val label = labels.getOrNull(bestIndex)
            ?: AGRIBOT_LABEL_ORDER.getOrNull(bestIndex)?.displayName
            ?: "Uncertain"
        return ScoreInfo((objectness * bestClassScore).coerceIn(0f, 1f), label)
    }

    private fun toProbability(value: Float): Float = when {
        !value.isFinite() -> Float.NaN
        value in 0f..1f -> value
        else -> (1f / (1f + exp(-value.coerceIn(-80f, 80f)))).coerceIn(0f, 1f)
    }

    private fun Float.roundToIntOrNull(): Int? =
        takeIf { isFinite() }?.roundToInt()?.takeIf { kotlin.math.abs(this - it) < 0.01f }

    private fun overlap(first: FrontOverviewCandidate, second: FrontOverviewCandidate): Float {
        val left = maxOf(first.bboxPx.left, second.bboxPx.left)
        val top = maxOf(first.bboxPx.top, second.bboxPx.top)
        val right = minOf(first.bboxPx.right, second.bboxPx.right)
        val bottom = minOf(first.bboxPx.bottom, second.bboxPx.bottom)
        val intersection = (right - left).coerceAtLeast(0f) * (bottom - top).coerceAtLeast(0f)
        val firstArea = first.bboxPx.width.coerceAtLeast(0f) * first.bboxPx.height.coerceAtLeast(0f)
        val secondArea = second.bboxPx.width.coerceAtLeast(0f) * second.bboxPx.height.coerceAtLeast(0f)
        val union = firstArea + secondArea - intersection
        return if (union <= 0f) 0f else intersection / union
    }

    private fun BoundingBox.isValidPlantCandidate(frameWidth: Int, frameHeight: Int): Boolean {
        if (width < MIN_BOX_SIDE_PX || height < MIN_BOX_SIDE_PX) return false
        val frameArea = (frameWidth.toLong() * frameHeight.toLong()).toFloat().coerceAtLeast(1f)
        if (width * height < frameArea * MIN_BOX_AREA_RATIO) return false
        val aspectRatio = width / height.coerceAtLeast(1f)
        return aspectRatio in MIN_ASPECT_RATIO..MAX_ASPECT_RATIO
    }

    private fun mapX(value: Float, inputSize: Int, frameSize: Int,
        coordinateSpace: DetectorCoordinateSpace, transform: DetectorInputTransform?): Float {
        val modelPixels = toModelPixels(value, inputSize, coordinateSpace)
        return transform?.modelXToFrame(modelPixels) ?: modelPixels / inputSize.coerceAtLeast(1) * frameSize
    }

    private fun mapY(value: Float, inputSize: Int, frameSize: Int,
        coordinateSpace: DetectorCoordinateSpace, transform: DetectorInputTransform?): Float {
        val modelPixels = toModelPixels(value, inputSize, coordinateSpace)
        return transform?.modelYToFrame(modelPixels) ?: modelPixels / inputSize.coerceAtLeast(1) * frameSize
    }

    private fun toModelPixels(value: Float, inputSize: Int, coordinateSpace: DetectorCoordinateSpace): Float = when (coordinateSpace) {
        DetectorCoordinateSpace.MODEL_INPUT_PIXELS -> value
        DetectorCoordinateSpace.NORMALIZED -> value * inputSize
        DetectorCoordinateSpace.AUTO -> if (value.absoluteValue <= 1.5f) value * inputSize else value
    }
}

enum class DetectorCoordinateSpace {
    MODEL_INPUT_PIXELS,
    NORMALIZED,
    AUTO,
}
