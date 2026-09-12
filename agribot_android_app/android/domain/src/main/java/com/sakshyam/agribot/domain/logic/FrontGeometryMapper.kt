package com.sakshyam.agribot.domain.logic

import com.sakshyam.agribot.domain.model.BoundingBox
import com.sakshyam.agribot.domain.model.FrontCaptureCalibration
import com.sakshyam.agribot.domain.model.FrontGeometryAssignment
import com.sakshyam.agribot.domain.model.RowSide
import kotlin.math.abs
import kotlin.math.roundToInt

class FrontGeometryMapper(
    private val calibration: FrontCaptureCalibration,
    private val minimumGeometryConfidence: Float = 0.70f,
) {
    fun assign(bbox: BoundingBox, confidence: Float): FrontGeometryAssignment {
        if (calibration.calibrationQuality < minimumGeometryConfidence) {
            return unknown(bbox, "ambiguous_geometry")
        }
        if (confidence < minimumGeometryConfidence) {
            return unknown(bbox, "low_detector_confidence")
        }

        val leftDistance = abs(bbox.centerX - calibration.guideRailLeftPx)
        val rightDistance = abs(bbox.centerX - calibration.guideRailRightPx)
        val railSeparation = abs(calibration.guideRailRightPx - calibration.guideRailLeftPx).coerceAtLeast(1f)
        val nearestDistance = minOf(leftDistance, rightDistance)
        val halfSeparation = (railSeparation / 2f).coerceAtLeast(1f)
        val assignmentConfidence = (
            (1f - nearestDistance / halfSeparation).coerceIn(0f, 1f) *
                calibration.calibrationQuality
            ).coerceIn(0f, 1f)

        if (assignmentConfidence < minimumGeometryConfidence) {
            return unknown(bbox, "ambiguous_geometry")
        }

        val side = if (leftDistance < rightDistance) RowSide.LEFT else RowSide.RIGHT
        val rowId = if (side == RowSide.LEFT) calibration.leftRowId else calibration.rightRowId
        val nearestPlant = if (side == RowSide.LEFT) calibration.nearestLeftPlantNumber else calibration.nearestRightPlantNumber
        val normalizedDepth = (1f - (bbox.centerY / calibration.frameHeightPx.coerceAtLeast(1))).coerceIn(0f, 1f)
        val columnOffset = (normalizedDepth * 4f).roundToInt()
        val plantNumber = nearestPlant + columnOffset
        return FrontGeometryAssignment(
            bboxPx = bbox,
            rowSide = side,
            rowId = rowId,
            plantColumnEstimate = columnOffset + 1,
            plantNumber = plantNumber,
            rowAssignmentConfidence = assignmentConfidence,
            plantPositionConfidence = assignmentConfidence,
            geometryReason = "row_rail_fit_and_spacing_estimate",
        )
    }

    private fun unknown(bbox: BoundingBox, reason: String) = FrontGeometryAssignment(
        bboxPx = bbox,
        rowSide = RowSide.UNKNOWN,
        rowId = null,
        plantColumnEstimate = null,
        plantNumber = null,
        rowAssignmentConfidence = 0f,
        plantPositionConfidence = 0f,
        geometryReason = reason,
    )
}
