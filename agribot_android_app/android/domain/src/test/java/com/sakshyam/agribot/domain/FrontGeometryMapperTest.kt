package com.sakshyam.agribot.domain

import com.sakshyam.agribot.domain.logic.FrontGeometryMapper
import com.sakshyam.agribot.domain.model.BoundingBox
import com.sakshyam.agribot.domain.model.FrontCaptureCalibration
import com.sakshyam.agribot.domain.model.RowSide
import kotlin.test.Test
import kotlin.test.assertEquals

class FrontGeometryMapperTest {
    @Test
    fun clearLeftAndRightDetectionsAreAssignedToConfiguredRows() {
        val mapper = FrontGeometryMapper(
            FrontCaptureCalibration(
                cameraHeightM = 1.1,
                cameraDistanceToNearestRowM = 1.0,
                cameraTiltDegrees = 15.0,
                rowSpacingM = 1.4,
                plantSpacingM = 0.45,
                leftRowId = "A",
                rightRowId = "B",
                nearestLeftPlantNumber = 10,
                nearestRightPlantNumber = 20,
                guideRailLeftPx = 120f,
                guideRailRightPx = 520f,
                frameWidthPx = 640,
                frameHeightPx = 480,
                calibrationQuality = 0.9f,
            )
        )

        val left = mapper.assign(BoundingBox(100f, 220f, 180f, 340f), confidence = 0.82f)
        val right = mapper.assign(BoundingBox(500f, 230f, 570f, 350f), confidence = 0.81f)

        assertEquals(RowSide.LEFT, left.rowSide)
        assertEquals("A", left.rowId)
        assertEquals(RowSide.RIGHT, right.rowSide)
        assertEquals("B", right.rowId)
    }

    @Test
    fun ambiguousCenterDetectionStaysUnknownInsteadOfGuessing() {
        val mapper = FrontGeometryMapper(
            FrontCaptureCalibration(
                cameraHeightM = 1.1,
                cameraDistanceToNearestRowM = 1.0,
                cameraTiltDegrees = 15.0,
                rowSpacingM = 1.4,
                plantSpacingM = 0.45,
                leftRowId = "A",
                rightRowId = "B",
                nearestLeftPlantNumber = 10,
                nearestRightPlantNumber = 20,
                guideRailLeftPx = 120f,
                guideRailRightPx = 520f,
                frameWidthPx = 640,
                frameHeightPx = 480,
                calibrationQuality = 0.55f,
            )
        )

        val assigned = mapper.assign(BoundingBox(300f, 220f, 340f, 330f), confidence = 0.8f)

        assertEquals(RowSide.UNKNOWN, assigned.rowSide)
        assertEquals(null, assigned.rowId)
        assertEquals("ambiguous_geometry", assigned.geometryReason)
    }
}
