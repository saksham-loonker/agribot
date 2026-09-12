package com.sakshyam.agribot.ml

import com.sakshyam.agribot.domain.model.BoundingBox
import com.sakshyam.agribot.ml.inference.FrontCandidateCropper
import com.sakshyam.agribot.ml.preprocessing.RgbImage
import kotlin.test.Test
import kotlin.test.assertEquals

class FrontCandidateCropperTest {
    @Test
    fun padsCandidateBoxBeforeCropping() {
        val rect = FrontCandidateCropper.cropRect(
            image = image(width = 640, height = 480),
            bbox = BoundingBox(left = 100f, top = 120f, right = 220f, bottom = 260f),
        )

        assertEquals(93, rect.left)
        assertEquals(113, rect.top)
        assertEquals(227, rect.right)
        assertEquals(267, rect.bottom)
    }

    @Test
    fun clampsCandidateCropToFrameBounds() {
        val rect = FrontCandidateCropper.cropRect(
            image = image(width = 640, height = 480),
            bbox = BoundingBox(left = -10f, top = 2f, right = 40f, bottom = 90f),
        )

        assertEquals(0, rect.left)
        assertEquals(0, rect.top)
        assertEquals(44, rect.right)
        assertEquals(94, rect.bottom)
    }

    @Test
    fun cropsExpectedPixelsFromCandidateBox() {
        val source = image(width = 8, height = 6)

        val crop = FrontCandidateCropper.crop(
            image = source,
            bbox = BoundingBox(left = 2f, top = 1f, right = 6f, bottom = 5f),
            padRatio = 0f,
        )

        assertEquals(4, crop.width)
        assertEquals(4, crop.height)
        assertEquals(source.pixel(2, 1), crop.pixel(0, 0))
        assertEquals(source.pixel(5, 4), crop.pixel(3, 3))
    }

    private fun image(width: Int, height: Int): RgbImage =
        RgbImage(
            width = width,
            height = height,
            pixels = IntArray(width * height) { index -> index },
        )
}
