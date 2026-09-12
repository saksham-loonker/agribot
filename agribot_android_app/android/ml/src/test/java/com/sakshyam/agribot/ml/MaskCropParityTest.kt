package com.sakshyam.agribot.ml

import com.sakshyam.agribot.ml.preprocessing.CropRect
import com.sakshyam.agribot.ml.preprocessing.MaskCropPreprocessor
import com.sakshyam.agribot.ml.preprocessing.RgbImage
import kotlin.test.Test
import kotlin.test.assertEquals

class MaskCropParityTest {
    @Test
    fun smallImagesReturnOriginalBoundsLikePython() {
        val image = RgbImage(width = 3, height = 3, pixels = IntArray(9) { rgb(30, 200, 40) })

        assertEquals(CropRect(0, 0, 3, 3), MaskCropPreprocessor.maskCropBounds(image))
    }

    @Test
    fun missingMaskFallsBackToCenterCropScale() {
        val image = RgbImage(width = 100, height = 80, pixels = IntArray(100 * 80) { rgb(4, 4, 4) })

        assertEquals(CropRect(22, 18, 77, 62), MaskCropPreprocessor.maskCropBounds(image, cropPad = 0.05f, fallbackScale = 0.55f))
    }

    @Test
    fun largeVegetationBoxFallsBackToCenterCropLikePythonGuard() {
        val image = RgbImage(width = 50, height = 50, pixels = IntArray(50 * 50) { rgb(30, 190, 40) })

        assertEquals(CropRect(11, 11, 39, 39), MaskCropPreprocessor.maskCropBounds(image, cropPad = 0.05f, fallbackScale = 0.55f))
    }

    @Test
    fun maskCropBoundsMatchesPythonBgrHsvGoldenCase() {
        val pixels = IntArray(20 * 24) { index ->
            val x = index % 24
            val y = index / 24
            when {
                x in 7 until 16 && y in 5 until 12 -> rgb(35, 190, 45)
                x in 15 until 19 && y == 7 -> rgb(40, 185, 50)
                x in 4 until 8 && y in 14 until 17 -> rgb(30, 160, 35)
                x in 18 until 22 && y in 3 until 5 -> rgb(15, 90, 150)
                x in 18 until 22 && y in 16 until 18 -> rgb(5, 60, 5)
                else -> rgb(120, 120, 120)
            }
        }
        val image = RgbImage(width = 24, height = 20, pixels = pixels)

        assertEquals(CropRect(7, 5, 16, 12), MaskCropPreprocessor.maskCropBounds(image, cropPad = 0.05f, fallbackScale = 0.55f))
    }

    @Test
    fun redDiseasePixelsAreIncludedWithoutSwappingRgbChannels() {
        val pixels = IntArray(32 * 24) { index ->
            val x = index % 32
            val y = index / 32
            if (x in 10 until 22 && y in 7 until 17) rgb(185, 35, 25) else rgb(120, 120, 120)
        }
        val image = RgbImage(width = 32, height = 24, pixels = pixels)

        assertEquals(CropRect(10, 7, 22, 17), MaskCropPreprocessor.maskCropBounds(image, cropPad = 0f))
    }

    @Test
    fun preprocessFindsBoundsOnPreviewButPreservesOriginalPixels() {
        val image = RgbImage(width = 512, height = 384, pixels = IntArray(512 * 384) { rgb(120, 120, 120) })
        val preprocessed = MaskCropPreprocessor.preprocessForSideScan(image)

        assertEquals(282, preprocessed.width)
        assertEquals(212, preprocessed.height)
    }

    @Test
    fun fineImageDetailSurvivesCropExtraction() {
        val image = RgbImage(512, 384, IntArray(512 * 384) { index ->
            val value = if (index % 2 == 0) 100 else 140
            rgb(value, value, value)
        })
        val crop = MaskCropPreprocessor.preprocessForSideScan(image)
        assertEquals(setOf(rgb(100, 100, 100), rgb(140, 140, 140)), crop.pixels.toSet())
    }

    private fun rgb(red: Int, green: Int, blue: Int): Int =
        (red shl 16) or (green shl 8) or blue
}
