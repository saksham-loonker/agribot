package com.sakshyam.agribot.ml

import com.sakshyam.agribot.ml.inference.DetectorInputPreprocessor
import com.sakshyam.agribot.ml.inference.RgbImageResizer
import com.sakshyam.agribot.ml.preprocessing.RgbImage
import kotlin.test.Test
import kotlin.test.assertEquals

class DetectorInputPreprocessorTest {
    @Test
    fun bilinearResizeInterpolatesTheCompleteChannelValue() {
        val image = RgbImage(width = 2, height = 1, pixels = intArrayOf(0xFF0000, 0x0000FF))

        val resized = RgbImageResizer.resizeBilinear(image, outWidth = 3, outHeight = 1)

        assertEquals(0xFF0000, resized.pixels[0])
        assertEquals(0x800080, resized.pixels[1])
        assertEquals(0x0000FF, resized.pixels[2])
    }

    @Test
    fun letterboxPreservesAspectRatioAndMapsModelBoxBackToFrame() {
        val image = RgbImage(width = 640, height = 480, pixels = IntArray(640 * 480) { 0x204020 })
        val input = DetectorInputPreprocessor.letterbox(image, inputWidth = 256, inputHeight = 256)

        assertEquals(256, input.image.width)
        assertEquals(256, input.image.height)
        assertEquals(0f, input.transform.modelXToFrame(0f), absoluteTolerance = 0.01f)
        assertEquals(0f, input.transform.modelYToFrame(32f), absoluteTolerance = 0.01f)
        assertEquals(640f, input.transform.modelXToFrame(256f), absoluteTolerance = 0.01f)
        assertEquals(480f, input.transform.modelYToFrame(224f), absoluteTolerance = 0.01f)
    }
}
