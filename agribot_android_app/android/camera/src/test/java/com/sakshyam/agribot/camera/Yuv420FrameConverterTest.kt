package com.sakshyam.agribot.camera

import kotlin.test.assertContentEquals
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class Yuv420FrameConverterTest {
    @Test
    fun convertsNeutralBlackAndWhiteFrames() {
        val black = Yuv420FrameConverter.toRgbPixels(
            width = 2,
            height = 2,
            y = byteArrayOf(16, 16, 16, 16),
            u = byteArrayOf(128.toByte()),
            v = byteArrayOf(128.toByte()),
            yRowStride = 2,
            uRowStride = 1,
            vRowStride = 1,
            uPixelStride = 1,
            vPixelStride = 1,
        )
        val white = Yuv420FrameConverter.toRgbPixels(
            width = 2,
            height = 2,
            y = byteArrayOf(235.toByte(), 235.toByte(), 235.toByte(), 235.toByte()),
            u = byteArrayOf(128.toByte()),
            v = byteArrayOf(128.toByte()),
            yRowStride = 2,
            uRowStride = 1,
            vRowStride = 1,
            uPixelStride = 1,
            vPixelStride = 1,
        )

        assertEquals(0x000000, black.first())
        assertEquals(0xffffff, white.first())
    }

    @Test
    fun honorsChromaPixelStride() {
        val pixels = Yuv420FrameConverter.toRgbPixels(
            width = 4,
            height = 2,
            y = ByteArray(8) { 81 },
            u = byteArrayOf(90.toByte(), 0, 240.toByte(), 0),
            v = byteArrayOf(240.toByte(), 0, 90.toByte(), 0),
            yRowStride = 4,
            uRowStride = 4,
            vRowStride = 4,
            uPixelStride = 2,
            vPixelStride = 2,
        )

        assertEquals(pixels[0], pixels[1])
        assertEquals(pixels[2], pixels[3])
        assertTrue(pixels[0] != pixels[2])
    }

    @Test
    fun appliesNonZeroCropOffsetsToLumaAndChromaPlanes() {
        val fullLuma = ByteArray(16) { index ->
            when (index) {
                5, 6, 9, 10 -> 235.toByte()
                else -> 16
            }
        }
        val pixels = Yuv420FrameConverter.toRgbPixels(
            width = 2,
            height = 2,
            y = fullLuma,
            u = ByteArray(4) { 128.toByte() },
            v = ByteArray(4) { 128.toByte() },
            yRowStride = 4,
            uRowStride = 2,
            vRowStride = 2,
            uPixelStride = 1,
            vPixelStride = 1,
            cropLeft = 1,
            cropTop = 1,
        )

        assertContentEquals(IntArray(4) { 0xffffff }, pixels)
    }

    @Test
    fun rotatesRgbFramesToMatchCameraRotation() {
        val pixels = intArrayOf(
            0x010203, 0x040506, 0x070809,
            0x0a0b0c, 0x0d0e0f, 0x101112,
        )

        val analyzer = SideScanAnalyzer(targetFps = 5) { }

        assertContentEquals(intArrayOf(0x0a0b0c, 0x010203, 0x0d0e0f, 0x040506, 0x101112, 0x070809), analyzer.rotateForTest(pixels, 3, 2, 90))
        assertContentEquals(intArrayOf(0x101112, 0x0d0e0f, 0x0a0b0c, 0x070809, 0x040506, 0x010203), analyzer.rotateForTest(pixels, 3, 2, 180))
        assertContentEquals(intArrayOf(0x070809, 0x101112, 0x040506, 0x0d0e0f, 0x010203, 0x0a0b0c), analyzer.rotateForTest(pixels, 3, 2, 270))
    }
}
