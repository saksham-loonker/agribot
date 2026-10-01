package com.sakshyam.agribot.camera

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class RgbaFrameConverterTest {
    // 3x2 image, pixel value = index (R channel), row stride padded to 16 bytes.
    private val w = 3
    private val h = 2
    private val stride = 16
    private val rgba = ByteArray(stride * h).also { b -> for (y in 0 until h) for (x in 0 until w) b[y * stride + x * 4] = (y * w + x).toByte() }

    private fun r(u: RgbaFrameConverter.Upright) = u.argb.map { (it shr 16) and 0xFF }

    @Test fun `no rotation keeps order and opaque alpha`() {
        val u = RgbaFrameConverter.toUpright(rgba, stride, 0, 0, w, h, 0)
        assertEquals(listOf(0, 1, 2, 3, 4, 5), r(u))
        assertEquals(0xFF, (u.argb[0] ushr 24))
    }

    @Test fun `90 degrees clockwise`() {
        val u = RgbaFrameConverter.toUpright(rgba, stride, 0, 0, w, h, 90)
        assertEquals(2, u.width); assertEquals(3, u.height)
        assertEquals(listOf(3, 0, 4, 1, 5, 2), r(u))
    }

    @Test fun `180 and 270 degrees`() {
        assertEquals(listOf(5, 4, 3, 2, 1, 0), r(RgbaFrameConverter.toUpright(rgba, stride, 0, 0, w, h, 180)))
        assertEquals(listOf(2, 5, 1, 4, 0, 3), r(RgbaFrameConverter.toUpright(rgba, stride, 0, 0, w, h, 270)))
    }

    @Test fun `crop is applied before rotation`() {
        val u = RgbaFrameConverter.toUpright(rgba, stride, 1, 0, 2, 2, 0)
        assertArrayEquals(intArrayOf(1, 2, 4, 5), r(u).toIntArray())
    }
}
