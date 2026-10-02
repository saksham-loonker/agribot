package com.sakshyam.agribot.camera

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotSame
import org.junit.Assert.assertThrows
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder

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

    @Test fun `direct RGBA buffer preserves channels crop row padding and each rotation`() {
        // A non-square crop with nonzero top/left catches swapped coordinates and row padding.
        val bytes = ByteArray(24 * 4) { 0x7F }
        for (y in 0 until 4) for (x in 0 until 5) {
            val i = y * 24 + x * 4
            bytes[i] = (y * 5 + x).toByte()
            bytes[i + 1] = (128 + x).toByte()
            bytes[i + 2] = (250 - y).toByte()
            bytes[i + 3] = 0 // Camera alpha is intentionally ignored, as before.
        }
        val buffer = ByteBuffer.allocateDirect(bytes.size).put(bytes)
        val expectedRed = mapOf(
            0 to listOf(6, 7, 8, 11, 12, 13),
            90 to listOf(11, 6, 12, 7, 13, 8),
            180 to listOf(13, 12, 11, 8, 7, 6),
            270 to listOf(8, 13, 7, 12, 6, 11),
        )
        for ((rotation, reds) in expectedRed) {
            val u = RgbaFrameConverter.toUpright(buffer, 24, 1, 1, 3, 2, rotation)
            assertEquals(reds, r(u))
            val expectedArgb = reds.map { red ->
                (0xFF shl 24) or (red shl 16) or ((128 + red % 5) shl 8) or (250 - red / 5)
            }.toIntArray()
            assertArrayEquals(expectedArgb, u.argb)
            assertEquals(if (rotation == 90 || rotation == 270) 2 else 3, u.width)
            assertEquals(if (rotation == 90 || rotation == 270) 3 else 2, u.height)
        }
    }

    @Test fun `buffer conversion ignores cursor and preserves source cursor limit and byte order`() {
        val buffer = ByteBuffer.wrap(rgba).order(ByteOrder.LITTLE_ENDIAN)
        buffer.limit(28) // The final row need not include its trailing padding.
        buffer.position(23)
        buffer.mark()
        assertEquals(listOf(0, 1, 2, 3, 4, 5), r(RgbaFrameConverter.toUpright(buffer, stride, 0, 0, w, h, 0)))
        assertEquals(23, buffer.position())
        assertEquals(28, buffer.limit())
        assertEquals(ByteOrder.LITTLE_ENDIAN, buffer.order())
        buffer.reset() // Its mark is also retained.
        assertEquals(23, buffer.position())
    }

    @Test fun `read only sliced buffer treats its slice origin as frame origin`() {
        val backing = ByteBuffer.allocateDirect(rgba.size + 7).put(ByteArray(7)).put(rgba)
        backing.position(7)
        val slice = backing.slice().asReadOnlyBuffer()
        slice.position(slice.limit())
        assertEquals(listOf(2, 5, 1, 4, 0, 3), r(RgbaFrameConverter.toUpright(slice, stride, 0, 0, w, h, 270)))
        assertEquals(slice.limit(), slice.position())
    }

    @Test fun `each conversion owns pixels after source or next frame changes`() {
        val buffer = ByteBuffer.wrap(rgba.copyOf())
        val first = RgbaFrameConverter.toUpright(buffer, stride, 0, 0, w, h, 0)
        val expected = first.argb.copyOf()
        buffer.put(0, 99.toByte())
        val second = RgbaFrameConverter.toUpright(buffer, stride, 0, 0, w, h, 0)
        assertNotSame(first.argb, second.argb)
        assertArrayEquals(expected, first.argb)
        assertEquals(99, r(second).first())
    }

    @Test fun `invalid crop stride rotation or truncated buffer fails before reading`() {
        val buffer = ByteBuffer.wrap(rgba)
        fun convert(left: Int = 0, top: Int = 0, width: Int = w, height: Int = h,
                    rowStride: Int = stride, rotation: Int = 0) =
            RgbaFrameConverter.toUpright(buffer, rowStride, left, top, width, height, rotation)
        assertThrows(IllegalArgumentException::class.java) { convert(left = -1) }
        assertThrows(IllegalArgumentException::class.java) { convert(top = -1) }
        assertThrows(IllegalArgumentException::class.java) { convert(width = 0) }
        assertThrows(IllegalArgumentException::class.java) { convert(height = 0) }
        assertThrows(IllegalArgumentException::class.java) { convert(rowStride = 0) }
        assertThrows(IllegalArgumentException::class.java) { convert(left = 3, width = 2) }
        assertThrows(IllegalArgumentException::class.java) { convert(rotation = 45) }
        assertThrows(IllegalArgumentException::class.java) { convert(top = Int.MAX_VALUE) }
        assertThrows(IllegalArgumentException::class.java) { convert(width = Int.MAX_VALUE, height = Int.MAX_VALUE) }
        buffer.limit(27)
        assertThrows(IllegalArgumentException::class.java) { convert() }
    }
}
