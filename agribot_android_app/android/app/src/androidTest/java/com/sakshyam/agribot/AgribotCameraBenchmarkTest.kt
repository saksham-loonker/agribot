package com.sakshyam.agribot

import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.sakshyam.agribot.camera.RgbaFrameConverter
import java.nio.ByteBuffer
import org.junit.Assert.assertArrayEquals
import org.junit.Test
import org.junit.runner.RunWith

/** Isolates the unchanged 1280x960 RGBA/crop/rotation workload from model inference. */
@RunWith(AndroidJUnit4::class)
class AgribotCameraBenchmarkTest {
    @Test fun directConversionMatchesOriginalAndReportsLatency() {
        val width = 1280
        val height = 960
        val rgba = ByteArray(width * height * 4) { (it * 37).toByte() }
        val plane = ByteBuffer.allocateDirect(rgba.size).apply { put(rgba); rewind() }
        val staging = ByteArray(rgba.size)
        fun original(): IntArray {
            plane.duplicate().apply { rewind(); get(staging) }
            return originalRotation(staging, width, height)
        }
        fun optimized(): IntArray = RgbaFrameConverter.toUpright(plane, width * 4, 0, 0, width, height, 90).argb
        assertArrayEquals(original(), optimized())
        repeat(5) { original(); optimized() }
        val before = DoubleArray(20)
        val after = DoubleArray(20)
        var checksum = 0
        fun measure(operation: () -> IntArray): Double {
            val start = SystemClock.elapsedRealtimeNanos()
            val pixels = operation()
            val elapsed = (SystemClock.elapsedRealtimeNanos() - start) / 1e6
            checksum = checksum xor pixels[checksum and (pixels.size - 1)]
            return elapsed
        }
        for (i in before.indices) {
            // Alternate order so CPU frequency/GC effects do not consistently favour one path.
            if (i % 2 == 0) { before[i] = measure(::original); after[i] = measure(::optimized) }
            else { after[i] = measure(::optimized); before[i] = measure(::original) }
        }
        Log.i("AgribotCameraBench", "1280x960 rotation=90 originalAvg=${before.average()} ms optimizedAvg=${after.average()} ms samples=20 checksum=$checksum")
    }

    /** Original pixel conversion, retained only as benchmark/reference evidence. */
    private fun originalRotation(rgba: ByteArray, width: Int, height: Int): IntArray {
        val output = IntArray(width * height)
        for (y in 0 until height) for (x in 0 until width) {
            val i = (y * width + x) * 4
            val argb = (0xff shl 24) or ((rgba[i].toInt() and 0xff) shl 16) or
                ((rgba[i + 1].toInt() and 0xff) shl 8) or (rgba[i + 2].toInt() and 0xff)
            val (ox, oy) = (height - 1 - y) to x
            output[oy * height + ox] = argb
        }
        return output
    }
}
