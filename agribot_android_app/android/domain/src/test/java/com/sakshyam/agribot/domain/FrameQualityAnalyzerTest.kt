package com.sakshyam.agribot.domain

import com.sakshyam.agribot.domain.logic.FrameQualityAnalyzer
import com.sakshyam.agribot.domain.logic.FrameQualityStatus
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class FrameQualityAnalyzerTest {
    @Test
    fun darkFrameRequestsMoreLight() {
        val result = FrameQualityAnalyzer.evaluate(16, 16, IntArray(16 * 16) { 0xff080808.toInt() })

        assertEquals(FrameQualityStatus.RETRY, result.status)
        assertTrue(result.message.contains("dark", ignoreCase = true))
    }

    @Test
    fun invalidFrameCannotReachInference() {
        val result = FrameQualityAnalyzer.evaluate(16, 16, IntArray(3))

        assertEquals(FrameQualityStatus.RETRY, result.status)
        assertTrue(result.message.contains("invalid", ignoreCase = true))
    }

    @Test
    fun detailedFrameIsAccepted() {
        val pixels = IntArray(64 * 64) { index ->
            val x = index % 64
            val y = index / 64
            if ((x / 4 + y / 4) % 2 == 0) 0xff45a85a.toInt() else 0xff382d22.toInt()
        }

        val result = FrameQualityAnalyzer.evaluate(64, 64, pixels)

        assertEquals(FrameQualityStatus.ACCEPT, result.status)
        assertTrue(result.sharpness > 0.012f)
    }
}
