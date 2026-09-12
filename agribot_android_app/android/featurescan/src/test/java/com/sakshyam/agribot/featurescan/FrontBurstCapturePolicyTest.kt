package com.sakshyam.agribot.featurescan

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class FrontBurstCapturePolicyTest {
    @Test
    fun rejectsCaptureBeforeMinimumBurstFrames() {
        val readiness = FrontBurstCapturePolicy.evaluate(frameCount = 4)

        assertFalse(readiness.canCapture)
        assertEquals("Collecting burst frames (4/7)", readiness.status)
        assertEquals("Wait for at least 5 Front Overview frames before capture", readiness.error)
    }

    @Test
    fun acceptsCaptureAtMinimumBurstFrames() {
        val readiness = FrontBurstCapturePolicy.evaluate(frameCount = 5)

        assertTrue(readiness.canCapture)
        assertEquals("Capture ready (5/7 frames)", readiness.status)
        assertNull(readiness.error)
    }

    @Test
    fun acceptsCaptureAtTargetBurstFrames() {
        val readiness = FrontBurstCapturePolicy.evaluate(frameCount = 7)

        assertTrue(readiness.canCapture)
        assertEquals("Capture ready (7/7 frames)", readiness.status)
        assertNull(readiness.error)
    }
}
