package com.sakshyam.agribot.domain.scan

import com.sakshyam.agribot.domain.logic.FrameQualityStatus
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test

class CheckedLeafAnalysisTest {
    private class FakeVision : LeafVision {
        override val status = MutableStateFlow<VisionStatus>(VisionStatus.Loading)
        var calls = 0
        override fun warmUp() = Unit
        override suspend fun analyze(frame: VisionFrame, maxLeaves: Int): FrameAnalysis {
            calls++
            return FrameAnalysis(frame.width, frame.height, emptyList(), 0, 0.0, 0.0)
        }
    }

    @Test fun `dark glare and detail-free views do not reach the classifier`() = runBlocking {
        val vision = FakeVision()
        for (pixel in listOf(0xff000000.toInt(), 0xffffffff.toInt(), 0xff808080.toInt())) {
            val result = vision.analyzeUsable(VisionFrame(32, 32, IntArray(1024) { pixel }, 1L), 6)
            assertEquals(FrameQualityStatus.RETRY, result.quality.status)
            assertNull(result.analysis)
        }
        assertEquals(0, vision.calls)
    }

    @Test fun `a usable view reaches inference once with its assessment`() = runBlocking {
        val vision = FakeVision()
        val pixels = IntArray(1024) { if ((it % 32 + it / 32) % 2 == 0) 0xff205020.toInt() else 0xffa0c0a0.toInt() }
        val result = vision.analyzeUsable(VisionFrame(32, 32, pixels, 1L), 6)
        assertEquals(FrameQualityStatus.ACCEPT, result.quality.status)
        assertNotNull(result.analysis)
        assertEquals(1, vision.calls)
    }

    @Test fun `a dim view with useful structure is not discarded by the warning threshold`() = runBlocking {
        val vision = FakeVision()
        val pixels = IntArray(1024) { if ((it % 32 + it / 32) % 2 == 0) 0xff000000.toInt() else 0xff202020.toInt() }
        val result = vision.analyzeUsable(VisionFrame(32, 32, pixels, 1L), 6)
        assertEquals(FrameQualityStatus.RETRY, result.quality.status)
        assertNotNull(result.analysis)
        assertEquals(1, vision.calls)
    }
}
