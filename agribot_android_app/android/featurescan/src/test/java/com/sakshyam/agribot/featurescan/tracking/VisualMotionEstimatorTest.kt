package com.sakshyam.agribot.featurescan.tracking

import com.sakshyam.agribot.domain.model.AnalysisFrame
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class VisualMotionEstimatorTest {
    private val texture = Random(42).let { random -> IntArray(96 * 72) { random.nextInt(256) } }

    private fun frame(time: Long, dx: Int = 0, dy: Int = 0, blank: Boolean = false): AnalysisFrame {
        val pixels = IntArray(96 * 72) { index ->
            val x = index % 96 - dx
            val y = index / 96 - dy
            val v = if (blank || x !in 0..95 || y !in 0..71) 100 else texture[y * 96 + x]
            (255 shl 24) or (v shl 16) or (v shl 8) or v
        }
        return AnalysisFrame(96, 72, 0, time, pixels, byteArrayOf())
    }

    @Test
    fun translatesInBothDirectionsWithoutTurningPixelsIntoMetres() {
        for (dx in listOf(-3, 3)) {
            val estimator = VisualMotionEstimator()
            assertFalse(estimator.observe(frame(1), 0f).accepted)
            val motion = estimator.observe(frame(200_000_001, dx, 2), 0f)
            assertTrue(motion.accepted)
            assertEquals(-dx / 96f, motion.cameraRight, 0.001f)
            assertEquals(-2 / 72f, motion.cameraDown, 0.001f)
        }
    }

    @Test
    fun rejectsRotationMissingAttitudeFrameGapsAndBlankScenes() {
        for ((rotation, time) in listOf(0.1f to 200_000_001L, null to 200_000_001L, 0f to 900_000_001L)) {
            val estimator = VisualMotionEstimator()
            estimator.observe(frame(1), 0f)
            assertFalse(estimator.observe(frame(time, 3), rotation).accepted)
        }
        val estimator = VisualMotionEstimator()
        estimator.observe(frame(1, blank = true), 0f)
        assertFalse(estimator.observe(frame(200_000_001, blank = true), 0f).accepted)
    }

    @Test
    fun stationaryTextureIsAcceptedAsZeroAndResetDropsOldFrame() {
        val estimator = VisualMotionEstimator()
        estimator.observe(frame(1), 0f)
        val motion = estimator.observe(frame(200_000_001), 0f)
        assertTrue(motion.accepted)
        assertEquals(0f, motion.cameraRight)
        estimator.reset()
        assertFalse(estimator.observe(frame(400_000_001), 0f).accepted)
    }
}
