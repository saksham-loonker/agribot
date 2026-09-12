package com.sakshyam.agribot.featurescan

import com.sakshyam.agribot.domain.model.ThermalStatus
import kotlin.test.Test
import kotlin.test.assertEquals

class SideScanPerformanceGovernorTest {
    @Test
    fun reducesTargetFpsWhenRollingWindowHasTooManyHardLateFrames() {
        val governor = SideScanPerformanceGovernor(initialTargetFps = 10)

        repeat(25) { governor.recordSelectedFrameLatency(90.0, ThermalStatus.NONE) }
        val decision = (1..5).map {
            governor.recordSelectedFrameLatency(205.0, ThermalStatus.NONE)
        }.last()

        assertEquals(5, decision.targetFps)
        assertEquals("5 FPS - reduced: hard_late_frames_5", decision.performanceStatus)
        assertEquals(false, decision.pauseAfterCurrentGroup)
    }

    @Test
    fun severeThermalStatusRequestsPauseAfterCurrentGroup() {
        val governor = SideScanPerformanceGovernor(initialTargetFps = 5)

        val decision = governor.recordSelectedFrameLatency(80.0, ThermalStatus.SEVERE)

        assertEquals(5, decision.targetFps)
        assertEquals("5 FPS - reduced: thermal_severe", decision.performanceStatus)
        assertEquals(true, decision.pauseAfterCurrentGroup)
    }

    @Test
    fun countsPreferredLateFramesForCurrentDecisionGroup() {
        val governor = SideScanPerformanceGovernor(initialTargetFps = 5)

        governor.recordSelectedFrameLatency(181.0, ThermalStatus.NONE)
        governor.recordSelectedFrameLatency(90.0, ThermalStatus.NONE)
        governor.recordSelectedFrameLatency(201.0, ThermalStatus.NONE)

        assertEquals(2, governor.lateFramesInCurrentDecisionGroup())

        governor.resetDecisionGroup()

        assertEquals(0, governor.lateFramesInCurrentDecisionGroup())
    }
}
