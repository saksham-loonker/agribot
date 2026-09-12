package com.sakshyam.agribot.domain

import com.sakshyam.agribot.domain.logic.ThermalPolicy
import com.sakshyam.agribot.domain.model.PerformanceProfile
import com.sakshyam.agribot.domain.model.ThermalStatus
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ThermalPolicyTest {
    @Test
    fun reducesFpsWhenHardLateWindowExceedsPolicy() {
        val profile = PerformanceProfile(targetFps = 10, resolutionStep = 0)
        val frameStats = List(25) { 90.0 } + List(5) { 205.0 }

        val result = ThermalPolicy.evaluate(profile, frameStats, ThermalStatus.NONE)

        assertEquals(5, result.profile.targetFps)
        assertTrue(result.reason.contains("hard_late"))
    }

    @Test
    fun severeThermalPausesAfterCurrentDecisionGroup() {
        val profile = PerformanceProfile(targetFps = 5, resolutionStep = 0)

        val result = ThermalPolicy.evaluate(profile, List(30) { 80.0 }, ThermalStatus.SEVERE)

        assertEquals(true, result.pauseAfterCurrentGroup)
        assertEquals(5, result.profile.targetFps)
    }
}
