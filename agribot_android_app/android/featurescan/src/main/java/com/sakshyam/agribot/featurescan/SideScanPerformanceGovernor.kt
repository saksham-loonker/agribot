package com.sakshyam.agribot.featurescan

import android.content.Context
import android.os.Build
import android.os.PowerManager
import com.sakshyam.agribot.domain.logic.ThermalPolicy
import com.sakshyam.agribot.domain.model.PerformanceProfile
import com.sakshyam.agribot.domain.model.ThermalStatus

class SideScanPerformanceGovernor(initialTargetFps: Int = SideScanDefaults.TARGET_FPS) {
    private var profile = PerformanceProfile(targetFps = initialTargetFps, resolutionStep = 0)
    private val rollingLatenciesMs = ArrayDeque<Double>()
    private val decisionGroupLatenciesMs = ArrayDeque<Double>()

    fun reset(targetFps: Int = SideScanDefaults.TARGET_FPS) {
        profile = PerformanceProfile(targetFps = targetFps, resolutionStep = 0)
        rollingLatenciesMs.clear()
        decisionGroupLatenciesMs.clear()
    }

    fun recordSelectedFrameLatency(latencyMs: Double, thermalStatus: ThermalStatus): SideScanPerformanceDecision {
        rollingLatenciesMs.addLast(latencyMs)
        while (rollingLatenciesMs.size > LATENCY_WINDOW_SIZE) {
            rollingLatenciesMs.removeFirst()
        }
        decisionGroupLatenciesMs.addLast(latencyMs)

        val result = ThermalPolicy.evaluate(
            current = profile,
            selectedFrameLatenciesMs = rollingLatenciesMs.toList(),
            thermalStatus = thermalStatus,
        )
        profile = result.profile
        return SideScanPerformanceDecision(
            targetFps = result.profile.targetFps,
            performanceStatus = statusText(result.profile.targetFps, result.reason),
            pauseAfterCurrentGroup = result.pauseAfterCurrentGroup,
            reason = result.reason,
        )
    }

    fun lateFramesInCurrentDecisionGroup(): Int =
        decisionGroupLatenciesMs.count { it > PREFERRED_LATE_MS }

    fun resetDecisionGroup() {
        decisionGroupLatenciesMs.clear()
    }

    private fun statusText(targetFps: Int, reason: String): String =
        when {
            reason == "within_policy" -> SideScanDefaults.performanceStatus(targetFps)
            reason.startsWith("preferred_late") -> "${SideScanDefaults.performanceStatus(targetFps)} - warning: $reason"
            else -> "${SideScanDefaults.performanceStatus(targetFps)} - reduced: $reason"
        }

    private companion object {
        const val LATENCY_WINDOW_SIZE = 30
        const val PREFERRED_LATE_MS = 180.0
    }
}

data class SideScanPerformanceDecision(
    val targetFps: Int,
    val performanceStatus: String,
    val pauseAfterCurrentGroup: Boolean,
    val reason: String,
)

object AndroidThermalStatusReader {
    fun read(context: Context): ThermalStatus {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return ThermalStatus.NONE
        val powerManager = context.getSystemService(PowerManager::class.java) ?: return ThermalStatus.NONE
        return when (powerManager.currentThermalStatus) {
            PowerManager.THERMAL_STATUS_LIGHT -> ThermalStatus.LIGHT
            PowerManager.THERMAL_STATUS_MODERATE -> ThermalStatus.MODERATE
            PowerManager.THERMAL_STATUS_SEVERE -> ThermalStatus.SEVERE
            PowerManager.THERMAL_STATUS_CRITICAL,
            PowerManager.THERMAL_STATUS_EMERGENCY,
            PowerManager.THERMAL_STATUS_SHUTDOWN,
            -> ThermalStatus.CRITICAL
            else -> ThermalStatus.NONE
        }
    }
}
