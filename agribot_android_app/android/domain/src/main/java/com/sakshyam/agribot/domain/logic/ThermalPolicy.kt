package com.sakshyam.agribot.domain.logic

import com.sakshyam.agribot.domain.model.PerformanceProfile
import com.sakshyam.agribot.domain.model.ThermalPolicyResult
import com.sakshyam.agribot.domain.model.ThermalStatus

object ThermalPolicy {
    private const val PREFERRED_LATE_MS = 180.0
    private const val HARD_LATE_MS = 200.0
    private const val AVG_LATE_MS = 160.0

    fun evaluate(
        current: PerformanceProfile,
        selectedFrameLatenciesMs: List<Double>,
        thermalStatus: ThermalStatus,
    ): ThermalPolicyResult {
        if (thermalStatus == ThermalStatus.SEVERE || thermalStatus == ThermalStatus.CRITICAL) {
            return ThermalPolicyResult(
                profile = current.copy(targetFps = minOf(current.targetFps, 5)),
                pauseAfterCurrentGroup = true,
                reason = "thermal_${thermalStatus.name.lowercase()}",
            )
        }

        if (thermalStatus == ThermalStatus.MODERATE) {
            return ThermalPolicyResult(
                profile = current.copy(targetFps = minOf(current.targetFps, 5)),
                pauseAfterCurrentGroup = false,
                reason = "thermal_moderate_low_power",
            )
        }

        val window = selectedFrameLatenciesMs.takeLast(30)
        val hardLate = window.count { it > HARD_LATE_MS }
        if (hardLate >= 5) {
            return ThermalPolicyResult(
                profile = current.copy(targetFps = reduceFps(current.targetFps)),
                pauseAfterCurrentGroup = false,
                reason = "hard_late_frames_$hardLate",
            )
        }

        val avg = if (window.isEmpty()) 0.0 else window.average()
        if (window.size >= 30 && avg > AVG_LATE_MS) {
            return ThermalPolicyResult(
                profile = current.copy(targetFps = reduceFps(current.targetFps)),
                pauseAfterCurrentGroup = false,
                reason = "rolling_avg_latency_${avg.toInt()}ms",
            )
        }

        val preferredLate = window.count { it > PREFERRED_LATE_MS }
        return ThermalPolicyResult(
            profile = current,
            pauseAfterCurrentGroup = false,
            reason = if (preferredLate > 0) "preferred_late_frames_$preferredLate" else "within_policy",
        )
    }

    private fun reduceFps(targetFps: Int): Int = when {
        targetFps > 5 -> 5
        targetFps > 3 -> 3
        else -> targetFps
    }
}
