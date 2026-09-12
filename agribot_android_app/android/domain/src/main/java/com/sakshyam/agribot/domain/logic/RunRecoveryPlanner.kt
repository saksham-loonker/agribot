package com.sakshyam.agribot.domain.logic

import com.sakshyam.agribot.domain.model.RecordedDecision
import com.sakshyam.agribot.domain.model.Run
import com.sakshyam.agribot.domain.model.RunState

object RunRecoveryPlanner {
    fun plan(run: Run?, decisions: List<RecordedDecision>): RunRecoveryPlan {
        if (run == null || run.state !in RECOVERABLE_STATES) {
            return RunRecoveryPlan(
                shouldRecover = false,
                shouldMarkPaused = false,
                recoveredState = null,
                nextScanIndex = 1,
                decisionCount = decisions.size,
                message = null,
            )
        }
        val nextScanIndex = decisions.maxOfOrNull { it.scanIndex }?.plus(1) ?: 1
        return RunRecoveryPlan(
            shouldRecover = true,
            shouldMarkPaused = run.state == RunState.RECORDING,
            recoveredState = RunState.PAUSED,
            nextScanIndex = nextScanIndex,
            decisionCount = decisions.size,
            message = "Recovered interrupted run ${run.runId.value}",
        )
    }

    private val RECOVERABLE_STATES = setOf(RunState.RECORDING, RunState.PAUSED)
}

data class RunRecoveryPlan(
    val shouldRecover: Boolean,
    val shouldMarkPaused: Boolean,
    val recoveredState: RunState?,
    val nextScanIndex: Int,
    val decisionCount: Int,
    val message: String?,
)
