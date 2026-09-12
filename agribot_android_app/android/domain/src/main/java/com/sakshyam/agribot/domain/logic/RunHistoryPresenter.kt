package com.sakshyam.agribot.domain.logic

import com.sakshyam.agribot.domain.model.RecordingMode
import com.sakshyam.agribot.domain.model.Run
import com.sakshyam.agribot.domain.model.RunHistoryItem
import com.sakshyam.agribot.domain.model.RunSummary
import java.util.Locale

object RunHistoryPresenter {
    fun toItem(run: Run, summary: RunSummary, fieldName: String): RunHistoryItem =
        RunHistoryItem(
            runId = run.runId,
            startedAt = run.startedAt,
            fieldName = fieldName,
            modeLabel = modeLabel(run.mode),
            stateLabel = run.state.name.lowercase().replaceFirstChar { it.titlecase(Locale.US) },
            sickCount = summary.sick,
            uncertainRateLabel = "%.1f%%".format(Locale.US, summary.uncertainRate * 100.0),
            decisionCountLabel = "${summary.decisions} ${if (summary.decisions == 1) "decision" else "decisions"}",
        )

    private fun modeLabel(mode: RecordingMode): String = when (mode) {
        RecordingMode.SIDE_SCAN -> "Side Scan"
        RecordingMode.FRONT_ROW_OVERVIEW -> "Front Overview"
    }
}
