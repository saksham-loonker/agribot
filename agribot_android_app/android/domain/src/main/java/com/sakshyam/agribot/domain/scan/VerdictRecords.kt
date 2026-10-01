package com.sakshyam.agribot.domain.scan

import com.sakshyam.agribot.domain.model.DecisionId
import com.sakshyam.agribot.domain.model.DecisionStatus
import com.sakshyam.agribot.domain.model.PlantHealthAction
import com.sakshyam.agribot.domain.model.RecordedDecision
import com.sakshyam.agribot.domain.model.RecordingMode
import com.sakshyam.agribot.domain.model.RunId
import java.time.Instant

/** Where a plant verdict was made. */
data class PlantPlace(val fieldId: String, val rowIndex: Int?, val plantNumber: Int?)

/** Maps a [PlantVerdict] to the persisted decision, so storage semantics stay in one tested place. */
object VerdictRecords {
    fun toDecision(
        id: DecisionId,
        runId: RunId,
        sequence: Int,
        mode: RecordingMode,
        place: PlantPlace,
        verdict: PlantVerdict,
        labels: List<String>,
        modelVersion: String,
        now: Instant,
        evidencePath: String? = null,
        latitude: Double? = null,
        longitude: Double? = null,
        gpsAccuracyM: Double? = null,
    ): RecordedDecision {
        val label = verdict.labelIndex?.let { labels.getOrNull(it) }.orEmpty()
        val (status, action) = when (verdict.kind) {
            VerdictKind.HEALTHY -> DecisionStatus.OK to PlantHealthAction.NONE
            VerdictKind.DISEASE, VerdictKind.UNKNOWN_CONDITION -> DecisionStatus.OK to PlantHealthAction.INSPECT_OR_TREAT
            VerdictKind.UNSURE -> DecisionStatus.UNCERTAIN to PlantHealthAction.RESCAN
            VerdictKind.NEED_MORE_VIEWS ->
                if (verdict.leavesSeen == 0) DecisionStatus.SKIPPED to PlantHealthAction.RESCAN else DecisionStatus.UNCERTAIN to PlantHealthAction.RESCAN
        }
        val reason = when (verdict.kind) {
            VerdictKind.NEED_MORE_VIEWS -> if (verdict.leavesSeen == 0) "not_seen" else "too_few_views"
            else -> verdict.kind.name.lowercase()
        }
        return RecordedDecision(
            id = id, runId = runId, sequence = sequence, timestamp = now, epochTime = now.toEpochMilli() / 1000.0,
            mode = mode, fieldId = place.fieldId,
            rowId = place.rowIndex?.toString(), rowSide = null, rowIndex = place.rowIndex, plantColumn = place.plantNumber,
            plantNumber = place.plantNumber,
            plantKey = if (place.rowIndex != null && place.plantNumber != null) "${place.rowIndex}-${place.plantNumber}" else null,
            xM = longitude, yM = latitude, bboxPx = null,
            geometryConfidence = null, plantPositionConfidence = null, geometryReason = null,
            label = if (status == DecisionStatus.SKIPPED) "" else label,
            confidence = if (status == DecisionStatus.SKIPPED) 0f else verdict.confidence,
            rawLabel = label, status = status, action = action,
            framesUsed = verdict.framesUsed, reason = reason, lateFrames = 0, gateAvgLatencyMs = 0.0, gateMaxLatencyMs = 0.0,
            modelVersion = modelVersion, scanIndex = sequence, scanPass = null, fieldComplete = null, plannedTotalPlants = null,
            plantDisplay = null, plantId = place.plantNumber, threads = null, fpsTarget = null,
            evidencePath = evidencePath, gpsAccuracyM = gpsAccuracyM,
            leavesSeen = verdict.leavesSeen, leavesAgreeing = verdict.leavesAgreeing, partialFinding = verdict.partial,
            runnerUpLabel = verdict.runnerUpIndex?.let { labels.getOrNull(it) }, runnerUpConfidence = verdict.runnerUpConfidence,
        )
    }
}

/** Rebuilds a displayable verdict from a stored decision. Index 0 = decision label, 1 = runner-up. */
fun RecordedDecision.toVerdict(): Pair<PlantVerdict, List<String>> {
    val labels = listOf(label, runnerUpLabel.orEmpty())
    val kind = when {
        status == com.sakshyam.agribot.domain.model.DecisionStatus.SKIPPED -> VerdictKind.NEED_MORE_VIEWS
        status == com.sakshyam.agribot.domain.model.DecisionStatus.UNCERTAIN -> VerdictKind.UNSURE
        action == PlantHealthAction.NONE -> VerdictKind.HEALTHY
        label == "Other" -> VerdictKind.UNKNOWN_CONDITION
        else -> VerdictKind.DISEASE
    }
    return PlantVerdict(
        kind = kind, labelIndex = if (label.isEmpty()) null else 0, confidence = confidence,
        runnerUpIndex = if (runnerUpLabel.isNullOrEmpty()) null else 1, runnerUpConfidence = runnerUpConfidence ?: 0f,
        leavesSeen = leavesSeen ?: 0, leavesAgreeing = leavesAgreeing ?: 0, framesUsed = framesUsed, partial = partialFinding == true,
    ) to labels
}
