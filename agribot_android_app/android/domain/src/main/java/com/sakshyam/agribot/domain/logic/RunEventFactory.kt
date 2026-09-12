package com.sakshyam.agribot.domain.logic

import com.sakshyam.agribot.domain.model.RunEvent
import com.sakshyam.agribot.domain.model.RunId
import com.sakshyam.agribot.domain.model.DecisionId
import com.sakshyam.agribot.domain.model.GpsPathPoint
import com.sakshyam.agribot.domain.model.GpsPathStatus
import com.sakshyam.agribot.domain.model.PlantDecision
import java.time.Instant
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter
import java.util.UUID

object RunEventFactory {
    fun started(runId: RunId, timestamp: Instant, mode: String, targetFps: Int): RunEvent =
        event(
            runId = runId,
            timestamp = timestamp,
            type = "started",
            payloadJson = """{"mode":"${mode.escapeJson()}","target_fps":$targetFps}""",
        )

    fun paused(runId: RunId, timestamp: Instant, decisionsRecorded: Int): RunEvent =
        event(
            runId = runId,
            timestamp = timestamp,
            type = "paused",
            payloadJson = """{"decisions_recorded":$decisionsRecorded}""",
        )

    fun resumed(runId: RunId, timestamp: Instant, nextScanIndex: Int): RunEvent =
        event(
            runId = runId,
            timestamp = timestamp,
            type = "resumed",
            payloadJson = """{"next_scan_index":$nextScanIndex}""",
        )

    fun measurementReset(runId: RunId, timestamp: Instant, reason: String = "operator_requested"): RunEvent =
        event(
            runId = runId,
            timestamp = timestamp,
            type = "measurement_reset",
            payloadJson = """{"reason":"${reason.escapeJson()}"}""",
        )

    fun stopped(runId: RunId, timestamp: Instant, decisionsRecorded: Int): RunEvent =
        event(
            runId = runId,
            timestamp = timestamp,
            type = "stopped",
            payloadJson = """{"decisions_recorded":$decisionsRecorded}""",
        )

    /**
     * Persist a bounded, run-local path snapshot without storing raw GPS
     * coordinates. The event is immutable and can be exported for support or
     * replayed by a future map presenter.
     */
    fun gpsPathSnapshot(
        runId: RunId,
        timestamp: Instant,
        status: GpsPathStatus,
        provider: String?,
        points: List<GpsPathPoint>,
        snapshotLabel: String = "checkpoint",
    ): RunEvent {
        val safeLabel = snapshotLabel
            .trim()
            .filter { it.isLetterOrDigit() || it == '_' || it == '-' }
            .take(24)
            .ifBlank { "checkpoint" }
        val pointJson = points.joinToString(separator = ",") { point ->
            """{"sequence":${point.sequence},"east_m":${point.eastM},"north_m":${point.northM},"accuracy_m":${point.accuracyM ?: "null"},"elapsed_realtime_nanos":${point.elapsedRealtimeNanos}}"""
        }
        return RunEvent(
            id = uniqueId(runId, "gps_path_${safeLabel}_${timestamp.toEpochMilli()}_${points.lastOrNull()?.sequence ?: 0}"),
            runId = runId,
            timestamp = timestamp,
            type = "gps_path_snapshot",
            payloadJson = """{"status":"${status.name}","provider":${provider?.let { "\"${it.escapeJson()}\"" } ?: "null"},"point_count":${points.size},"points":[$pointJson]}""",
        )
    }

    fun retakeRequested(runId: RunId, timestamp: Instant, decisionId: DecisionId, scanIndex: Int): RunEvent =
        event(
            runId = runId,
            timestamp = timestamp,
            type = "retake_requested",
            payloadJson = """{"decision_id":"${decisionId.value.escapeJson()}","scan_index":$scanIndex,"reason":"operator_retake"}""",
        )

    fun exportCreated(runId: RunId, timestamp: Instant, exportKind: String, absolutePath: String): RunEvent =
        event(
            runId = runId,
            timestamp = timestamp,
            type = "export_created",
            payloadJson = """{"export_kind":"${exportKind.escapeJson()}","absolute_path":"${absolutePath.escapeJson()}"}""",
        )

    fun permissionLost(runId: RunId, timestamp: Instant, permission: String, previousState: String): RunEvent =
        event(
            runId = runId,
            timestamp = timestamp,
            type = "permission_lost",
            payloadJson = """{"permission":"${permission.escapeJson()}","previous_state":"${previousState.escapeJson()}"}""",
        )

    fun modelLoaded(runId: RunId, timestamp: Instant, bundleId: String, cpuDefault: Boolean): RunEvent =
        event(
            runId = runId,
            timestamp = timestamp,
            type = "model_loaded",
            payloadJson = """{"bundle_id":"${bundleId.escapeJson()}","cpu_default":$cpuDefault}""",
        )

    fun thermalWarning(
        runId: RunId,
        timestamp: Instant,
        thermalStatus: String,
        action: String,
        targetFps: Int,
    ): RunEvent =
        event(
            runId = runId,
            timestamp = timestamp,
            type = "thermal_warning",
            payloadJson = """{"thermal_status":"${thermalStatus.escapeJson()}","action":"${action.escapeJson()}","target_fps":$targetFps}""",
        )

    fun manualSnapshot(
        runId: RunId,
        timestamp: Instant,
        evidencePath: String?,
        evidenceStatus: String,
        operatorDisplayName: String? = null,
    ): RunEvent =
        event(
            runId = runId,
            timestamp = timestamp,
            type = "manual_snapshot",
            payloadJson = """{"evidence_status":"${evidenceStatus.escapeJson()}","evidence_path":${evidencePath?.let { "\"${it.escapeJson()}\"" } ?: "null"},"operator":"${operatorDisplayName.auditOperator().escapeJson()}"}""",
        )

    fun manualOverride(
        runId: RunId,
        timestamp: Instant,
        decisionId: DecisionId,
        original: PlantDecision?,
        corrected: PlantDecision,
        reason: String?,
        operatorDisplayName: String?,
    ): RunEvent =
        RunEvent(
            id = uniqueId(runId, "manual_override_${decisionId.value}"),
            runId = runId,
            timestamp = timestamp,
            type = "manual_override",
            payloadJson = buildString {
                append("""{"decision_id":"${decisionId.value.escapeJson()}","operator":"${operatorDisplayName.auditOperator().escapeJson()}","reason":""")
                append(reason?.let { "\"${it.escapeJson()}\"" } ?: "null")
                append(""","original":""")
                append(original?.toManualAuditJson() ?: "null")
                append(""","updated":""")
                append(corrected.toManualAuditJson())
                append("}")
            },
        )

    private fun event(runId: RunId, timestamp: Instant, type: String, payloadJson: String): RunEvent =
        RunEvent(
            id = uniqueId(runId, "${type}_${EVENT_ID_TIME_FORMAT.format(timestamp)}"),
            runId = runId,
            timestamp = timestamp,
            type = type,
            payloadJson = payloadJson,
        )

    private val EVENT_ID_TIME_FORMAT: DateTimeFormatter =
        DateTimeFormatter.ofPattern("yyyyMMdd'T'HHmmss'Z'").withZone(ZoneOffset.UTC)

    private fun uniqueId(runId: RunId, descriptor: String): String =
        "${runId.value}_${descriptor}_${UUID.randomUUID().toString().replace("-", "").take(12)}"
}

private fun PlantDecision.toManualAuditJson(): String =
    """{"label":"${label.escapeJson()}","status":"${status.name.lowercase()}","confidence":$confidence}"""

private fun String?.auditOperator(): String =
    this
        ?.trim()
        ?.filterNot { it.isISOControl() }
        ?.replace(Regex("\\s+"), " ")
        ?.take(64)
        ?.takeIf { it.isNotBlank() }
        ?: "local_user"

private fun String.escapeJson(): String =
    buildString {
        this@escapeJson.forEach { char ->
            when (char) {
                '\\' -> append("\\\\")
                '"' -> append("\\\"")
                '\n' -> append("\\n")
                '\r' -> append("\\r")
                '\t' -> append("\\t")
                else -> append(char)
            }
        }
    }
