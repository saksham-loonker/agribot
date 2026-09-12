package com.sakshyam.agribot.domain.logic

import com.sakshyam.agribot.domain.model.BoundingBox
import com.sakshyam.agribot.domain.model.DecisionStatus
import com.sakshyam.agribot.domain.model.DiagnosticsSnapshot
import com.sakshyam.agribot.domain.model.FieldLayout
import com.sakshyam.agribot.domain.model.PlantHealthAction
import com.sakshyam.agribot.domain.model.RecordedDecision
import com.sakshyam.agribot.domain.model.RecordingMode
import com.sakshyam.agribot.domain.model.RowSide
import com.sakshyam.agribot.domain.model.Run
import com.sakshyam.agribot.domain.model.RunEvent
import com.sakshyam.agribot.domain.model.RunState
import com.sakshyam.agribot.domain.model.RunSummary
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonObjectBuilder
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

object ExportSerializer {
    private val json = Json

    val csvHeader = listOf(
        "sequence",
        "timestamp",
        "epoch_time",
        "field_id",
        "row_id",
        "row_index",
        "plant_column",
        "plant_number",
        "plant_key",
        "x_m",
        "y_m",
        "plant_display",
        "scan_index",
        "plant_id",
        "track_id",
        "label",
        "confidence",
        "status",
        "action",
        "frames_used",
        "reason",
        "late_frames",
        "gate_avg_latency_ms",
        "gate_max_latency_ms",
        "threads",
        "fps_target",
        "row_side",
        "bbox_px",
        "geometry_confidence",
        "plant_position_confidence",
        "geometry_reason",
        "raw_label",
        "manual_override",
        "evidence_path",
        "evidence_status",
        "measurement_distance_m",
        "measurement_source",
        "measurement_quality",
        "relative_plant_width",
        "relative_plant_height",
        "size_source",
        "size_quality",
        "gps_accuracy_m",
        "gps_fix_age_seconds",
        "top2_margin",
        "prediction_entropy",
        "treatment_status",
        "treatment_note",
    )

    fun toJsonl(decisions: List<RecordedDecision>): String =
        decisions.joinToString("\n") { json.encodeToString(decisionJson(it)) }

    fun toRunEventsJsonl(events: List<RunEvent>): String {
        if (events.isEmpty()) return ""
        return events.joinToString(separator = "\n", postfix = "\n") { event ->
            json.encodeToString(runEventJson(event))
        }
    }

    fun toEventsCsv(decisions: List<RecordedDecision>): String {
        val rows = decisions.map { decision ->
            listOf(
                decision.sequence,
                decision.timestamp.toString(),
                decision.epochTime,
                decision.fieldId,
                decision.rowId,
                decision.rowIndex,
                decision.plantColumn,
                decision.plantNumber,
                decision.plantKey,
                decision.xM,
                decision.yM,
                decision.plantDisplay,
                decision.scanIndex,
                decision.plantId,
                decision.trackId,
                decision.label,
                decision.confidence,
                statusValue(decision.status),
                actionValue(decision.action),
                decision.framesUsed,
                decision.reason,
                decision.lateFrames,
                decision.gateAvgLatencyMs,
                decision.gateMaxLatencyMs,
                decision.threads,
                decision.fpsTarget,
                decision.rowSide?.jsonValue(),
                decision.bboxPx?.toCsvValue(),
                decision.geometryConfidence,
                decision.plantPositionConfidence,
                decision.geometryReason,
                decision.rawLabel,
                decision.manualOverride,
                portableEvidencePath(decision.evidencePath),
                decision.evidenceStatus,
                decision.measurementDistanceM,
                decision.measurementSource,
                decision.measurementQuality,
                decision.relativePlantWidth,
                decision.relativePlantHeight,
                decision.sizeSource,
                decision.sizeQuality,
                decision.gpsAccuracyM,
                decision.gpsFixAgeSeconds,
                decision.top2Margin,
                decision.predictionEntropy,
                decision.treatmentStatus.name,
                decision.treatmentNote,
            ).joinToString(",") { csvEscape(it) }
        }
        return (listOf(csvHeader.joinToString(",")) + rows).joinToString("\n")
    }

    fun toMetadataJson(run: Run, layout: FieldLayout? = null): String = json.encodeToString(
        buildJsonObject {
            put("run_id", run.runId.value)
            put("origin", "android")
            put("mode", run.mode.name)
            put("field_layout_id", run.fieldLayoutId.value)
            put("started_at", run.startedAt.toString())
            putNullable("completed_at", run.completedAt?.toString())
            put("state", run.state.name.lowercase())
            put("target_fps", run.targetFps)
            put("model_bundle_id", run.modelBundleId)
            put("paths", pathsJson())
            if (layout != null) {
                put(
                    "field_map",
                    buildJsonObject {
                        put("field_id", layout.id.value)
                        put("active_field_id", layout.id.value)
                        put("row_id", layout.activeRowId.value)
                        put("active_row_id", layout.activeRowId.value)
                        put("start_plant", layout.startPlant)
                        put("plant_step", layout.plantStep)
                        put("plant_cooldown_sec", layout.plantCooldownSec)
                        put("mapping_method", "Sequential row/column scan with Android local inference.")
                    },
                )
                put("farm_layout", fieldLayoutJson(layout))
            }
        },
    ) + "\n"

    fun toSummaryJson(summary: RunSummary, run: Run? = null): String = json.encodeToString(
        buildJsonObject {
            put("run_id", summary.runId.value)
            run?.let {
                put("started_at", it.startedAt.toString())
                putNullable("last_updated_at", it.completedAt?.toString() ?: it.startedAt.toString())
                put("completed", it.state == RunState.COMPLETED)
            }
            put("decisions", summary.decisions)
            put("ok", summary.ok)
            put("sick", summary.sick)
            put("uncertain", summary.uncertain)
            put("total_decisions", summary.decisions)
            put("ok_count", summary.ok)
            put("sick_count", summary.sick)
            put("uncertain_count", summary.uncertain)
            put("uncertain_rate", summary.uncertainRate)
            put("avg_confidence", summary.avgConfidence)
            put("max_confidence", summary.maxConfidence)
            put("total_late_frames", summary.totalLateFrames)
            put("labels", JsonObject(summary.labels.mapValues { JsonPrimitive(it.value) }))
            put("statuses", JsonObject(summary.statuses.mapKeys { it.key.name.lowercase() }.mapValues { JsonPrimitive(it.value) }))
            summary.latestDecision?.let { put("latest_decision", decisionJson(it)) }
            put("paths", pathsJson())
        },
    ) + "\n"

    fun toLatestRunJson(run: Run, summary: RunSummary): String = json.encodeToString(
        buildJsonObject {
            put("latest_run_id", run.runId.value)
            put("run_id", run.runId.value)
            put("updated_at", run.completedAt?.toString() ?: run.startedAt.toString())
            put("completed", run.state == RunState.COMPLETED)
            put("run_dir", ".")
            put("summary", "summary.json")
            put("mode", run.mode.name)
            put("state", run.state.name.lowercase())
            put("started_at", run.startedAt.toString())
            putNullable("completed_at", run.completedAt?.toString())
            put("decision_count", summary.decisions)
            put("sick_count", summary.sick)
            put("uncertain_count", summary.uncertain)
            put("model_bundle_id", run.modelBundleId)
        },
    ) + "\n"

    fun toFieldLayoutJson(layout: FieldLayout): String = json.encodeToString(
        fieldLayoutJson(layout),
    ) + "\n"

    fun toDiagnosticsJson(snapshot: DiagnosticsSnapshot?): String = json.encodeToString(
        buildJsonObject {
            put("offline", true)
            put("network_required", false)
            if (snapshot == null) {
                put("status", "not_captured")
            } else {
                put("status", "captured")
                put("app_version", snapshot.appVersion)
                put("model_bundle_id", snapshot.modelBundleId)
                put("camera_id", snapshot.cameraId)
                put("supported_sizes", snapshot.supportedSizes)
                put("analysis_resolution", snapshot.analysisResolution)
                put("cpu_thread_profile", snapshot.cpuThreadProfile)
                put("latest_benchmark_result", snapshot.latestBenchmarkResult)
                put("readiness_summary", snapshot.readinessSummary)
                put("thermal_battery_warnings", snapshot.thermalBatteryWarnings)
                put("storage_use", snapshot.storageUse)
                put("permission_status", snapshot.permissionStatus)
                put("run_state", snapshot.runState)
                putNullable("last_export_path", snapshot.lastExportPath)
                put("measurement_source", snapshot.measurementSource)
                put("measurement_quality", snapshot.measurementQuality)
                put("gps_status", snapshot.gpsStatus)
                putNullable("gps_provider", snapshot.gpsProvider)
                put("gps_path_status", snapshot.gpsPathStatus.name)
                put("gps_path_point_count", snapshot.gpsPathPointCount)
                put("motion_events_observed", snapshot.motionEventsObserved)
                put("capture_quality_status", snapshot.captureQualityStatus)
                put("detection_status", snapshot.detectionStatus)
                put("step_length_m", snapshot.stepLengthM)
                put("step_calibration_status", snapshot.stepCalibrationStatus)
                putNullable("latest_decision_reason", snapshot.latestDecisionReason)
            }
        },
    ) + "\n"

    private fun fieldLayoutJson(layout: FieldLayout): JsonObject = buildJsonObject {
        put("active_field_id", layout.id.value)
        put("field_id", layout.name)
        put("active_row_id", layout.activeRowId.value)
        put("updated_at", layout.updatedAt.toString())
        put(
            "fields",
            JsonArray(
                listOf(
                    buildJsonObject {
                        put("id", layout.id.value)
                        put("name", layout.name)
                        put("field_id", layout.name)
                        put("active_row_id", layout.activeRowId.value)
                        put("row_count", layout.rows.size)
                        put("plants_per_row", layout.rows.firstOrNull { it.id == layout.activeRowId }?.plantsPerRow ?: 0)
                        put("row_spacing_m", layout.rows.firstOrNull()?.rowSpacingM ?: 0.0)
                        put("plant_spacing_m", layout.rows.firstOrNull { it.id == layout.activeRowId }?.plantSpacingM ?: 0.0)
                        put("start_plant", layout.startPlant)
                        put("plant_step", layout.plantStep)
                        put("plant_cooldown_sec", layout.plantCooldownSec)
                        put(
                            "rows",
                            JsonArray(
                                layout.rows.map { row ->
                                    buildJsonObject {
                                        put("id", row.id.value)
                                        put("row_id", row.id.value)
                                        put("row_index", row.rowIndex)
                                        put("plants_per_row", row.plantsPerRow)
                                        put("plant_spacing_m", row.plantSpacingM)
                                        put("row_spacing_m", row.rowSpacingM)
                                        put("y_m", row.yM)
                                    }
                                },
                            ),
                        )
                    },
                ),
            ),
        )
        put(
            "android_active_layout",
            buildJsonObject {
                put("field_id", layout.id.value)
                put("name", layout.name)
                put("active_row_id", layout.activeRowId.value)
                put("start_plant", layout.startPlant)
                put("plant_step", layout.plantStep)
                put("plant_cooldown_sec", layout.plantCooldownSec)
                put("updated_at", layout.updatedAt.toString())
                put(
                    "rows",
                    JsonArray(
                        layout.rows.map { row ->
                            buildJsonObject {
                                put("row_id", row.id.value)
                                put("row_index", row.rowIndex)
                                put("plants_per_row", row.plantsPerRow)
                                put("plant_spacing_m", row.plantSpacingM)
                                put("row_spacing_m", row.rowSpacingM)
                                put("y_m", row.yM)
                            }
                        },
                    ),
                )
            },
        )
    }

    fun decisionJson(decision: RecordedDecision): JsonObject = buildJsonObject {
        put("run_id", decision.runId.value)
        put("sequence", decision.sequence)
        put("timestamp", decision.timestamp.toString())
        put("mode", decision.mode.name)
        put("field_id", decision.fieldId)
        putNullable("row_id", decision.rowId)
        putNullable("row_side", decision.rowSide?.jsonValue())
        putNullable("row_index", decision.rowIndex)
        putNullable("plant_column", decision.plantColumn)
        putNullable("plant_number", decision.plantNumber)
        putNullable("plant_key", decision.plantKey)
        putNullable("x_m", decision.xM)
        putNullable("y_m", decision.yM)
        putNullable("bbox_px", decision.bboxPx?.toJsonArray())
        putNullable("geometry_confidence", decision.geometryConfidence)
        putNullable("plant_position_confidence", decision.plantPositionConfidence)
        putNullable("geometry_reason", decision.geometryReason)
        put("label", decision.label)
        put("confidence", decision.confidence)
        put("raw_label", decision.rawLabel)
        put("status", statusValue(decision.status))
        put("action", actionValue(decision.action))
        put("frames_used", decision.framesUsed)
        put("reason", decision.reason)
        put("late_frames", decision.lateFrames)
        put("gate_avg_latency_ms", decision.gateAvgLatencyMs)
        put("gate_max_latency_ms", decision.gateMaxLatencyMs)
        put("model_version", decision.modelVersion)
        put("scan_index", decision.scanIndex)
        putNullable("scan_pass", decision.scanPass)
        putNullable("field_complete", decision.fieldComplete)
        putNullable("planned_total_plants", decision.plannedTotalPlants)
        putNullable("plant_display", decision.plantDisplay)
        putNullable("plant_id", decision.plantId)
        putNullable("track_id", decision.trackId)
        putNullable("manual_override", decision.manualOverride.takeIf { it })
        putNullable("notes", decision.notes)
        putNullable("evidence_path", portableEvidencePath(decision.evidencePath))
        putNullable("evidence_status", decision.evidenceStatus)
        putNullable("measurement_distance_m", decision.measurementDistanceM)
        putNullable("measurement_source", decision.measurementSource)
        putNullable("measurement_quality", decision.measurementQuality)
        putNullable("relative_plant_width", decision.relativePlantWidth)
        putNullable("relative_plant_height", decision.relativePlantHeight)
        putNullable("size_source", decision.sizeSource)
        putNullable("size_quality", decision.sizeQuality)
        putNullable("gps_accuracy_m", decision.gpsAccuracyM)
        putNullable("gps_fix_age_seconds", decision.gpsFixAgeSeconds)
        putNullable("top2_margin", decision.top2Margin)
        putNullable("prediction_entropy", decision.predictionEntropy)
        put("treatment_status", decision.treatmentStatus.name)
        putNullable("treatment_note", decision.treatmentNote)
    }

    fun runEventJson(event: RunEvent): JsonObject = buildJsonObject {
        put("id", event.id)
        put("run_id", event.runId.value)
        put("timestamp", event.timestamp.toString())
        put("source", "agribot_android")
        put("level", "info")
        put("type", event.type)
        put("payload", parsePayload(event.payloadJson))
    }

    fun statusValue(status: DecisionStatus): String = when (status) {
        DecisionStatus.OK -> "ok"
        DecisionStatus.UNCERTAIN -> "uncertain"
        DecisionStatus.MANUAL -> "manual"
        DecisionStatus.SKIPPED -> "skipped"
    }

    fun actionValue(action: PlantHealthAction): String = when (action) {
        PlantHealthAction.NONE -> "No action"
        PlantHealthAction.INSPECT_OR_TREAT -> "Inspect or treat"
        PlantHealthAction.RESCAN -> "Rescan this plant"
    }

    fun actionForDecision(label: String, status: DecisionStatus): PlantHealthAction {
        val normalized = label.trim().lowercase().replace("_", " ").replace("-", " ")
        return when {
            status == DecisionStatus.UNCERTAIN || normalized == "uncertain" -> PlantHealthAction.RESCAN
            normalized == "healthy" -> PlantHealthAction.NONE
            else -> PlantHealthAction.INSPECT_OR_TREAT
        }
    }

    private fun BoundingBox.toJsonArray() = JsonArray(
        listOf(left, top, right, bottom).map { JsonPrimitive(it) },
    )

    private fun BoundingBox.toCsvValue(): String = "[$left,$top,$right,$bottom]"

    private fun portableEvidencePath(path: String?): String? {
        if (path.isNullOrBlank()) return null
        val normalized = path.replace('\\', '/')
        val marker = "evidence_frames/"
        val markerIndex = normalized.indexOf(marker)
        if (markerIndex >= 0) return normalized.substring(markerIndex)
        return normalized.substringAfterLast('/').takeIf { it.isNotBlank() }?.let { "$marker$it" }
    }

    private fun RowSide.jsonValue(): String = name.lowercase()

    private fun pathsJson(): JsonObject = buildJsonObject {
        // Android bundles are ZIPs rooted at the run directory. Keep the
        // manifest paths relative to that root so they resolve directly to
        // the entries consumed by the Pi validator and dashboard.
        put("run_dir", ".")
        put("metadata", "metadata.json")
        put("summary", "summary.json")
        put("decisions", "decisions.jsonl")
        put("events_csv", "events.csv")
        put("run_events", "run_events.jsonl")
        put("app_log", "app_log.jsonl")
    }

    private fun parsePayload(payloadJson: String): JsonElement =
        runCatching { json.parseToJsonElement(payloadJson) }
            .getOrElse { buildJsonObject { put("raw", payloadJson) } }

    private fun JsonObjectBuilder.putNullable(name: String, value: String?) {
        if (value != null) put(name, value)
    }

    private fun JsonObjectBuilder.putNullable(name: String, value: Int?) {
        if (value != null) put(name, value)
    }

    private fun JsonObjectBuilder.putNullable(name: String, value: Double?) {
        if (value != null) put(name, value)
    }

    private fun JsonObjectBuilder.putNullable(name: String, value: Float?) {
        if (value != null) put(name, value)
    }

    private fun JsonObjectBuilder.putNullable(name: String, value: Boolean?) {
        if (value != null) put(name, value)
    }

    private fun JsonObjectBuilder.putNullable(name: String, value: JsonArray?) {
        if (value != null) put(name, value)
    }

    private fun csvEscape(value: Any?): String {
        val text = value?.toString().orEmpty()
        return if (text.any { it == ',' || it == '"' || it == '\n' || it == '\r' }) {
            "\"" + text.replace("\"", "\"\"") + "\""
        } else {
            text
        }
    }
}
