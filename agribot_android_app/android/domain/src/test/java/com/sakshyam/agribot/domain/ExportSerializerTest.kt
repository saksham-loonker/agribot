package com.sakshyam.agribot.domain

import com.sakshyam.agribot.domain.logic.ExportSerializer
import com.sakshyam.agribot.domain.model.BoundingBox
import com.sakshyam.agribot.domain.model.DecisionId
import com.sakshyam.agribot.domain.model.DecisionStatus
import com.sakshyam.agribot.domain.model.DiagnosticsSnapshot
import com.sakshyam.agribot.domain.model.FieldId
import com.sakshyam.agribot.domain.model.FieldLayout
import com.sakshyam.agribot.domain.model.FieldRow
import com.sakshyam.agribot.domain.model.PlantHealthAction
import com.sakshyam.agribot.domain.model.RecordedDecision
import com.sakshyam.agribot.domain.model.RecordingMode
import com.sakshyam.agribot.domain.model.RowId
import com.sakshyam.agribot.domain.model.Run
import com.sakshyam.agribot.domain.model.RunEvent
import com.sakshyam.agribot.domain.model.RunId
import com.sakshyam.agribot.domain.model.RunState
import com.sakshyam.agribot.domain.model.RowSide
import com.sakshyam.agribot.domain.logic.RunSummaryReducer
import java.time.Instant
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ExportSerializerTest {
    @Test
    fun jsonlPreservesPiCompatibleDecisionKeys() {
        val jsonl = ExportSerializer.toJsonl(listOf(decision()))

        assertTrue(jsonl.contains("\"run_id\":\"android_20260608_145500\""))
        assertTrue(jsonl.contains("\"field_id\":\"Field 2\""))
        assertTrue(jsonl.contains("\"row_id\":\"B\""))
        assertTrue(jsonl.contains("\"plant_key\":\"Field 2|B|12\""))
        assertTrue(jsonl.contains("\"status\":\"ok\""))
        assertTrue(jsonl.contains("\"action\":\"No action\""))
        assertTrue(jsonl.contains("\"model_version\":\"agribot-model-bundle-v001\""))
    }

    @Test
    fun jsonlUsesPortableEvidenceFramePath() {
        val jsonl = ExportSerializer.toJsonl(
            listOf(
                decision().copy(
                    evidencePath = "/data/user/0/com.sakshyam.agribot/files/exports/agribot_run_android_20260608_145500/evidence_frames/decision_12_manual.jpg",
                    evidenceStatus = "saved",
                ),
            ),
        )

        assertTrue(jsonl.contains("\"evidence_path\":\"evidence_frames/decision_12_manual.jpg\""))
        assertTrue(jsonl.contains("\"evidence_status\":\"saved\""))
        assertTrue(!jsonl.contains("/data/user/0"))
    }

    @Test
    fun csvUsesPiEventHeaderOrder() {
        val csv = ExportSerializer.toEventsCsv(listOf(decision()))
        val lines = csv.lines()

        assertEquals(
            "sequence,timestamp,epoch_time,field_id,row_id,row_index,plant_column,plant_number,plant_key,x_m,y_m,plant_display,scan_index,plant_id,track_id,label,confidence,status,action,frames_used,reason,late_frames,gate_avg_latency_ms,gate_max_latency_ms,threads,fps_target,row_side,bbox_px,geometry_confidence,plant_position_confidence,geometry_reason,raw_label,manual_override,evidence_path,evidence_status,measurement_distance_m,measurement_source,measurement_quality,relative_plant_width,relative_plant_height,size_source,size_quality,gps_accuracy_m,gps_fix_age_seconds,top2_margin,prediction_entropy,treatment_status,treatment_note",
            lines.first(),
        )
        assertTrue(lines[1].contains("Field 2 / Row B / Plant 12"))
    }

    @Test
    fun jsonlPreservesMeasurementProvenanceWithoutInventingSize() {
        val jsonl = ExportSerializer.toJsonl(
            listOf(
                decision().copy(
                    trackId = "plant_12",
                    measurementDistanceM = 4.2,
                    measurementSource = "PHONE_STEP_SENSOR",
                    measurementQuality = "ESTIMATED",
                    relativePlantWidth = 0.18,
                    relativePlantHeight = 0.41,
                    sizeSource = "CAMERA_RELATIVE",
                    sizeQuality = "RELATIVE",
                    gpsAccuracyM = 8.5,
                    gpsFixAgeSeconds = 1.2,
                    top2Margin = 0.17,
                    predictionEntropy = 0.42,
                ),
            ),
        )

        assertTrue(jsonl.contains("\"measurement_distance_m\":4.2"))
        assertTrue(jsonl.contains("\"track_id\":\"plant_12\""))
        assertTrue(jsonl.contains("\"measurement_source\":\"PHONE_STEP_SENSOR\""))
        assertTrue(jsonl.contains("\"size_quality\":\"RELATIVE\""))
        assertTrue(jsonl.contains("\"gps_fix_age_seconds\":1.2"))
        assertTrue(jsonl.contains("\"top2_margin\":0.17"))
        assertTrue(jsonl.contains("\"prediction_entropy\":0.42"))
        assertTrue(!jsonl.contains("\"width_m\""))
    }

    @Test
    fun csvAppendsFrontOverviewGeometryWithoutMovingPiColumns() {
        val csv = ExportSerializer.toEventsCsv(listOf(frontDecision()))
        val lines = csv.lines()

        assertTrue(lines.first().startsWith("sequence,timestamp,epoch_time,field_id,row_id,row_index"))
        assertTrue(lines.first().contains(",row_side,bbox_px,geometry_confidence,plant_position_confidence,geometry_reason,raw_label,manual_override,evidence_path,evidence_status"))
        assertTrue(lines[1].contains(",left,\"[120.0,220.0,260.0,420.0]\",0.86,0.79,row_rail_fit_and_spacing_estimate,Late_blight,false,evidence_frames/front_1.jpg,saved"))
    }

    @Test
    fun runBundleJsonContainsPiCompatibleFilesPayloads() {
        val run = Run(
            runId = RunId("android_20260608_145500"),
            mode = RecordingMode.SIDE_SCAN,
            fieldLayoutId = FieldId("field_2"),
            startedAt = Instant.parse("2026-06-08T09:24:00Z"),
            completedAt = Instant.parse("2026-06-08T09:30:00Z"),
            state = RunState.COMPLETED,
            targetFps = 5,
            modelBundleId = "agribot-model-bundle-v001",
        )
        val decisions = listOf(decision())
        val summary = RunSummaryReducer.reduce(run.runId, decisions)
        val layout = fieldLayout()

        val metadata = ExportSerializer.toMetadataJson(run, layout)
        val summaryJson = ExportSerializer.toSummaryJson(summary, run)
        val latestRun = ExportSerializer.toLatestRunJson(run, summary)
        val fieldLayout = ExportSerializer.toFieldLayoutJson(layout)

        assertTrue(metadata.contains("\"run_id\":\"android_20260608_145500\""))
        assertTrue(metadata.contains("\"origin\":\"android\""))
        assertTrue(metadata.contains("\"paths\":"))
        assertTrue(metadata.contains("\"field_map\":"))
        assertTrue(metadata.contains("\"farm_layout\":"))
        assertTrue(summaryJson.contains("\"decisions\":1"))
        assertTrue(summaryJson.contains("\"ok\":1"))
        assertTrue(summaryJson.contains("\"uncertain\":0"))
        assertTrue(summaryJson.contains("\"completed\":true"))
        assertTrue(summaryJson.contains("\"paths\":"))
        assertTrue(summaryJson.contains("\"total_decisions\":1"))
        assertTrue(summaryJson.contains("\"sick_count\":0"))
        assertTrue(latestRun.contains("\"run_id\":\"android_20260608_145500\""))
        assertTrue(latestRun.contains("\"latest_run_id\":\"android_20260608_145500\""))
        assertTrue(latestRun.contains("\"run_dir\":\".\""))
        assertTrue(latestRun.contains("\"summary\":\"summary.json\""))
        assertTrue(metadata.contains("\"run_events\":\"run_events.jsonl\""))
        assertTrue(metadata.contains("\"app_log\":\"app_log.jsonl\""))
        assertTrue(fieldLayout.contains("\"active_field_id\":\"field_2\""))
        assertTrue(fieldLayout.contains("\"fields\":["))
        assertTrue(fieldLayout.contains("\"row_id\":\"B\""))
        assertTrue(fieldLayout.contains("\"plants_per_row\":57"))
    }

    @Test
    fun runEventJsonlExportsStructuredAppLogRecords() {
        val jsonl = ExportSerializer.toRunEventsJsonl(
            listOf(
                RunEvent(
                    id = "android_20260608_145500_model_loaded_20260608T092400Z",
                    runId = RunId("android_20260608_145500"),
                    timestamp = Instant.parse("2026-06-08T09:24:00Z"),
                    type = "model_loaded",
                    payloadJson = """{"bundle_id":"agribot-model-bundle-v001","cpu_default":true}""",
                ),
            ),
        )

        assertTrue(jsonl.endsWith("\n"))
        assertTrue(jsonl.contains("\"source\":\"agribot_android\""))
        assertTrue(jsonl.contains("\"level\":\"info\""))
        assertTrue(jsonl.contains("\"type\":\"model_loaded\""))
        assertTrue(jsonl.contains("\"payload\":{\"bundle_id\":\"agribot-model-bundle-v001\",\"cpu_default\":true}"))
    }

    @Test
    fun diagnosticsExportPreservesMeasurementAndDecisionProvenance() {
        val json = ExportSerializer.toDiagnosticsJson(
            DiagnosticsSnapshot(
                appVersion = "1.0",
                modelBundleId = "agribot-model-bundle-v001",
                cameraId = "rear-default",
                supportedSizes = "1920x1080",
                analysisResolution = "1080x1920",
                cpuThreadProfile = "5 FPS / 4 CPU threads",
                latestBenchmarkResult = "84.2 ms",
                readinessSummary = "ready",
                thermalBatteryWarnings = "none",
                storageUse = "1.0 MB",
                permissionStatus = "granted",
                runState = "recording",
                lastExportPath = null,
                measurementSource = "PHONE_STEP_SENSOR",
                measurementQuality = "ESTIMATED",
                gpsStatus = "GPS weak",
                motionEventsObserved = true,
                captureQualityStatus = "accept",
                detectionStatus = "1 plant box found",
                latestDecisionReason = "primary_agreement",
            ),
        )

        assertTrue(json.contains("\"offline\":true"))
        assertTrue(json.contains("\"network_required\":false"))
        assertTrue(json.contains("\"measurement_source\":\"PHONE_STEP_SENSOR\""))
        assertTrue(json.contains("\"motion_events_observed\":true"))
        assertTrue(json.contains("\"latest_decision_reason\":\"primary_agreement\""))
    }

    private fun decision() = RecordedDecision(
        id = DecisionId("decision_12"),
        runId = RunId("android_20260608_145500"),
        sequence = 12,
        timestamp = Instant.parse("2026-06-08T09:25:21Z"),
        epochTime = 1780910721.0,
        mode = RecordingMode.SIDE_SCAN,
        fieldId = "Field 2",
        rowId = "B",
        rowSide = null,
        rowIndex = 2,
        plantColumn = 12,
        plantNumber = 12,
        plantKey = "Field 2|B|12",
        xM = 3.85,
        yM = 1.4,
        bboxPx = null,
        geometryConfidence = null,
        plantPositionConfidence = null,
        geometryReason = null,
        label = "Healthy",
        confidence = 0.918234f,
        rawLabel = "Healthy",
        status = DecisionStatus.OK,
        action = PlantHealthAction.NONE,
        framesUsed = 2,
        reason = "primary_agreement",
        lateFrames = 0,
        gateAvgLatencyMs = 84.2,
        gateMaxLatencyMs = 88.1,
        modelVersion = "agribot-model-bundle-v001",
        scanIndex = 12,
        scanPass = 1,
        fieldComplete = false,
        plannedTotalPlants = 57,
        plantDisplay = "Field 2 / Row B / Plant 12",
        plantId = 12,
        threads = 4,
        fpsTarget = 5.0,
    )

    private fun frontDecision() = decision().copy(
        id = DecisionId("front_decision_1"),
        sequence = 1,
        mode = RecordingMode.FRONT_ROW_OVERVIEW,
        rowId = "A",
        rowSide = RowSide.LEFT,
        rowIndex = 1,
        plantColumn = 3,
        plantNumber = 12,
        plantKey = "Field 2|A|12",
        xM = null,
        yM = null,
        bboxPx = BoundingBox(120f, 220f, 260f, 420f),
        geometryConfidence = 0.86f,
        plantPositionConfidence = 0.79f,
        geometryReason = "row_rail_fit_and_spacing_estimate",
        label = "Late_blight",
        confidence = 0.91f,
        rawLabel = "Late_blight",
        status = DecisionStatus.OK,
        action = PlantHealthAction.INSPECT_OR_TREAT,
        framesUsed = 7,
        reason = "front_burst_grouped_candidate",
        plantDisplay = "Field 2 / Left row A / Plant 12",
        evidencePath = "/data/user/0/com.sakshyam.agribot/files/exports/agribot_run_android_20260608_145500/evidence_frames/front_1.jpg",
        evidenceStatus = "saved",
    )

    private fun fieldLayout() = FieldLayout(
        id = FieldId("field_2"),
        name = "Field 2",
        activeRowId = RowId("B"),
        rows = listOf(
            FieldRow(
                id = RowId("B"),
                rowIndex = 2,
                plantsPerRow = 57,
                plantSpacingM = 0.35,
                rowSpacingM = 0.7,
                yM = 1.4,
            ),
        ),
        startPlant = 1,
        plantStep = 1,
        plantCooldownSec = 2.0,
        updatedAt = Instant.parse("2026-06-08T09:00:00Z"),
    )
}
