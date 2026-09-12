package com.sakshyam.agribot.domain

import com.sakshyam.agribot.domain.logic.RunEventFactory
import com.sakshyam.agribot.domain.model.DecisionId
import com.sakshyam.agribot.domain.model.DecisionStatus
import com.sakshyam.agribot.domain.model.GpsPathPoint
import com.sakshyam.agribot.domain.model.GpsPathStatus
import com.sakshyam.agribot.domain.model.PlantDecision
import com.sakshyam.agribot.domain.model.RunId
import java.time.Instant
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class RunEventFactoryTest {
    @Test
    fun createsPauseResumeStopEventsWithStableIdsAndPayloads() {
        val runId = RunId("android_20260608_145500")
        val now = Instant.parse("2026-06-08T09:25:00Z")

        val paused = RunEventFactory.paused(runId, now, decisionsRecorded = 4)
        val resumed = RunEventFactory.resumed(runId, now, nextScanIndex = 5)
        val stopped = RunEventFactory.stopped(runId, now, decisionsRecorded = 7)

        assertTrue(paused.id.startsWith("android_20260608_145500_paused_20260608T092500Z_"))
        assertEquals("paused", paused.type)
        assertTrue(paused.payloadJson.contains("\"decisions_recorded\":4"))
        assertEquals("resumed", resumed.type)
        assertTrue(resumed.payloadJson.contains("\"next_scan_index\":5"))
        assertEquals("stopped", stopped.type)
        assertTrue(stopped.payloadJson.contains("\"decisions_recorded\":7"))
    }

    @Test
    fun exportCreatedEventIncludesPathAndKind() {
        val event = RunEventFactory.exportCreated(
            runId = RunId("android_20260608_145500"),
            timestamp = Instant.parse("2026-06-08T09:26:00Z"),
            exportKind = "csv",
            absolutePath = "C:\\runs\\agribot \"export\".csv",
        )

        assertTrue(event.id.startsWith("android_20260608_145500_export_created_20260608T092600Z_"))
        assertEquals("export_created", event.type)
        assertTrue(event.payloadJson.contains("\"export_kind\":\"csv\""))
        assertTrue(event.payloadJson.contains("\"absolute_path\":\"C:\\\\runs\\\\agribot \\\"export\\\".csv\""))
    }

    @Test
    fun retakeRequestedEventIncludesDecisionAndScanIndex() {
        val event = RunEventFactory.retakeRequested(
            runId = RunId("android_20260608_145500"),
            timestamp = Instant.parse("2026-06-08T09:27:00Z"),
            decisionId = DecisionId("android_20260608_145500_4"),
            scanIndex = 4,
        )

        assertTrue(event.id.startsWith("android_20260608_145500_retake_requested_20260608T092700Z_"))
        assertEquals("retake_requested", event.type)
        assertTrue(event.payloadJson.contains("\"decision_id\":\"android_20260608_145500_4\""))
        assertTrue(event.payloadJson.contains("\"scan_index\":4"))
        assertTrue(event.payloadJson.contains("\"reason\":\"operator_retake\""))
    }

    @Test
    fun manualSnapshotEventIncludesEvidencePathAndStatus() {
        val event = RunEventFactory.manualSnapshot(
            runId = RunId("android_20260608_145500"),
            timestamp = Instant.parse("2026-06-08T09:28:00Z"),
            evidencePath = "C:\\runs\\evidence_frames\\snap.jpg",
            evidenceStatus = "saved",
            operatorDisplayName = "Field Lead",
        )

        assertTrue(event.id.startsWith("android_20260608_145500_manual_snapshot_20260608T092800Z_"))
        assertEquals("manual_snapshot", event.type)
        assertTrue(event.payloadJson.contains("\"evidence_status\":\"saved\""))
        assertTrue(event.payloadJson.contains("\"evidence_path\":\"C:\\\\runs\\\\evidence_frames\\\\snap.jpg\""))
        assertTrue(event.payloadJson.contains("\"operator\":\"Field Lead\""))
    }

    @Test
    fun manualOverrideEventUsesPlanPayloadShapeAndConfiguredOperator() {
        val event = RunEventFactory.manualOverride(
            runId = RunId("android_20260608_145500"),
            timestamp = Instant.parse("2026-06-08T09:32:00Z"),
            decisionId = DecisionId("decision_123"),
            original = PlantDecision(
                label = "Uncertain",
                confidence = 0.42f,
                status = DecisionStatus.UNCERTAIN,
                framesUsed = 3,
                reason = "low_confidence",
            ),
            corrected = PlantDecision(
                label = "Healthy",
                confidence = 1f,
                status = DecisionStatus.MANUAL,
                framesUsed = 1,
                reason = "manual_override",
            ),
            reason = "leaf partly hidden",
            operatorDisplayName = "Field Lead",
        )

        assertEquals("manual_override", event.type)
        assertTrue(event.payloadJson.contains("\"decision_id\":\"decision_123\""))
        assertTrue(event.payloadJson.contains("\"operator\":\"Field Lead\""))
        assertTrue(event.payloadJson.contains("\"reason\":\"leaf partly hidden\""))
        assertTrue(event.payloadJson.contains("\"original\":{\"label\":\"Uncertain\",\"status\":\"uncertain\",\"confidence\":0.42}"))
        assertTrue(event.payloadJson.contains("\"updated\":{\"label\":\"Healthy\",\"status\":\"manual\",\"confidence\":1.0}"))
    }

    @Test
    fun manualOverrideFallsBackToLocalUserWhenOperatorIsBlank() {
        val event = RunEventFactory.manualOverride(
            runId = RunId("android_20260608_145500"),
            timestamp = Instant.parse("2026-06-08T09:33:00Z"),
            decisionId = DecisionId("decision_124"),
            original = null,
            corrected = PlantDecision(
                label = "Sick",
                confidence = 1f,
                status = DecisionStatus.MANUAL,
                framesUsed = 1,
                reason = "manual_override",
            ),
            reason = null,
            operatorDisplayName = " ",
        )

        assertEquals("manual_override", event.type)
        assertTrue(event.payloadJson.contains("\"operator\":\"local_user\""))
        assertTrue(event.payloadJson.contains("\"original\":null"))
        assertTrue(event.payloadJson.contains("\"reason\":null"))
    }

    @Test
    fun permissionLostEventIncludesPermissionAndState() {
        val event = RunEventFactory.permissionLost(
            runId = RunId("android_20260608_145500"),
            timestamp = Instant.parse("2026-06-08T09:29:00Z"),
            permission = "camera",
            previousState = "recording",
        )

        assertTrue(event.id.startsWith("android_20260608_145500_permission_lost_20260608T092900Z_"))
        assertEquals("permission_lost", event.type)
        assertTrue(event.payloadJson.contains("\"permission\":\"camera\""))
        assertTrue(event.payloadJson.contains("\"previous_state\":\"recording\""))
    }

    @Test
    fun modelLoadedEventIncludesBundleAndCpuDefault() {
        val event = RunEventFactory.modelLoaded(
            runId = RunId("android_20260608_145500"),
            timestamp = Instant.parse("2026-06-08T09:30:00Z"),
            bundleId = "agribot-model-bundle-v001",
            cpuDefault = true,
        )

        assertTrue(event.id.startsWith("android_20260608_145500_model_loaded_20260608T093000Z_"))
        assertEquals("model_loaded", event.type)
        assertTrue(event.payloadJson.contains("\"bundle_id\":\"agribot-model-bundle-v001\""))
        assertTrue(event.payloadJson.contains("\"cpu_default\":true"))
    }

    @Test
    fun thermalWarningEventIncludesStatusAndAction() {
        val event = RunEventFactory.thermalWarning(
            runId = RunId("android_20260608_145500"),
            timestamp = Instant.parse("2026-06-08T09:31:00Z"),
            thermalStatus = "moderate",
            action = "reduced_fps",
            targetFps = 3,
        )

        assertTrue(event.id.startsWith("android_20260608_145500_thermal_warning_20260608T093100Z_"))
        assertEquals("thermal_warning", event.type)
        assertTrue(event.payloadJson.contains("\"thermal_status\":\"moderate\""))
        assertTrue(event.payloadJson.contains("\"action\":\"reduced_fps\""))
        assertTrue(event.payloadJson.contains("\"target_fps\":3"))
    }

    @Test
    fun gpsPathSnapshotStoresLocalCoordinatesAndProviderWithoutRawLocation() {
        val event = RunEventFactory.gpsPathSnapshot(
            runId = RunId("android_20260608_145500"),
            timestamp = Instant.parse("2026-06-08T09:34:00Z"),
            status = GpsPathStatus.REFERENCE_PATH,
            provider = "gps",
            points = listOf(
                GpsPathPoint(1, 0.0, 0.0, 8f, "gps", 10L),
                GpsPathPoint(2, 4.2, 1.5, 8f, "gps", 20L),
            ),
        )

        assertEquals("gps_path_snapshot", event.type)
        assertTrue(event.id.startsWith("android_20260608_145500_gps_path_checkpoint_1780911240000_2_"))
        assertTrue(event.payloadJson.contains("\"status\":\"REFERENCE_PATH\""))
        assertTrue(event.payloadJson.contains("\"provider\":\"gps\""))
        assertTrue(event.payloadJson.contains("\"point_count\":2"))
        assertTrue(event.payloadJson.contains("\"east_m\":4.2"))
        assertTrue(!event.payloadJson.contains("latitude"))
        assertTrue(!event.payloadJson.contains("longitude"))
    }

    @Test
    fun repeatedEventsAtTheSameInstantHaveDistinctPrimaryKeys() {
        val runId = RunId("android_20260608_145500")
        val timestamp = Instant.parse("2026-06-08T09:35:00Z")

        val first = RunEventFactory.paused(runId, timestamp, decisionsRecorded = 1)
        val second = RunEventFactory.paused(runId, timestamp, decisionsRecorded = 1)

        assertTrue(first.id != second.id)
    }
}
