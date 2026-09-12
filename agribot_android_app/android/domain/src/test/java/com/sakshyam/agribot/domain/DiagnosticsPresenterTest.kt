package com.sakshyam.agribot.domain

import com.sakshyam.agribot.domain.logic.DiagnosticsPresenter
import com.sakshyam.agribot.domain.model.RunState
import kotlin.test.Test
import kotlin.test.assertEquals

class DiagnosticsPresenterTest {
    @Test
    fun createsOfflineCpuDiagnosticsForCurrentRuntimeState() {
        val snapshot = DiagnosticsPresenter.snapshot(
            appVersion = "1.0",
            modelBundleId = "agribot-model-bundle-v001",
            cameraPermissionGranted = true,
            analysisWidth = 640,
            analysisHeight = 480,
            targetFps = 10,
            cpuThreads = 4,
            modelStatus = "Ready: agribot-model-bundle-v001",
            latestLatencyMs = 82.6,
            runState = RunState.RECORDING,
            storageUsedBytes = 1_572_864,
            lastExportPath = "C:\\runs\\agribot_run",
            measurementSource = "GPS_REFERENCE",
            measurementQuality = "REFERENCED",
            gpsStatus = "GPS reference active",
            motionEventsObserved = true,
            captureQualityStatus = "accept",
            detectionStatus = "2 plant boxes found",
            latestDecisionReason = "primary_agreement",
        )

        assertEquals("1.0", snapshot.appVersion)
        assertEquals("agribot-model-bundle-v001", snapshot.modelBundleId)
        assertEquals("granted", snapshot.permissionStatus)
        assertEquals("640x480", snapshot.analysisResolution)
        assertEquals("10 FPS / 4 CPU threads", snapshot.cpuThreadProfile)
        assertEquals("82.6 ms", snapshot.latestBenchmarkResult)
        assertEquals(
            "Development ready; field pilot still requires physical-phone benchmark and real field export",
            snapshot.readinessSummary,
        )
        assertEquals("1.5 MB", snapshot.storageUse)
        assertEquals("C:\\runs\\agribot_run", snapshot.lastExportPath)
        assertEquals("GPS_REFERENCE", snapshot.measurementSource)
        assertEquals("REFERENCED", snapshot.measurementQuality)
        assertEquals("GPS reference active", snapshot.gpsStatus)
        assertEquals(true, snapshot.motionEventsObserved)
        assertEquals("accept", snapshot.captureQualityStatus)
        assertEquals("2 plant boxes found", snapshot.detectionStatus)
        assertEquals("primary_agreement", snapshot.latestDecisionReason)
    }

    @Test
    fun readinessSummaryCallsOutSetupBlockersBeforePilotEvidence() {
        val noCamera = DiagnosticsPresenter.snapshot(
            appVersion = "1.0",
            modelBundleId = "agribot-model-bundle-v001",
            cameraPermissionGranted = false,
            analysisWidth = null,
            analysisHeight = null,
            targetFps = 5,
            cpuThreads = 4,
            modelStatus = "Ready: agribot-model-bundle-v001",
            latestLatencyMs = null,
            runState = RunState.READY,
            storageUsedBytes = 0,
            lastExportPath = null,
        )
        val noModel = DiagnosticsPresenter.snapshot(
            appVersion = "1.0",
            modelBundleId = "agribot-model-bundle-v001",
            cameraPermissionGranted = true,
            analysisWidth = null,
            analysisHeight = null,
            targetFps = 5,
            cpuThreads = 4,
            modelStatus = "Missing classifier",
            latestLatencyMs = null,
            runState = RunState.READY,
            storageUsedBytes = 0,
            lastExportPath = null,
        )

        assertEquals("Setup incomplete: camera permission required", noCamera.readinessSummary)
        assertEquals("Setup incomplete: model bundle not ready", noModel.readinessSummary)
    }
}
