package com.sakshyam.agribot.domain.logic

import com.sakshyam.agribot.domain.model.DiagnosticsSnapshot
import com.sakshyam.agribot.domain.model.GpsPathStatus
import com.sakshyam.agribot.domain.model.RunState
import java.util.Locale

object DiagnosticsPresenter {
    fun snapshot(
        appVersion: String,
        modelBundleId: String,
        cameraPermissionGranted: Boolean,
        analysisWidth: Int?,
        analysisHeight: Int?,
        targetFps: Int,
        cpuThreads: Int,
        modelStatus: String,
        latestLatencyMs: Double?,
        runState: RunState,
        storageUsedBytes: Long,
        lastExportPath: String?,
        cameraId: String = "rear-default",
        supportedSizes: String = "reported by CameraX at runtime",
        thermalBatteryWarnings: String = "none",
        measurementSource: String = "none",
        measurementQuality: String = "not_started",
        gpsStatus: String = "not_started",
        gpsProvider: String? = null,
        gpsPathStatus: GpsPathStatus = GpsPathStatus.UNAVAILABLE,
        gpsPathPointCount: Int = 0,
        motionEventsObserved: Boolean = false,
        captureQualityStatus: String = "waiting",
        detectionStatus: String = "waiting",
        latestDecisionReason: String? = null,
        stepLengthM: Float = 0.72f,
        stepCalibrationStatus: String = "default",
    ): DiagnosticsSnapshot =
        DiagnosticsSnapshot(
            appVersion = appVersion,
            modelBundleId = modelBundleId,
            cameraId = cameraId,
            supportedSizes = supportedSizes,
            analysisResolution = resolutionLabel(analysisWidth, analysisHeight),
            cpuThreadProfile = "$targetFps FPS / $cpuThreads CPU threads",
            latestBenchmarkResult = latencyLabel(latestLatencyMs),
            readinessSummary = readinessSummary(cameraPermissionGranted, modelStatus),
            thermalBatteryWarnings = thermalBatteryWarnings,
            storageUse = bytesLabel(storageUsedBytes),
            permissionStatus = if (cameraPermissionGranted) "granted" else "required",
            runState = runState.name.lowercase(),
            lastExportPath = lastExportPath,
            measurementSource = measurementSource,
            measurementQuality = measurementQuality,
            gpsStatus = gpsStatus,
            gpsProvider = gpsProvider,
            gpsPathStatus = gpsPathStatus,
            gpsPathPointCount = gpsPathPointCount,
            motionEventsObserved = motionEventsObserved,
            captureQualityStatus = captureQualityStatus,
            detectionStatus = detectionStatus,
            latestDecisionReason = latestDecisionReason,
            stepLengthM = stepLengthM,
            stepCalibrationStatus = stepCalibrationStatus,
        )

    private fun resolutionLabel(width: Int?, height: Int?): String =
        if (width != null && height != null) "${width}x$height" else "pending first frame"

    private fun latencyLabel(latencyMs: Double?): String =
        latencyMs?.let { "%.1f ms".format(Locale.US, it) } ?: "pending first inference"

    private fun readinessSummary(cameraPermissionGranted: Boolean, modelStatus: String): String =
        when {
            !cameraPermissionGranted -> "Setup incomplete: camera permission required"
            !modelStatus.lowercase(Locale.US).startsWith("ready") -> "Setup incomplete: model bundle not ready"
            else -> "Development ready; field pilot still requires physical-phone benchmark and real field export"
        }

    private fun bytesLabel(bytes: Long): String {
        val mb = bytes / (1024.0 * 1024.0)
        return "%.1f MB".format(Locale.US, mb)
    }
}
