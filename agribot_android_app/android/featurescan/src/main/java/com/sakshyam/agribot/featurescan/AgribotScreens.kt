package com.sakshyam.agribot.featurescan

import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.os.Handler
import android.os.Looper
import androidx.compose.foundation.background
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.core.content.FileProvider
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.sakshyam.agribot.designsystem.AgribotTheme
import com.sakshyam.agribot.designsystem.healthy
import com.sakshyam.agribot.designsystem.sick
import com.sakshyam.agribot.designsystem.uncertain
import com.sakshyam.agribot.domain.logic.LeadingIssuePolicy
import com.sakshyam.agribot.domain.model.DiagnosticsSnapshot
import com.sakshyam.agribot.domain.model.GpsPathStatus
import com.sakshyam.agribot.domain.model.ObservedIssue
import com.sakshyam.agribot.domain.model.ExportedFile
import com.sakshyam.agribot.domain.model.RecordingMode
import com.sakshyam.agribot.domain.model.RunState
import com.sakshyam.agribot.domain.model.StepCalibrationStatus
import com.sakshyam.agribot.featurescan.tracking.TrackedPlant

@Composable
fun AgribotScanApp(
    cameraPermissionGranted: Boolean,
    locationPermissionGranted: Boolean = false,
    locationPermissionRequested: Boolean = false,
    activityRecognitionPermissionGranted: Boolean = true,
    activityRecognitionPermissionRequested: Boolean = false,
    onRequestCameraPermission: () -> Unit = {},
    onRequestLocationPermission: () -> Unit = {},
    onRequestActivityRecognitionPermission: () -> Unit = {},
    viewModel: SideScanViewModel = hiltViewModel(),
) {
    val state by viewModel.state.collectAsState()
    val lifecycleOwner = LocalLifecycleOwner.current

    LaunchedEffect(cameraPermissionGranted, locationPermissionGranted, locationPermissionRequested, activityRecognitionPermissionGranted, activityRecognitionPermissionRequested) {
        viewModel.setCameraPermission(cameraPermissionGranted)
        viewModel.setLocationPermission(locationPermissionGranted, locationPermissionRequested)
        viewModel.setActivityRecognitionPermission(activityRecognitionPermissionGranted, activityRecognitionPermissionRequested)
    }
    DisposableEffect(lifecycleOwner, viewModel) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_STOP) {
                // CameraX can briefly stop and recreate the activity during a
                // configuration change (for example when the phone rotates).
                // Treating that as a real background pause leaves the camera
                // preview visible while onAnalysisFrame correctly ignores
                // every frame because the run is PAUSED. Only pause for an
                // actual background transition.
                val activity = lifecycleOwner as? Activity
                if (activity?.isChangingConfigurations != true) {
                    viewModel.onAppBackgrounded()
                }
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
    }

    AgribotTheme(darkTheme = state.darkMode, sunMode = state.sunMode) {
        Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
            when {
                state.showOnboarding -> OnboardingScreen(viewModel::completeOnboarding)
                state.isDataMode -> DataAnalysisScreen(
                    state = state,
                    viewModel = viewModel,
                    onNavigateToRecording = viewModel::exitDataMode,
                )
                state.pendingFrontReview != null -> FrontReviewScreen(
                    state = state,
                    viewModel = viewModel,
                )
                state.runState == RunState.RECORDING || state.runState == RunState.PAUSED ->
                    EnhancedRecordingScreen(
                        state = state,
                        viewModel = viewModel,
                        onBack = viewModel::stop,
                        onNavigateToData = viewModel::enterDataMode,
                    )
                else -> FarmerHomeScreen(
                    state = state,
                    viewModel = viewModel,
                    onRequestCameraPermission = onRequestCameraPermission,
                    onRequestLocationPermission = onRequestLocationPermission,
                    onRequestActivityRecognitionPermission = onRequestActivityRecognitionPermission,
                )
            }
        }
    }
}

@Composable
private fun RowScope.ExportButton(label: String, onClick: () -> Unit) {
    OutlinedButton(onClick = onClick, modifier = Modifier.weight(1f)) { Text(label) }
}

@Composable
internal fun ExportActions(context: Context, exported: ExportedFile) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(stringResource(R.string.last_export, exported.displayName), style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(onClick = { shareExport(context, exported) }, modifier = Modifier.weight(1f)) { Text(stringResource(R.string.share)) }
            OutlinedButton(onClick = { openExport(context, exported) }, modifier = Modifier.weight(1f)) { Text(stringResource(R.string.open)) }
        }
    }
}

@Composable
internal fun SettingsPanel(state: ScanUiState, viewModel: SideScanViewModel) {
    Card(modifier = Modifier.fillMaxWidth(), shape = RoundedCornerShape(22.dp)) {
        val decreaseAnalysisRateDescription = stringResource(R.string.decrease_analysis_rate)
        val increaseAnalysisRateDescription = stringResource(R.string.increase_analysis_rate)
        Column(modifier = Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
            Text(stringResource(R.string.scan_settings), style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
            Text(stringResource(R.string.scan_settings_description), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            SettingToggle(stringResource(R.string.high_contrast_sun_mode), state.sunMode, { viewModel.setSunMode(!state.sunMode) })
            SettingToggle(stringResource(R.string.dark_mode), state.darkMode, { viewModel.setDarkMode(!state.darkMode) })
            SettingToggle(stringResource(R.string.use_gps_reference), state.gpsReferenceEnabled, { viewModel.setGpsReferenceEnabled(!state.gpsReferenceEnabled) })
            var showAdvanced by rememberSaveable { mutableStateOf(false) }
            TextButton(onClick = { showAdvanced = !showAdvanced }) {
                Text(if (showAdvanced) "Hide advanced settings" else "Advanced: walking calibration & runtime")
            }
            Text(
                stringResource(R.string.gps_request_note),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            if (showAdvanced) {
            WalkingCalibrationCard(state, viewModel)
            Text("Walking calibration is for handheld use only. On a tractor, phone motion estimates are not travel measurements.", style = MaterialTheme.typography.bodySmall)
            Text(stringResource(R.string.detection_sensitivity_value, (state.confidenceThreshold * 100).toInt()), style = MaterialTheme.typography.labelLarge)
            Slider(
                value = state.confidenceThreshold,
                onValueChange = viewModel::setConfidenceThreshold,
                valueRange = 0.1f..0.95f,
                modifier = Modifier.fillMaxWidth(),
            )
            Text(stringResource(R.string.sensitivity_description), style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                Column {
                    Text(stringResource(R.string.analysis_rate), style = MaterialTheme.typography.labelLarge)
                    Text(stringResource(R.string.frames_per_second, state.targetFps), style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                    OutlinedButton(
                        onClick = { viewModel.adjustTargetFps(-1) },
                        enabled = state.targetFps > 1,
                        modifier = Modifier.semantics { contentDescription = decreaseAnalysisRateDescription },
                    ) { Text("−") }
                    OutlinedButton(
                        onClick = { viewModel.adjustTargetFps(1) },
                        enabled = state.targetFps < 12,
                        modifier = Modifier.semantics { contentDescription = increaseAnalysisRateDescription },
                    ) { Text("+") }
                }
            }
            state.diagnostics?.let { DiagnosticsCard(it) }
            }
        }
    }
}

@Composable
private fun WalkingCalibrationCard(state: ScanUiState, viewModel: SideScanViewModel) {
    var knownDistanceText by remember { mutableStateOf("10") }
    val calibration = state.walkingCalibration
    val sensor = state.sensorData
    val scanActive = state.runState in setOf(RunState.RECORDING, RunState.PAUSED) || state.pendingFrontReview != null
    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(18.dp),
        color = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.52f),
    ) {
        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(stringResource(R.string.walking_calibration_title), style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
            Text(
                if (sensor.stepCalibrationStatus == StepCalibrationStatus.CALIBRATED) {
                    stringResource(R.string.walking_calibration_saved, sensor.stepLengthM)
                } else {
                    stringResource(R.string.walking_calibration_default, sensor.stepLengthM)
                },
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(
                stringResource(R.string.walking_calibration_help),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            OutlinedTextField(
                value = knownDistanceText,
                onValueChange = { knownDistanceText = it.filter { character -> character.isDigit() || character == '.' } },
                enabled = !calibration.active && !scanActive,
                label = { Text(stringResource(R.string.walking_calibration_distance)) },
                singleLine = true,
                modifier = Modifier.fillMaxWidth(),
            )
            if (calibration.active) {
                Text(
                    stringResource(
                        R.string.walking_calibration_walk,
                        formatMeters(calibration.knownDistanceM),
                        calibration.currentSteps,
                    ),
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.SemiBold,
                )
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(onClick = viewModel::finishWalkingCalibration, modifier = Modifier.weight(1f)) {
                        Text(stringResource(R.string.walking_calibration_finish))
                    }
                    OutlinedButton(onClick = viewModel::cancelWalkingCalibration, modifier = Modifier.weight(1f)) {
                        Text(stringResource(R.string.cancel))
                    }
                }
            } else {
                Button(
                    onClick = { viewModel.startWalkingCalibration(knownDistanceText.toFloatOrNull() ?: Float.NaN) },
                    enabled = !scanActive,
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text(stringResource(R.string.walking_calibration_start))
                }
            }
            calibration.message?.let { message ->
                Text(message, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }
}

@Composable
private fun DiagnosticsCard(snapshot: DiagnosticsSnapshot) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(18.dp),
        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.58f),
    ) {
        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(7.dp)) {
            Text(stringResource(R.string.diagnostics_for_support), style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
            Text(snapshot.readinessSummary, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            DiagnosticRow(stringResource(R.string.model_label), snapshot.modelBundleId)
            DiagnosticRow(stringResource(R.string.camera_label), "${snapshot.analysisResolution} · ${snapshot.latestBenchmarkResult}")
            DiagnosticRow(stringResource(R.string.measurement_label), "${snapshot.measurementSource} · ${snapshot.measurementQuality}")
            DiagnosticRow(stringResource(R.string.walking_profile_label), stringResource(R.string.walking_profile_value, snapshot.stepLengthM, snapshot.stepCalibrationStatus))
            DiagnosticRow(stringResource(R.string.gps_label), snapshot.gpsStatus)
            DiagnosticRow(
                stringResource(R.string.gps_path_label),
                stringResource(
                    R.string.gps_path_value,
                    snapshot.gpsPathStatus.name.lowercase(),
                    snapshot.gpsPathPointCount,
                    snapshot.gpsProvider?.let { " · $it" } ?: "",
                ),
            )
            DiagnosticRow(stringResource(R.string.frame_detector_label), "${snapshot.captureQualityStatus} · ${snapshot.detectionStatus}")
            snapshot.latestDecisionReason?.let { DiagnosticRow(stringResource(R.string.last_decision_gate), it) }
            DiagnosticRow(stringResource(R.string.permissions_label), snapshot.permissionStatus)
            DiagnosticRow(stringResource(R.string.storage_label), snapshot.storageUse)
            if (snapshot.thermalBatteryWarnings != "none") {
                DiagnosticRow(stringResource(R.string.device_health_label), snapshot.thermalBatteryWarnings)
            }
        }
    }
}

@Composable
private fun DiagnosticRow(label: String, value: String) {
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(label, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(value, style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.SemiBold, maxLines = 2, overflow = TextOverflow.Ellipsis)
    }
}

@Composable
private fun SettingToggle(label: String, checked: Boolean, onCheckedChange: () -> Unit) {
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
        Text(label, modifier = Modifier.weight(1f).padding(end = 12.dp), style = MaterialTheme.typography.bodyMedium)
        Switch(
            checked = checked,
            onCheckedChange = { onCheckedChange() },
            modifier = Modifier.semantics {
                contentDescription = label
                stateDescription = if (checked) "On" else "Off"
            },
        )
    }
}

@Composable
private fun ModeSelector(selected: RecordingMode, onSelect: (RecordingMode) -> Unit) {
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        FilterChip(
            selected = selected == RecordingMode.SIDE_SCAN,
            onClick = { onSelect(RecordingMode.SIDE_SCAN) },
            label = { Text(stringResource(R.string.walk_and_scan)) },
            modifier = Modifier.weight(1f),
        )
        FilterChip(
            selected = selected == RecordingMode.FRONT_ROW_OVERVIEW,
            onClick = { onSelect(RecordingMode.FRONT_ROW_OVERVIEW) },
            label = { Text(stringResource(R.string.burst_overview)) },
            modifier = Modifier.weight(1f),
        )
    }
}

private fun isSickPlant(plant: TrackedPlant): Boolean = when {
    plant.classificationPending || plant.isOccluded -> false
    plant.healthStatus.equals("healthy", true) ||
        plant.healthStatus.equals("uncertain", true) ||
        plant.healthStatus.equals("unknown", true) -> false
    else -> true
}

private fun isUncertainPlant(plant: TrackedPlant): Boolean =
    plant.classificationPending ||
        plant.healthStatus.equals("uncertain", true) ||
        plant.healthStatus.equals("unknown", true)

private fun leadingIssue(
    plants: List<TrackedPlant>,
    plannedCount: Int?,
) = LeadingIssuePolicy.evaluate(
    observations = plants.map { plant ->
        ObservedIssue(
            stableId = plant.plantId,
            label = plant.healthStatus,
            confirmed = plant.detectionCount >= 2 && !plant.isOccluded,
            reviewRequired = plant.isOccluded ||
                plant.classificationPending ||
                plant.healthStatus.equals("uncertain", true) ||
                plant.healthStatus.equals("unknown", true),
            geometryValid = plant.currentBbox.width > 0f && plant.currentBbox.height > 0f,
        )
    },
    plannedCount = plannedCount,
)

private fun displayPlantLabel(label: String): String = label
    .replace('_', ' ')
    .replace("Pottassium Deficiency", "Potassium Deficiency", ignoreCase = true)

private fun plantColor(plant: TrackedPlant): Color = when {
    plant.isOccluded || isUncertainPlant(plant) -> Color(0xFFD69E2E)
    isSickPlant(plant) -> Color(0xFFC95B47)
    else -> Color(0xFF3F9B6B)
}

@Composable
private fun formatMeters(value: Float?): String = value?.let {
    if (it < 10f) stringResource(R.string.meters_value, it) else stringResource(R.string.meters_value_rounded, it)
} ?: stringResource(R.string.unavailable)

@Composable
private fun formatGpsAccuracy(value: Float?): String =
    if (value != null && value > 0f) stringResource(
        R.string.gps_accuracy_value,
        if (value < 10f) "%.1f".format(value) else "%.0f".format(value),
    ) else stringResource(R.string.gps_accuracy_pending)

@Composable
private fun formatSpeed(value: Float?): String = value?.let { stringResource(R.string.speed_value, it) } ?: stringResource(R.string.unavailable)

private fun shareExport(context: Context, exported: ExportedFile) {
    val file = java.io.File(exported.absolutePath)
    if (!file.isFile) return
    val uri = FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", file)
    context.postExportIntent(Intent.createChooser(Intent(Intent.ACTION_SEND).apply {
        type = exported.contentType
        putExtra(Intent.EXTRA_STREAM, uri)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }, context.getString(R.string.share_agribot_report)))
}

private fun openExport(context: Context, exported: ExportedFile) {
    val file = java.io.File(exported.absolutePath)
    if (!file.isFile) return
    val uri = FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", file)
    context.postExportIntent(Intent(Intent.ACTION_VIEW).apply {
        setDataAndType(uri, exported.contentType)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    })
}

private fun Context.postExportIntent(intent: Intent) {
    Handler(Looper.getMainLooper()).post {
        try {
            startActivity(intent)
        } catch (_: ActivityNotFoundException) {
            // No compatible viewer/share target; the file remains available for export.
        }
    }
}
