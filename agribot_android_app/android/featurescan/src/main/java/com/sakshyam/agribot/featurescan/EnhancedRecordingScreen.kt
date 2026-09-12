package com.sakshyam.agribot.featurescan

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.sakshyam.agribot.camera.CameraPreview
import com.sakshyam.agribot.designsystem.healthy
import com.sakshyam.agribot.designsystem.sick
import com.sakshyam.agribot.designsystem.uncertain
import com.sakshyam.agribot.domain.model.RecordingMode
import com.sakshyam.agribot.domain.model.RunState
import com.sakshyam.agribot.featurescan.tracking.TrackedPlant
import kotlin.math.max
import kotlin.math.roundToInt

@Composable
fun EnhancedRecordingScreen(
    state: ScanUiState,
    viewModel: SideScanViewModel,
    onBack: () -> Unit,
    onNavigateToData: () -> Unit,
) {
    var confirmFinish by rememberSaveable { mutableStateOf(false) }
    var resumeAfterCancel by rememberSaveable { mutableStateOf(false) }
    val requestFinish = {
        resumeAfterCancel = state.isRecording
        if (state.isRecording) viewModel.pause()
        confirmFinish = true
    }
    BackHandler(onBack = requestFinish)
    Column(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background).safeDrawingPadding()) {
        FarmerHeader(
            stringResource(if (state.isRecording) R.string.farmer_scan_active else R.string.scan_paused_title),
            stringResource(R.string.farmer_saved_count, state.decisionsRecorded),
        ) {
            TextButton(onClick = requestFinish) { Text(stringResource(R.string.farmer_finish)) }
        }
        BoxWithConstraints(Modifier.weight(1f)) {
            val controlsMaxHeight = maxHeight * 0.48f
            if (maxWidth > maxHeight) {
                Row(Modifier.fillMaxSize()) {
                    ScanCameraArea(state, viewModel, Modifier.weight(1f).fillMaxSize())
                    RecordingBottomSheet(state, viewModel, onNavigateToData, requestFinish, Modifier.weight(1f).fillMaxSize())
                }
            } else {
                Column(Modifier.fillMaxSize()) {
                    ScanCameraArea(state, viewModel, Modifier.weight(1f).fillMaxWidth())
                    RecordingBottomSheet(state, viewModel, onNavigateToData, requestFinish, Modifier.heightIn(max = controlsMaxHeight))
                }
            }
        }
    }
    if (confirmFinish) {
        val cancel = {
            confirmFinish = false
            if (resumeAfterCancel) viewModel.resume()
        }
        AlertDialog(
            onDismissRequest = cancel,
            title = { Text(stringResource(R.string.farmer_finish_title)) },
            text = { Text(stringResource(R.string.farmer_finish_body)) },
            confirmButton = { TextButton(onClick = { confirmFinish = false; onBack() }) { Text(stringResource(R.string.farmer_finish)) } },
            dismissButton = { TextButton(onClick = cancel) { Text(stringResource(R.string.farmer_keep_scanning)) } },
        )
    }
}

@Composable
private fun ScanCameraArea(state: ScanUiState, viewModel: SideScanViewModel, modifier: Modifier) {
    Box(modifier.background(Color(0xFF101511))) {
        if (state.cameraPermissionGranted) {
            CameraPreview(modifier = Modifier.fillMaxSize(), targetFps = state.targetFps, onFrame = viewModel::onAnalysisFrame)
        } else {
            Text(stringResource(R.string.camera_access_needed), color = Color.White, modifier = Modifier.align(Alignment.Center).padding(20.dp))
        }
        PlantBoundingBoxOverlay(
            plants = state.trackedPlants.filterNot { it.isOccluded }.sortedByDescending { it.isConfirmed }.take(8),
            frameWidth = state.analysisWidth ?: 0, frameHeight = state.analysisHeight ?: 0,
            modifier = Modifier.fillMaxSize(),
        )
        Surface(
            modifier = Modifier.align(Alignment.BottomCenter).padding(12.dp),
            color = Color(0xE6101511), contentColor = Color.White, shape = RoundedCornerShape(12.dp),
        ) {
            Text(
                when {
                    !state.isRecording -> stringResource(R.string.scan_paused_action)
                    state.captureQualityStatus == "retry" -> state.captureQualityAction
                    state.selectedMode == RecordingMode.FRONT_ROW_OVERVIEW -> stringResource(R.string.align_row_in_frame)
                    else -> stringResource(R.string.farmer_aim)
                },
                Modifier.padding(horizontal = 16.dp, vertical = 10.dp), style = MaterialTheme.typography.bodyMedium,
            )
        }
    }
}

@Composable
private fun RecordingBottomSheet(
    state: ScanUiState,
    viewModel: SideScanViewModel,
    onNavigateToData: () -> Unit,
    onFinish: () -> Unit,
    modifier: Modifier = Modifier,
) {
    var showManual by rememberSaveable { mutableStateOf(false) }
    Surface(modifier.fillMaxWidth()) {
        Column(
            Modifier.verticalScroll(rememberScrollState()).padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            state.error?.let { Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodyMedium) }
            val decision = state.latestDecision
            val title = if (state.isRecording) R.string.farmer_aim else R.string.scan_paused_title
            Text(stringResource(title), style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
            if (decision != null && state.isRecording) {
                Text(stringResource(R.string.farmer_saved_record) + " · " + displayLabel(decision.label), style = MaterialTheme.typography.bodyMedium)
            }
            if (state.selectedMode == RecordingMode.FRONT_ROW_OVERVIEW) {
                Button(onClick = viewModel::captureFrontBurst, enabled = state.isRecording, modifier = Modifier.fillMaxWidth().heightIn(min = 54.dp)) {
                    Text(stringResource(R.string.capture_burst))
                }
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                PauseButton(state, viewModel)
                OutlinedButton(onClick = onFinish, modifier = Modifier.weight(1f).heightIn(min = 52.dp)) { Text(stringResource(R.string.farmer_finish)) }
            }
            TextButton(onClick = onNavigateToData, enabled = state.activeRunId != null || state.trackedPlants.isNotEmpty()) {
                Text(stringResource(R.string.farmer_view_results))
            }
            if (state.selectedMode == RecordingMode.SIDE_SCAN) {
                TextButton(onClick = { showManual = !showManual }) {
                    Text(stringResource(if (showManual) R.string.farmer_close_controls else R.string.farmer_correct))
                }
                if (showManual) {
                    Text(stringResource(R.string.farmer_manual_note), style = MaterialTheme.typography.bodySmall)
                    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        listOf("Healthy" to R.string.healthy_action, "Sick" to R.string.sick_action, "Uncertain" to R.string.uncertain_action).forEach { (label, resource) ->
                            OutlinedButton(
                                onClick = { viewModel.markManual(label); showManual = false },
                                enabled = state.isRecording, modifier = Modifier.fillMaxWidth().heightIn(min = 48.dp),
                            ) { Text(stringResource(resource)) }
                        }
                        TextButton(onClick = viewModel::skipCurrentPlant, enabled = state.isRecording) { Text(stringResource(R.string.skip)) }
                    }
                }
            }
        }
    }
}

@Composable
private fun RowScope.QuickMarkButton(label: String, color: Color, onClick: () -> Unit) {
    OutlinedButton(onClick = onClick, modifier = Modifier.weight(1f).height(46.dp)) {
        Text(label, color = color, fontSize = 12.sp, fontWeight = FontWeight.Bold)
    }
}

@Composable
private fun RowScope.PauseButton(state: ScanUiState, viewModel: SideScanViewModel) {
    if (state.runState == RunState.PAUSED) {
        Button(onClick = viewModel::resume, modifier = Modifier.weight(1f).heightIn(min = 52.dp)) { Text(stringResource(R.string.resume)) }
    } else {
        OutlinedButton(onClick = viewModel::pause, modifier = Modifier.weight(1f).heightIn(min = 52.dp)) { Text(stringResource(R.string.pause)) }
    }
}

@Composable
fun PlantBoundingBoxOverlay(
    plants: List<TrackedPlant>,
    frameWidth: Int,
    frameHeight: Int,
    modifier: Modifier = Modifier,
) {
    val density = LocalDensity.current
    var overlaySize by remember { mutableStateOf(IntSize.Zero) }
    Box(modifier = modifier.onSizeChanged { overlaySize = it }) {
        val canvasWidth = overlaySize.width.toFloat()
        val canvasHeight = overlaySize.height.toFloat()
        val safeFrameWidth = frameWidth.coerceAtLeast(1).toFloat()
        val safeFrameHeight = frameHeight.coerceAtLeast(1).toFloat()
        val scale = max(canvasWidth / safeFrameWidth, canvasHeight / safeFrameHeight)
        val cropX = (canvasWidth - safeFrameWidth * scale) / 2f
        val cropY = (canvasHeight - safeFrameHeight * scale) / 2f

        Canvas(modifier = Modifier.matchParentSize()) {
            plants.forEach { plant ->
                val color = plantColor(plant)
                val left = plant.currentBbox.left * scale + cropX
                val top = plant.currentBbox.top * scale + cropY
                val right = plant.currentBbox.right * scale + cropX
                val bottom = plant.currentBbox.bottom * scale + cropY
                val boxSize = Size((right - left).coerceAtLeast(2f), (bottom - top).coerceAtLeast(2f))
                drawRoundRect(
                    color = color.copy(alpha = 0.15f),
                    topLeft = Offset(left, top),
                    size = boxSize,
                    cornerRadius = CornerRadius(12f, 12f),
                )
                drawRoundRect(
                    color = color,
                    topLeft = Offset(left, top),
                    size = boxSize,
                    cornerRadius = CornerRadius(12f, 12f),
                    style = Stroke(width = 5f),
                )
            }
        }

        plants.forEach { plant ->
            val left = (plant.currentBbox.left * scale + cropX).roundToInt().coerceAtLeast(4)
            val top = (plant.currentBbox.top * scale + cropY - with(density) { 48.dp.toPx() }).roundToInt().coerceAtLeast(4)
            val status = if (!plant.isConfirmed || plant.classificationPending) stringResource(R.string.farmer_preview) else displayLabel(plant.healthStatus)
            Surface(
                modifier = Modifier
                    .offset { IntOffset(left, top) }
                    .width(128.dp)
                    .semantics {
                        contentDescription = "${plant.plantId}, $status"
                    },
                shape = RoundedCornerShape(10.dp),
                color = Color(0xE51A211C),
            ) {
                Column(modifier = Modifier.padding(horizontal = 9.dp, vertical = 6.dp)) {
                    Text(plant.plantId, color = Color.White, style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.Bold)
                    Text(
                        status,
                        color = plantColor(plant),
                        style = MaterialTheme.typography.labelSmall,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
            }
        }
    }
}

@Composable
fun SensorDataOverlay(sensorData: SensorData, modifier: Modifier = Modifier) {
    Surface(modifier = modifier, shape = RoundedCornerShape(18.dp), color = Color(0xE51A211C)) {
        Row(modifier = Modifier.padding(horizontal = 12.dp, vertical = 9.dp), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            CompactSensorValue(stringResource(R.string.distance_label), formatMeters(sensorData.distanceM))
            CompactSensorValue(stringResource(R.string.speed_label), formatSpeed(sensorData.speedMps))
            CompactSensorValue(stringResource(R.string.tilt_label), "${(sensorData.tilt * 57.3f).roundToInt()}°")
        }
    }
}

@Composable
private fun CompactSensorValue(label: String, value: String) {
    Column {
        Text(value, color = Color.White, style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.Bold)
        Text(label, color = Color.White.copy(alpha = 0.66f), style = MaterialTheme.typography.labelSmall)
    }
}

@Composable
fun PlantCountOverlay(plantCount: Int, modifier: Modifier = Modifier) {
    Surface(modifier = modifier, shape = RoundedCornerShape(14.dp), color = Color(0xE51A211C)) {
        Text(stringResource(R.string.plant_count_overlay, plantCount), color = Color.White, fontWeight = FontWeight.Bold, modifier = Modifier.padding(10.dp))
    }
}

private fun plantColor(plant: TrackedPlant): Color = when {
    plant.isOccluded || plant.healthStatus.equals("uncertain", true) || plant.healthStatus.equals("unknown", true) -> Color(0xFFD69E2E)
    plant.healthStatus.equals("healthy", true) -> Color(0xFF5BC084)
    else -> Color(0xFFE2765A)
}

private fun displayLabel(label: String): String = label.replace('_', ' ')

@Composable
private fun formatMeters(value: Float?): String = value?.let { stringResource(R.string.meters_value, it) } ?: stringResource(R.string.unavailable)

@Composable
private fun formatSpeed(value: Float?): String = value?.let { stringResource(R.string.speed_value, it) } ?: stringResource(R.string.unavailable)
