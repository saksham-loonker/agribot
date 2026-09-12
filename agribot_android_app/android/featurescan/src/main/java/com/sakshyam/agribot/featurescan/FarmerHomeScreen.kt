package com.sakshyam.agribot.featurescan

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.role
import androidx.compose.ui.semantics.selected
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import com.sakshyam.agribot.domain.model.RecordingMode
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import kotlin.math.roundToInt

/** The home screen keeps the next action visible while supporting large text. */
@Composable
internal fun FarmerHomeScreen(
    state: ScanUiState,
    viewModel: SideScanViewModel,
    onRequestCameraPermission: () -> Unit,
    onRequestLocationPermission: () -> Unit,
    onRequestActivityRecognitionPermission: () -> Unit,
) {
    var settings by rememberSaveable { mutableStateOf(false) }
    var fieldSetup by rememberSaveable { mutableStateOf(false) }
    val context = LocalContext.current
    val colors = MaterialTheme.colorScheme
    Column(Modifier.fillMaxSize().safeDrawingPadding()) {
        FarmerHeader(stringResource(R.string.farmer_home_title), stringResource(R.string.farmer_offline)) {
            TextButton(onClick = { settings = true }) { Text(stringResource(R.string.settings)) }
        }
        LazyColumn(
            modifier = Modifier.weight(1f),
            contentPadding = PaddingValues(horizontal = 20.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(20.dp),
        ) {
            state.error?.let { message -> item { FarmerNotice(stringResource(R.string.attention_title), message) } }
            item {
                Text(stringResource(R.string.farmer_scan_heading), style = MaterialTheme.typography.headlineMedium)
                Spacer(Modifier.height(6.dp))
                Text(stringResource(R.string.farmer_scan_intro), style = MaterialTheme.typography.bodyLarge, color = colors.onSurfaceVariant)
            }
            item {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    ScanChoice(
                        title = stringResource(R.string.walk_and_scan),
                        description = stringResource(R.string.farmer_side_description),
                        chosen = state.selectedMode == RecordingMode.SIDE_SCAN,
                        onClick = { viewModel.selectMode(RecordingMode.SIDE_SCAN) },
                    )
                    ScanChoice(
                        title = stringResource(R.string.burst_overview),
                        description = stringResource(R.string.farmer_burst_description),
                        chosen = state.selectedMode == RecordingMode.FRONT_ROW_OVERVIEW,
                        onClick = { viewModel.selectMode(RecordingMode.FRONT_ROW_OVERVIEW) },
                    )
                }
            }
            item {
                Surface(shape = RoundedCornerShape(16.dp), color = colors.surfaceVariant) {
                    Column(Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text(state.activeLayout?.name ?: stringResource(R.string.default_field_name), style = MaterialTheme.typography.titleMedium)
                        Text(stringResource(R.string.farmer_row_plan_note), style = MaterialTheme.typography.bodyMedium)
                        TextButton(onClick = { fieldSetup = !fieldSetup }) { Text(stringResource(R.string.farmer_row_plan)) }
                        if (fieldSetup) {
                            state.activeLayout?.rows?.forEach { row ->
                                FilterChip(
                                    selected = row.id == state.activeLayout?.activeRowId,
                                    onClick = { viewModel.selectActiveRow(row.id.value) },
                                    label = { Text(stringResource(R.string.farmer_row_count, row.rowIndex, row.plantsPerRow)) },
                                )
                            }
                            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                                OutlinedButton(onClick = { viewModel.adjustActiveRowPlants(-1) }, modifier = Modifier.weight(1f)) { Text(stringResource(R.string.farmer_fewer_plants)) }
                                OutlinedButton(onClick = { viewModel.adjustActiveRowPlants(1) }, modifier = Modifier.weight(1f)) { Text(stringResource(R.string.farmer_more_plants)) }
                            }
                        }
                    }
                }
            }
            item {
                Text(stringResource(R.string.farmer_saved_scans), style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
            }
            if (state.runHistory.isEmpty()) {
                item {
                    Text(stringResource(R.string.farmer_no_scans), style = MaterialTheme.typography.bodyLarge, color = colors.onSurfaceVariant)
                }
            }
            items(state.runHistory, key = { it.runId.value }) { run ->
                Surface(
                    onClick = { viewModel.enterDataMode(); viewModel.selectRunDetail(run.runId) },
                    shape = RoundedCornerShape(16.dp),
                    border = BorderStroke(1.dp, colors.outlineVariant),
                ) {
                    Column(Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(5.dp)) {
                        Text(run.fieldName, style = MaterialTheme.typography.titleMedium)
                        Text(
                            DateTimeFormatter.ofPattern("d MMM, HH:mm").withZone(ZoneId.systemDefault()).format(run.startedAt),
                            style = MaterialTheme.typography.bodyMedium, color = colors.onSurfaceVariant,
                        )
                        Text("${run.decisionCountLabel} · ${run.modeLabel}", style = MaterialTheme.typography.bodyMedium)
                        Text(stringResource(R.string.farmer_view_results), color = colors.primary, fontWeight = FontWeight.SemiBold)
                    }
                }
            }
            state.lastExportedFile?.let { file -> item { ExportActions(context, file) } }
        }
        Surface(shadowElevation = 6.dp) {
            Column(Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Button(
                    onClick = { if (!state.cameraPermissionGranted) onRequestCameraPermission() else viewModel.start() },
                    enabled = !state.cameraPermissionGranted || state.modelReady,
                    modifier = Modifier.fillMaxWidth().heightIn(min = 56.dp),
                    shape = RoundedCornerShape(16.dp),
                ) {
                    Text(stringResource(if (!state.cameraPermissionGranted) R.string.start_allow_camera else R.string.start_field_scan), style = MaterialTheme.typography.titleMedium)
                }
                if (!state.modelReady) {
                    Text(stringResource(R.string.farmer_model_wait), style = MaterialTheme.typography.bodySmall)
                    if (state.modelReadinessErrors.isEmpty()) {
                        val modelProgress = state.modelLoadingProgress.coerceIn(0f, 1f)
                        val percent = (modelProgress * 100f).roundToInt()
                        val progressDescription = stringResource(R.string.farmer_model_progress, percent)
                        Text(progressDescription, style = MaterialTheme.typography.labelSmall, color = colors.onSurfaceVariant)
                        LinearProgressIndicator(
                            progress = { modelProgress },
                            modifier = Modifier
                                .fillMaxWidth()
                                .semantics {
                                    contentDescription = progressDescription
                                    stateDescription = progressDescription
                                },
                        )
                    } else {
                        Text(state.modelStatus, style = MaterialTheme.typography.bodySmall, color = colors.error)
                    }
                }
            }
        }
    }
    if (settings) {
        Dialog(onDismissRequest = { settings = false }) {
            Surface(shape = RoundedCornerShape(24.dp)) {
                Column(Modifier.fillMaxWidth().heightIn(max = 640.dp)) {
                    Row(Modifier.fillMaxWidth().padding(horizontal = 8.dp), horizontalArrangement = Arrangement.End) {
                        TextButton(onClick = { settings = false }) { Text(stringResource(R.string.farmer_done)) }
                    }
                    Column(Modifier.weight(1f, fill = false).verticalScroll(rememberScrollState())) {
                        SettingsPanel(state, viewModel)
                        if (state.gpsReferenceEnabled && !state.locationPermissionGranted) {
                            TextButton(onClick = onRequestLocationPermission) { Text(stringResource(R.string.start_allow_gps)) }
                        }
                        if (!state.activityRecognitionPermissionGranted) {
                            TextButton(onClick = onRequestActivityRecognitionPermission) { Text(stringResource(R.string.farmer_enable_steps)) }
                        }
                    }
                }
            }
        }
    }
}

@Composable
internal fun FarmerHeader(title: String, subtitle: String, actions: @Composable RowScope.() -> Unit) {
    Row(Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 12.dp), verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
            Text(subtitle, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        actions()
    }
}

@Composable
private fun ScanChoice(title: String, description: String, chosen: Boolean, onClick: () -> Unit) {
    Surface(
        onClick = onClick,
        modifier = Modifier.fillMaxWidth().semantics { role = Role.RadioButton; selected = chosen },
        shape = RoundedCornerShape(18.dp),
        color = if (chosen) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surface,
        border = BorderStroke(if (chosen) 2.dp else 1.dp, if (chosen) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outlineVariant),
    ) {
        Row(Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(5.dp)) {
                Text(title, style = MaterialTheme.typography.titleMedium)
                Text(description, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            RadioButton(selected = chosen, onClick = null)
        }
    }
}

@Composable
internal fun FarmerNotice(title: String, body: String) {
    Surface(color = MaterialTheme.colorScheme.secondaryContainer, shape = RoundedCornerShape(16.dp)) {
        Column(Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(5.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium)
            Text(body, style = MaterialTheme.typography.bodyMedium)
        }
    }
}
