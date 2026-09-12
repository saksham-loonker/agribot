package com.sakshyam.agribot.featurescan

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.sakshyam.agribot.designsystem.healthy
import com.sakshyam.agribot.designsystem.sick
import com.sakshyam.agribot.designsystem.uncertain

@Composable
fun DataAnalysisScreen(state: ScanUiState, viewModel: SideScanViewModel, onNavigateToRecording: () -> Unit) {
    BackHandler(onBack = onNavigateToRecording)
    val selectedSaved = state.selectedRunId != null
    val map = if (selectedSaved) state.selectedRunFieldMap else state.activeRunFieldMap
    val summary = FarmerResultsSummary.from(if (selectedSaved) emptyList() else state.trackedPlants, map)
    val runId = state.selectedRunId ?: state.activeRunId
    var filter by rememberSaveable(runId?.value) { mutableStateOf("ALL") }
    var query by rememberSaveable(runId?.value) { mutableStateOf("") }
    var showMap by rememberSaveable { mutableStateOf(false) }
    var showExport by rememberSaveable { mutableStateOf(false) }
    var expanded by rememberSaveable(runId?.value) { mutableStateOf<String?>(null) }
    val context = LocalContext.current
    val findings = summary.findings.filter {
        (filter == "ALL" || it.category.name == filter) &&
            (query.isBlank() || it.location.contains(query.trim(), true) || it.label.contains(query.trim(), true))
    }
    Column(Modifier.fillMaxSize().safeDrawingPadding()) {
        FarmerHeader(stringResource(R.string.farmer_findings), stringResource(R.string.farmer_checked, summary.total)) {
            TextButton(onClick = onNavigateToRecording) { Text(stringResource(R.string.back)) }
        }
        LazyColumn(
            modifier = Modifier.weight(1f), contentPadding = PaddingValues(20.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            state.error?.let { message -> item { FarmerNotice(stringResource(R.string.attention_title), message) } }
            item {
                Text(map?.fieldLabel ?: state.activeLayout?.name.orEmpty(), style = MaterialTheme.typography.titleLarge)
                Spacer(Modifier.height(8.dp))
                Text(stringResource(R.string.farmer_result_note), style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            item {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    SummaryLine(stringResource(R.string.farmer_check), summary.check, MaterialTheme.colorScheme.sick)
                    SummaryLine(stringResource(R.string.farmer_review), summary.review, MaterialTheme.colorScheme.uncertain)
                    SummaryLine(stringResource(R.string.farmer_healthy), summary.healthy, MaterialTheme.colorScheme.healthy)
                }
            }
            if (summary.total > 0) {
                item {
                    OutlinedTextField(
                        value = query, onValueChange = { query = it }, singleLine = true,
                        modifier = Modifier.fillMaxWidth(), label = { Text(stringResource(R.string.find_plant_condition)) },
                        shape = RoundedCornerShape(14.dp),
                    )
                }
                item {
                    Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        listOf("ALL" to R.string.farmer_all, "CHECK" to R.string.farmer_check, "REVIEW" to R.string.farmer_review, "HEALTHY" to R.string.farmer_healthy).forEach { (key, label) ->
                            FilterChip(selected = filter == key, onClick = { filter = key }, label = { Text(stringResource(label)) })
                        }
                    }
                }
            }
            if (state.runDetailLoading) {
                item { LinearProgressIndicator(Modifier.fillMaxWidth()) }
            } else if (findings.isEmpty()) {
                item {
                    FarmerNotice(
                        stringResource(R.string.farmer_no_findings),
                        stringResource(if (summary.total == 0) R.string.farmer_empty_help else R.string.farmer_filter_empty),
                    )
                }
            }
            items(findings, key = { it.id }) { finding ->
                val color = when (finding.category) {
                    FarmerResultCategory.HEALTHY -> MaterialTheme.colorScheme.healthy
                    FarmerResultCategory.CHECK -> MaterialTheme.colorScheme.sick
                    FarmerResultCategory.REVIEW -> MaterialTheme.colorScheme.uncertain
                }
                val status = when (finding.category) {
                    FarmerResultCategory.HEALTHY -> R.string.farmer_healthy
                    FarmerResultCategory.CHECK -> R.string.farmer_check
                    FarmerResultCategory.REVIEW -> R.string.farmer_review
                }
                Surface(
                    onClick = { expanded = if (expanded == finding.id) null else finding.id },
                    shape = RoundedCornerShape(16.dp), border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant),
                ) {
                    Column(Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        Text(finding.location, style = MaterialTheme.typography.titleMedium)
                        Text(stringResource(status), color = color, fontWeight = FontWeight.SemiBold)
                        if (finding.category == FarmerResultCategory.CHECK) {
                            Text(finding.label.replace('_', ' '), style = MaterialTheme.typography.bodyLarge)
                        }
                        if (expanded == finding.id) {
                            HorizontalDivider(Modifier.padding(vertical = 6.dp))
                            Text(stringResource(if (finding.saved) R.string.farmer_saved_record else R.string.farmer_live_record), style = MaterialTheme.typography.labelLarge)
                            Text(stringResource(if (finding.category == FarmerResultCategory.REVIEW) R.string.farmer_recheck_hint else R.string.farmer_result_note), style = MaterialTheme.typography.bodyMedium)
                        } else Text(stringResource(R.string.farmer_details), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.primary)
                    }
                }
            }
            if (map != null) {
                item {
                    TextButton(onClick = { showMap = !showMap }) { Text(stringResource(if (showMap) R.string.farmer_hide_map else R.string.farmer_row_map)) }
                    if (showMap) {
                        Text(stringResource(R.string.farmer_row_plan_note), style = MaterialTheme.typography.bodySmall)
                        FieldMapGrid(fieldMap = map, layout = null)
                    }
                }
            }
            item {
                OutlinedButton(onClick = { showExport = !showExport }, enabled = runId != null, modifier = Modifier.fillMaxWidth().heightIn(min = 52.dp)) {
                    Text(stringResource(R.string.farmer_save_report))
                }
                if (showExport) {
                    Column(Modifier.fillMaxWidth()) {
                        TextButton(onClick = { viewModel.exportPdf(runId) }) { Text(stringResource(R.string.export_pdf_short)) }
                        TextButton(onClick = { viewModel.exportCsv(runId) }) { Text(stringResource(R.string.export_csv_short)) }
                        TextButton(onClick = { viewModel.exportBundle(runId) }) { Text(stringResource(R.string.export_bundle_short)) }
                    }
                }
            }
            state.lastExportedFile?.let { file -> item { ExportActions(context, file) } }
        }
    }
}

@Composable
private fun SummaryLine(label: String, count: Int, color: Color) {
    Surface(color = color.copy(alpha = 0.08f), shape = RoundedCornerShape(12.dp)) {
        Row(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 12.dp)) {
            Text(label, Modifier.weight(1f), style = MaterialTheme.typography.bodyLarge, color = color)
            Text(count.toString(), style = MaterialTheme.typography.titleMedium, color = color)
        }
    }
}
