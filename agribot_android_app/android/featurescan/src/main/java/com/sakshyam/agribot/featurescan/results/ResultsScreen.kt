package com.sakshyam.agribot.featurescan.results

import android.content.Context
import android.content.Intent
import android.graphics.BitmapFactory
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.sakshyam.agribot.domain.model.RecordedDecision
import com.sakshyam.agribot.domain.model.RecordingMode
import com.sakshyam.agribot.domain.model.ReportText
import com.sakshyam.agribot.domain.model.RunFieldMapCell
import com.sakshyam.agribot.domain.scan.toVerdict
import com.sakshyam.agribot.featurescan.Conditions
import com.sakshyam.agribot.featurescan.R
import com.sakshyam.agribot.featurescan.ui.StatTile
import com.sakshyam.agribot.featurescan.ui.Tone
import com.sakshyam.agribot.featurescan.ui.VerdictCard
import com.sakshyam.agribot.featurescan.ui.color
import java.io.File
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.time.format.FormatStyle
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ResultsScreen(onBack: () -> Unit, vm: ResultsViewModel = hiltViewModel()) {
    val s by vm.state.collectAsStateWithLifecycle()
    val context = LocalContext.current
    var confirmDelete by remember { mutableStateOf(false) }
    LaunchedEffect(s.deleted) { if (s.deleted) onBack() }
    LaunchedEffect(s.share) { s.share?.let { share(context, File(it.absolutePath), it.contentType); vm.shareHandled() } }
    val report = reportText(s)

    Scaffold(topBar = {
        TopAppBar(
            title = { Text(stringResource(R.string.results_title)) },
            navigationIcon = { IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, stringResource(R.string.back)) } },
            actions = { if (s.run != null) IconButton(onClick = { confirmDelete = true }) { Icon(Icons.Filled.Delete, stringResource(R.string.delete)) } },
        )
    }) { pad ->
        Box(Modifier.padding(pad).fillMaxSize()) {
            when {
                s.loading -> CircularProgressIndicator(Modifier.align(Alignment.Center))
                s.missing -> Text(stringResource(R.string.result_missing), Modifier.padding(24.dp))
                else -> LazyColumn(Modifier.fillMaxSize().testTag("results"), contentPadding = androidx.compose.foundation.layout.PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    item { Text(formatTime(s.run!!.startedAt), style = MaterialTheme.typography.bodyMedium) }
                    if (s.run!!.mode == RecordingMode.CHECK_PLANT) {
                        s.plants.firstOrNull()?.let { d ->
                            item { val (v, labels) = d.toVerdict(); VerdictCard(v, labels) }
                            item { EvidencePhoto(d) }
                        }
                    } else {
                        item { SummaryRow(s) }
                        s.walkedM?.let { m -> item { Text(stringResource(R.string.gps_trail, m.toInt())) } }
                        s.map?.let { map ->
                            item { Text(stringResource(R.string.field_map), style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold) }
                            item { Legend() }
                            map.rows.forEach { row -> item { MapRow(stringResource(R.string.row_label, row.rowLabel), row.cells) } }
                        }
                        item { Text(stringResource(R.string.plants_needing_attention), style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold) }
                        if (s.attention.isEmpty()) item { Text(stringResource(R.string.none_need_attention)) }
                        s.attention.forEach { d -> item { AttentionCard(d) } }
                    }
                    item {
                        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            Button(onClick = { vm.exportPdf(report) }, modifier = Modifier.fillMaxWidth()) { Text(stringResource(R.string.export_pdf)) }
                            OutlinedButton(onClick = vm::exportCsv, modifier = Modifier.fillMaxWidth()) { Text(stringResource(R.string.export_csv)) }
                        }
                    }
                }
            }
            s.error?.let { Text(stringResource(R.string.export_failed, it), color = MaterialTheme.colorScheme.error, modifier = Modifier.align(Alignment.BottomCenter).padding(12.dp)) }
        }
    }
    if (confirmDelete) AlertDialog(
        onDismissRequest = { confirmDelete = false },
        text = { Text(stringResource(R.string.delete_confirm)) },
        confirmButton = { TextButton(onClick = { confirmDelete = false; vm.delete() }) { Text(stringResource(R.string.delete)) } },
        dismissButton = { TextButton(onClick = { confirmDelete = false }) { Text(stringResource(R.string.cancel)) } },
    )
}

@Composable
private fun SummaryRow(s: ResultsUiState) {
    val sum = s.summary ?: return
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        StatTile(sum.decisions.toString(), stringResource(R.string.summary_checked), Tone.NEUTRAL, Modifier.weight(1f))
        StatTile(sum.sick.toString(), stringResource(R.string.summary_need_attention), Tone.ATTENTION, Modifier.weight(1f))
        StatTile(sum.ok.toString(), stringResource(R.string.summary_healthy), Tone.HEALTHY, Modifier.weight(1f))
        StatTile(sum.uncertain.toString(), stringResource(R.string.summary_unsure), Tone.UNSURE, Modifier.weight(1f))
    }
}

private fun cellTone(c: RunFieldMapCell): Tone = when (c.status) {
    "ok" -> Tone.HEALTHY
    "sick" -> Tone.ATTENTION
    "uncertain", "skipped" -> Tone.UNSURE
    "manual" -> Tone.HEALTHY
    else -> Tone.NEUTRAL
}

@Composable
private fun Legend() {
    Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        listOf(Tone.HEALTHY to R.string.legend_healthy, Tone.ATTENTION to R.string.legend_attention, Tone.UNSURE to R.string.legend_unsure, Tone.NEUTRAL to R.string.legend_empty).forEach { (t, l) ->
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(Modifier.size(14.dp).background(if (t == Tone.NEUTRAL) t.color().copy(alpha = 0.3f) else t.color(), RoundedCornerShape(3.dp)))
                Text(" " + stringResource(l), style = MaterialTheme.typography.labelLarge)
            }
        }
    }
}

@Composable
private fun MapRow(label: String, cells: List<RunFieldMapCell>) {
    Column {
        Text(label, style = MaterialTheme.typography.labelLarge)
        Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(3.dp)) {
            cells.forEach { c ->
                val t = cellTone(c)
                Box(Modifier.size(width = 22.dp, height = 22.dp).background(if (c.status == "empty") t.color().copy(alpha = 0.3f) else t.color(), RoundedCornerShape(4.dp)), contentAlignment = Alignment.Center) {
                    Text(c.plantNumber.toString(), style = MaterialTheme.typography.labelSmall, color = androidx.compose.ui.graphics.Color.White)
                }
            }
        }
    }
}

@Composable
private fun AttentionCard(d: RecordedDecision) {
    Card(Modifier.fillMaxWidth()) {
        Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
            EvidencePhoto(d, Modifier.size(64.dp))
            Column(Modifier.padding(start = 12.dp)) {
                Text(stringResource(R.string.row_label, (d.rowIndex ?: 0).toString()) + " · " + stringResource(R.string.plant_label, d.plantNumber ?: 0), style = MaterialTheme.typography.labelLarge)
                val name = stringResource(Conditions.nameRes(d.label))
                Text(if (d.partialFinding == true) stringResource(R.string.verdict_disease_partial, name) else name, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold, color = Tone.ATTENTION.color())
                Text(stringResource(R.string.percent, (d.confidence * 100).toInt()) + " · " + stringResource(R.string.leaves_agree, d.leavesAgreeing ?: 0, d.leavesSeen ?: 0), style = MaterialTheme.typography.bodyMedium)
            }
        }
    }
}

@Composable
private fun EvidencePhoto(d: RecordedDecision, modifier: Modifier = Modifier.fillMaxWidth().height(220.dp)) {
    val path = d.evidencePath ?: return
    val bmp by produceState<android.graphics.Bitmap?>(null, path) {
        value = withContext(Dispatchers.IO) { runCatching { BitmapFactory.decodeFile(path, BitmapFactory.Options().apply { inSampleSize = 2 }) }.getOrNull() }
    }
    bmp?.let { Image(it.asImageBitmap(), contentDescription = null, contentScale = ContentScale.Crop, modifier = modifier) }
}

private fun formatTime(i: java.time.Instant): String =
    DateTimeFormatter.ofLocalizedDateTime(FormatStyle.MEDIUM, FormatStyle.SHORT).withZone(ZoneId.systemDefault()).format(i)

@Composable
private fun reportText(s: ResultsUiState): ReportText {
    val names = Conditions.knownKeys.associateWith { stringResource(Conditions.nameRes(it)) }
    val sum = s.summary
    return ReportText(
        title = stringResource(R.string.home_title) + " — " + stringResource(R.string.results_title),
        generatedAt = s.run?.let { formatTime(it.startedAt) }.orEmpty(),
        fieldLine = s.layout?.name ?: stringResource(R.string.run_item_check),
        summaryLine = if (sum == null) "" else listOf(
            stringResource(R.string.summary_checked) + ": " + sum.decisions,
            stringResource(R.string.summary_need_attention) + ": " + sum.sick,
            stringResource(R.string.summary_healthy) + ": " + sum.ok,
            stringResource(R.string.summary_unsure) + ": " + sum.uncertain,
        ).joinToString("   "),
        columns = listOf(R.string.report_col_row, R.string.report_col_plant, R.string.report_col_result, R.string.report_col_conf, R.string.report_col_leaves).map { stringResource(it) },
        labelNames = names,
        unsure = stringResource(R.string.summary_unsure),
        notSeen = stringResource(R.string.walk_not_seen),
        partial = stringResource(R.string.report_partial),
        disclaimer = stringResource(R.string.disclaimer_short),
    )
}

private fun share(context: Context, file: File, type: String) {
    val uri = FileProvider.getUriForFile(context, context.packageName + ".fileprovider", file)
    val send = Intent(Intent.ACTION_SEND).setType(type).putExtra(Intent.EXTRA_STREAM, uri).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    context.startActivity(Intent.createChooser(send, null).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
}
