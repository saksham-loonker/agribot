package com.sakshyam.agribot.featurescan.walk

import android.Manifest
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.sakshyam.agribot.camera.LeafCamera
import com.sakshyam.agribot.domain.scan.VerdictKind
import com.sakshyam.agribot.domain.scan.VisionStatus
import com.sakshyam.agribot.featurescan.R
import com.sakshyam.agribot.featurescan.ui.CameraPermissionGate
import com.sakshyam.agribot.featurescan.ui.LeafOverlay
import com.sakshyam.agribot.featurescan.ui.Tone
import com.sakshyam.agribot.featurescan.ui.color
import com.sakshyam.agribot.featurescan.ui.hasPermission
import com.sakshyam.agribot.featurescan.ui.tone
import com.sakshyam.agribot.featurescan.ui.verdictTitle

/** Validated numeric field input. Returns the parsed value or null. */
internal fun parseInRange(text: String, min: Double, max: Double): Double? =
    text.trim().replace(',', '.').toDoubleOrNull()?.takeIf { it in min..max }

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun WalkRowsScreen(onBack: () -> Unit, onFinished: (String) -> Unit, defaultStride: Double, vm: WalkRowsViewModel = hiltViewModel()) {
    val s by vm.state.collectAsStateWithLifecycle()
    LaunchedEffect(s.finished, s.runId) { if (s.finished) s.runId?.let(onFinished) }
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(if (s.started) R.string.mode_walk_title else R.string.walk_setup_title)) },
                navigationIcon = { IconButton(onClick = { if (s.started && !s.finished) vm.finish() else onBack() }) { Icon(Icons.AutoMirrored.Filled.ArrowBack, stringResource(R.string.back)) } },
            )
        },
    ) { pad ->
        Box(Modifier.padding(pad).fillMaxSize()) {
            if (!s.started) WalkSetupForm(defaultStride, s.vision, vm.stepsSupported, onStart = vm::start)
            else CameraPermissionGate { WalkLive(s, vm) }
        }
    }
}

@Composable
private fun WalkSetupForm(defaultStride: Double, vision: VisionStatus, stepsSupported: Boolean, onStart: (WalkSetup) -> Unit) {
    val context = LocalContext.current
    val defaultName = stringResource(R.string.field_name_default)
    var name by rememberSaveable { mutableStateOf(defaultName) }
    var rows by rememberSaveable { mutableStateOf("4") }
    var plants by rememberSaveable { mutableStateOf("20") }
    var spacing by rememberSaveable { mutableStateOf("0.6") }
    var stride by rememberSaveable { mutableStateOf("%.2f".format(java.util.Locale.US, defaultStride)) }
    var asked by remember { mutableStateOf(false) }
    val perms = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { asked = true }
    val needsSteps = stepsSupported && Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q && !context.hasPermission(Manifest.permission.ACTIVITY_RECOGNITION)
    val needsLocation = !context.hasPermission(Manifest.permission.ACCESS_FINE_LOCATION)

    val r = parseInRange(rows, 1.0, 200.0)?.takeIf { it % 1.0 == 0.0 }
    val p = parseInRange(plants, 1.0, 1000.0)?.takeIf { it % 1.0 == 0.0 }
    val sp = parseInRange(spacing, 0.1, 5.0)
    val st = parseInRange(stride, 0.2, 2.0)
    val valid = name.isNotBlank() && r != null && p != null && sp != null && st != null

    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        OutlinedTextField(name, { name = it.take(40) }, label = { Text(stringResource(R.string.field_name)) }, singleLine = true, modifier = Modifier.fillMaxWidth())
        NumberField(rows, { rows = it }, R.string.rows_count, r != null, "1", "200", KeyboardType.Number)
        NumberField(plants, { plants = it }, R.string.plants_per_row, p != null, "1", "1000", KeyboardType.Number)
        NumberField(spacing, { spacing = it }, R.string.plant_spacing, sp != null, "0.1", "5", KeyboardType.Decimal)
        NumberField(stride, { stride = it }, R.string.stride, st != null, "0.2", "2", KeyboardType.Decimal)
        Text(stringResource(R.string.stride_help), style = MaterialTheme.typography.bodySmall)
        if ((needsSteps || needsLocation) && !asked) {
            Text(stringResource(if (needsSteps) R.string.perm_steps_body else R.string.perm_location_body), style = MaterialTheme.typography.bodyMedium)
            OutlinedButton(onClick = {
                val list = buildList {
                    if (needsSteps) add(Manifest.permission.ACTIVITY_RECOGNITION)
                    if (needsLocation) { add(Manifest.permission.ACCESS_FINE_LOCATION); add(Manifest.permission.ACCESS_COARSE_LOCATION) }
                }
                perms.launch(list.toTypedArray())
            }, modifier = Modifier.fillMaxWidth()) { Text(stringResource(if (needsSteps) R.string.perm_steps_grant else R.string.perm_location_grant)) }
        }
        Button(
            onClick = { onStart(WalkSetup(name.trim(), r!!.toInt(), p!!.toInt(), sp!!, st!!)) },
            enabled = valid && vision is VisionStatus.Ready,
            modifier = Modifier.fillMaxWidth().height(56.dp).testTag("start_walk"),
        ) { Text(stringResource(if (vision is VisionStatus.Ready) R.string.start_walking else R.string.model_loading)) }
    }
}

@Composable
private fun NumberField(value: String, onChange: (String) -> Unit, label: Int, ok: Boolean, min: String, max: String, type: KeyboardType) {
    OutlinedTextField(
        value, { onChange(it.take(8)) }, label = { Text(stringResource(label)) }, singleLine = true, isError = !ok,
        supportingText = { if (!ok) Text(stringResource(R.string.invalid_number, min, max)) },
        keyboardOptions = KeyboardOptions(keyboardType = type), modifier = Modifier.fillMaxWidth(),
    )
}

@Composable
private fun WalkLive(s: WalkUiState, vm: WalkRowsViewModel) {
    Column(Modifier.fillMaxSize()) {
        Box(Modifier.weight(1f).fillMaxWidth()) {
            if (!s.paused) {
                LeafCamera(Modifier.fillMaxSize(), isReady = vm::isReady, onFrame = vm::onFrame)
                LeafOverlay(s.boxes, Modifier.fillMaxSize())
            } else {
                Text(stringResource(R.string.walk_paused), style = MaterialTheme.typography.headlineMedium, modifier = Modifier.align(Alignment.Center))
            }
            Column(Modifier.align(Alignment.TopCenter).padding(10.dp).background(Color(0xCC000000), RoundedCornerShape(14.dp)).padding(12.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                Text(stringResource(R.string.walk_row_plant, s.row, s.plant, s.plantsPerRow), color = Color.White, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold, modifier = Modifier.testTag("row_plant"))
                val v = s.currentVerdict
                Text(
                    if (v != null) stringResource(R.string.walk_current_plant) + ": " + verdictTitle(v, s.labels) else stringResource(R.string.leaves_seen, s.currentLeaves),
                    color = Color.White, style = MaterialTheme.typography.bodyLarge,
                )
                if (!s.stepsAvailable) Text(stringResource(R.string.walk_steps_unavailable), color = Color(0xFFFFE066), style = MaterialTheme.typography.bodyMedium)
                if (s.plant == s.plantsPerRow) Text(stringResource(if (s.row == s.rows) R.string.walk_all_done else R.string.walk_row_end, s.row), color = Color(0xFFFFE066))
            }
        }
        RowStrip(s.rowResults, s.plant)
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            s.lastSaved?.let { (plant, v) -> Text(stringResource(R.string.walk_plant_saved, plant, if (v.leavesSeen == 0) stringResource(R.string.walk_not_seen) else verdictTitle(v, s.labels)), style = MaterialTheme.typography.bodyMedium) }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(onClick = vm::previousPlant, enabled = s.plant > 1, modifier = Modifier.weight(1f).height(56.dp)) { Text(stringResource(R.string.walk_prev_plant)) }
                Button(onClick = vm::nextPlant, enabled = s.plant < s.plantsPerRow, modifier = Modifier.weight(2f).height(56.dp).testTag("next_plant")) { Text(stringResource(R.string.walk_next_plant)) }
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilledTonalButton(onClick = vm::togglePause, modifier = Modifier.weight(1f)) { Text(stringResource(if (s.paused) R.string.walk_resume else R.string.walk_pause)) }
                FilledTonalButton(onClick = vm::nextRow, enabled = s.row < s.rows, modifier = Modifier.weight(1f)) { Text(stringResource(R.string.walk_next_row)) }
                Button(onClick = vm::finish, modifier = Modifier.weight(1f).testTag("finish_walk")) { Text(stringResource(R.string.walk_finish)) }
            }
        }
    }
}

/** One coloured cell per plant in the current row. */
@Composable
private fun RowStrip(results: List<VerdictKind?>, current: Int) {
    if (results.isEmpty()) return
    val shown = if (results.size <= 40) results.indices else ((current - 20).coerceAtLeast(0) until (current + 20).coerceAtMost(results.size))
    Row(Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 6.dp), horizontalArrangement = Arrangement.spacedBy(2.dp)) {
        for (i in shown) {
            val tone = results[i]?.tone() ?: Tone.NEUTRAL
            Box(Modifier.weight(1f).height(if (i + 1 == current) 18.dp else 12.dp).background(if (results[i] == null) tone.color().copy(alpha = 0.3f) else tone.color(), RoundedCornerShape(3.dp)))
        }
    }
    Spacer(Modifier.height(2.dp))
}
