package com.sakshyam.agribot.featurescan.home

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.DirectionsWalk
import androidx.compose.material.icons.filled.Eco
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.foundation.background
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.repeatOnLifecycle
import com.sakshyam.agribot.domain.model.RecordingMode
import com.sakshyam.agribot.domain.scan.VisionStatus
import com.sakshyam.agribot.domain.scan.toVerdict
import com.sakshyam.agribot.featurescan.R
import com.sakshyam.agribot.featurescan.ui.ModeCard
import com.sakshyam.agribot.featurescan.ui.color
import com.sakshyam.agribot.featurescan.ui.tone
import com.sakshyam.agribot.featurescan.ui.verdictTitle
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.time.format.FormatStyle

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(
    onCheck: () -> Unit,
    onWalk: () -> Unit,
    onSettings: () -> Unit,
    onOpenRun: (String) -> Unit,
    onSeeAll: () -> Unit,
    vm: HomeViewModel = hiltViewModel(),
) {
    val items by vm.items.collectAsStateWithLifecycle()
    val vision by vm.visionStatus.collectAsStateWithLifecycle()
    RefreshOnResume(vm)
    Scaffold(topBar = {
        TopAppBar(
            title = { Text(stringResource(R.string.home_title), fontWeight = FontWeight.Bold) },
            actions = { IconButton(onClick = onSettings, modifier = Modifier.testTag("open_settings")) { Icon(Icons.Filled.Settings, stringResource(R.string.settings)) } },
        )
    }) { pad ->
        LazyColumn(Modifier.padding(pad).fillMaxSize(), contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            item { Text(stringResource(R.string.home_subtitle), style = MaterialTheme.typography.bodyLarge) }
            item { VisionBanner(vision, vm::retryModels) }
            item { ModeCard(stringResource(R.string.mode_check_title), stringResource(R.string.mode_check_body), Icons.Filled.Eco, onCheck, Modifier.testTag("mode_check")) }
            item { ModeCard(stringResource(R.string.mode_walk_title), stringResource(R.string.mode_walk_body), Icons.AutoMirrored.Filled.DirectionsWalk, onWalk, Modifier.testTag("mode_walk")) }
            item {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(stringResource(R.string.recent_title), style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold, modifier = Modifier.weight(1f))
                    if ((items?.size ?: 0) > 5) TextButton(onClick = onSeeAll) { Text(stringResource(R.string.see_all)) }
                }
            }
            val list = items
            if (list != null && list.isEmpty()) item { Text(stringResource(R.string.recent_empty), modifier = Modifier.testTag("recent_empty")) }
            items(list.orEmpty().take(5), key = { it.run.runId.value }) { RunRow(it) { onOpenRun(it.run.runId.value) } }
        }
    }
}

@Composable
private fun RefreshOnResume(vm: HomeViewModel) {
    val lifecycle = LocalLifecycleOwner.current.lifecycle
    LaunchedEffect(lifecycle) { lifecycle.repeatOnLifecycle(Lifecycle.State.RESUMED) { vm.refresh() } }
}

@Composable
private fun VisionBanner(vision: VisionStatus, onRetry: () -> Unit) {
    when (vision) {
        is VisionStatus.Loading -> Text(stringResource(R.string.model_loading), style = MaterialTheme.typography.bodyMedium)
        is VisionStatus.Failed -> Card {
            Column(Modifier.padding(12.dp)) {
                Text(stringResource(R.string.model_failed, vision.reason), color = MaterialTheme.colorScheme.error)
                TextButton(onClick = onRetry) { Text(stringResource(R.string.retry)) }
            }
        }
        is VisionStatus.Ready -> Unit
    }
}

@Composable
fun RunRow(item: RunListItem, onClick: () -> Unit) {
    val time = DateTimeFormatter.ofLocalizedDateTime(FormatStyle.MEDIUM, FormatStyle.SHORT).withZone(ZoneId.systemDefault()).format(item.run.startedAt)
    Card(Modifier.fillMaxWidth().clickable(onClick = onClick)) {
        Row(Modifier.padding(14.dp), verticalAlignment = Alignment.CenterVertically) {
            val first = item.first
            val tone = if (item.run.mode == RecordingMode.CHECK_PLANT && first != null) first.toVerdict().first.kind.tone()
            else if (item.summary.sick > 0) com.sakshyam.agribot.featurescan.ui.Tone.ATTENTION else com.sakshyam.agribot.featurescan.ui.Tone.HEALTHY
            Box(Modifier.size(14.dp).clip(CircleShape).background(tone.color()))
            Column(Modifier.padding(start = 12.dp)) {
                if (item.run.mode == RecordingMode.CHECK_PLANT && first != null) {
                    val (v, labels) = first.toVerdict()
                    Text(verdictTitle(v, labels), style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                    Text(stringResource(R.string.run_item_check) + " · " + time, style = MaterialTheme.typography.bodyMedium)
                } else {
                    Text(stringResource(R.string.run_item_walk, item.fieldName ?: ""), style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                    Text(stringResource(R.string.run_item_plants, item.summary.decisions, item.summary.sick) + " · " + time, style = MaterialTheme.typography.bodyMedium)
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HistoryScreen(onBack: () -> Unit, onOpenRun: (String) -> Unit, vm: HomeViewModel = hiltViewModel()) {
    val items by vm.items.collectAsStateWithLifecycle()
    RefreshOnResume(vm)
    Scaffold(topBar = {
        TopAppBar(
            title = { Text(stringResource(R.string.history_title)) },
            navigationIcon = { IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, stringResource(R.string.back)) } },
        )
    }) { pad ->
        LazyColumn(Modifier.padding(pad).fillMaxSize(), contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            items(items.orEmpty(), key = { it.run.runId.value }) { RunRow(it) { onOpenRun(it.run.runId.value) } }
        }
    }
}
