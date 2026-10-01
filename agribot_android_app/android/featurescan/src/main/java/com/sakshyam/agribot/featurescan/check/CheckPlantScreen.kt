package com.sakshyam.agribot.featurescan.check

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.FlashlightOff
import androidx.compose.material.icons.filled.FlashlightOn
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.sakshyam.agribot.camera.LeafCamera
import com.sakshyam.agribot.domain.scan.VisionStatus
import com.sakshyam.agribot.featurescan.R
import com.sakshyam.agribot.featurescan.ui.CameraPermissionGate
import com.sakshyam.agribot.featurescan.ui.LeafOverlay
import com.sakshyam.agribot.featurescan.ui.VerdictCard

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CheckPlantScreen(onBack: () -> Unit, onOpenResult: (String) -> Unit, vm: CheckPlantViewModel = hiltViewModel()) {
    val s by vm.state.collectAsStateWithLifecycle()
    var torch by remember { mutableStateOf(false) }
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.mode_check_title)) },
                navigationIcon = { IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, stringResource(R.string.back)) } },
                actions = {
                    if (s.scanning) IconButton(onClick = { torch = !torch }) {
                        Icon(if (torch) Icons.Filled.FlashlightOff else Icons.Filled.FlashlightOn, stringResource(if (torch) R.string.torch_off else R.string.torch_on))
                    }
                },
            )
        },
    ) { pad ->
        Box(Modifier.padding(pad).fillMaxSize()) {
            val verdict = s.verdict
            when {
                s.vision is VisionStatus.Failed -> Text(
                    stringResource(R.string.model_failed, (s.vision as VisionStatus.Failed).reason),
                    modifier = Modifier.padding(24.dp), color = MaterialTheme.colorScheme.error,
                )
                verdict != null -> Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    VerdictCard(verdict, s.labels, Modifier.testTag("verdict"))
                    if (s.savedRunId == null) {
                        Button(onClick = vm::save, enabled = !s.saving, modifier = Modifier.fillMaxWidth().testTag("save")) { Text(stringResource(R.string.save_result)) }
                    } else {
                        Button(onClick = { onOpenResult(s.savedRunId!!) }, modifier = Modifier.fillMaxWidth()) { Text(stringResource(R.string.saved) + " ✓") }
                    }
                    OutlinedButton(onClick = vm::checkAgain, modifier = Modifier.fillMaxWidth()) {
                        Text(stringResource(if (s.savedRunId == null) R.string.check_again else R.string.check_another))
                    }
                }
                else -> CameraPermissionGate {
                    Box(Modifier.fillMaxSize()) {
                        LeafCamera(Modifier.fillMaxSize(), torchOn = torch, isReady = vm::isReady, onFrame = vm::onFrame)
                        LeafOverlay(s.boxes, Modifier.fillMaxSize())
                        HintBanner(s, Modifier.align(Alignment.TopCenter).padding(12.dp))
                        ProgressFooter(s, Modifier.align(Alignment.BottomCenter).padding(16.dp))
                    }
                }
            }
            s.error?.let { e ->
                Text(e, color = MaterialTheme.colorScheme.error, modifier = Modifier.align(Alignment.BottomCenter).padding(8.dp))
            }
        }
    }
}

@Composable
private fun HintBanner(s: CheckUiState, modifier: Modifier) {
    val text = when {
        s.vision is VisionStatus.Loading -> stringResource(R.string.model_loading)
        else -> stringResource(
            when (s.hint) {
                CheckHint.FIND_LEAVES -> R.string.check_hint_find
                CheckHint.MOVE_CLOSER -> R.string.check_hint_closer
                CheckHint.HOLD_STEADY -> R.string.check_hint_hold
                CheckHint.TOO_DARK -> R.string.check_hint_dark
                CheckHint.NO_LEAVES -> R.string.check_no_leaves
            },
        )
    }
    Text(
        text, color = Color.White, style = MaterialTheme.typography.titleMedium,
        modifier = modifier.background(Color(0xCC000000), RoundedCornerShape(14.dp)).padding(horizontal = 16.dp, vertical = 10.dp).testTag("hint"),
    )
}

@Composable
private fun ProgressFooter(s: CheckUiState, modifier: Modifier) {
    Row(
        modifier.background(Color(0xCC000000), RoundedCornerShape(14.dp)).padding(12.dp),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        CircularProgressIndicator(progress = { s.progress }, modifier = Modifier.size(36.dp), color = Color.White, trackColor = Color(0x55FFFFFF))
        Text(stringResource(R.string.check_progress, s.leavesSeen, s.framesUsed), color = Color.White, style = MaterialTheme.typography.bodyLarge)
    }
}
