package com.sakshyam.agribot.featurescan.settings

import androidx.appcompat.app.AppCompatDelegate
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.os.LocaleListCompat
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.ViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewModelScope
import com.sakshyam.agribot.domain.model.ScanSettings
import com.sakshyam.agribot.domain.repository.ScanSettingsRepository
import com.sakshyam.agribot.domain.scan.LeafVision
import com.sakshyam.agribot.domain.scan.VisionStatus
import com.sakshyam.agribot.featurescan.R
import dagger.hilt.android.lifecycle.HiltViewModel
import java.util.Locale
import javax.inject.Inject
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

@HiltViewModel
class SettingsViewModel @Inject constructor(private val repo: ScanSettingsRepository, vision: LeafVision) : ViewModel() {
    val settings: StateFlow<ScanSettings?> = repo.observeSettings().stateIn(viewModelScope, SharingStarted.Eagerly, null)
    val vision: StateFlow<VisionStatus> = vision.status
    fun update(f: (ScanSettings) -> ScanSettings) {
        viewModelScope.launch { repo.saveSettings(f(repo.observeSettings().first())) }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(onBack: () -> Unit, vm: SettingsViewModel = hiltViewModel()) {
    val s by vm.settings.collectAsStateWithLifecycle()
    val vision by vm.vision.collectAsStateWithLifecycle()
    Scaffold(topBar = {
        TopAppBar(
            title = { Text(stringResource(R.string.settings)) },
            navigationIcon = { IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, stringResource(R.string.back)) } },
        )
    }) { pad ->
        val settings = s ?: return@Scaffold
        Column(Modifier.padding(pad).fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
            Section(stringResource(R.string.settings_language))
            val current = AppCompatDelegate.getApplicationLocales().toLanguageTags().substringBefore('-').ifEmpty { "" }
            listOf("" to R.string.lang_system, "en" to R.string.lang_en, "hi" to R.string.lang_hi).forEach { (tag, label) ->
                Row(
                    Modifier.fillMaxWidth().selectable(selected = current == tag, onClick = {
                        AppCompatDelegate.setApplicationLocales(if (tag.isEmpty()) LocaleListCompat.getEmptyLocaleList() else LocaleListCompat.forLanguageTags(tag))
                    }).testTag("lang_$tag"),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    RadioButton(selected = current == tag, onClick = null)
                    Text(stringResource(label), Modifier.padding(start = 8.dp), style = MaterialTheme.typography.bodyLarge)
                }
            }
            HorizontalDivider()
            Toggle(stringResource(R.string.settings_sun), stringResource(R.string.settings_sun_body), settings.sunMode) { v -> vm.update { it.copy(sunMode = v) } }
            Toggle(stringResource(R.string.settings_dark), null, settings.darkMode) { v -> vm.update { it.copy(darkMode = v) } }
            Toggle(stringResource(R.string.settings_save_photos), stringResource(R.string.settings_save_photos_body), settings.saveEvidenceFrames) { v -> vm.update { it.copy(saveEvidenceFrames = v) } }
            HorizontalDivider()
            var stride by remember(settings.stepLengthM) { mutableFloatStateOf(settings.stepLengthM) }
            Text(stringResource(R.string.settings_stride, String.format(Locale.US, "%.2f", stride)), style = MaterialTheme.typography.titleMedium)
            Slider(value = stride, onValueChange = { stride = it }, valueRange = 0.3f..1.2f, steps = 17, onValueChangeFinished = { vm.update { it.copy(stepLengthM = stride) } })
            Text(stringResource(R.string.stride_help), style = MaterialTheme.typography.bodySmall)
            HorizontalDivider()
            Section(stringResource(R.string.settings_model))
            (vision as? VisionStatus.Ready)?.info?.let { Text(stringResource(R.string.settings_model_info, it.bundleId, it.labels.size - 1)) }
            Section(stringResource(R.string.settings_about))
            Text(stringResource(R.string.about_body), style = MaterialTheme.typography.bodyMedium)
        }
    }
}

@Composable
private fun Section(title: String) = Text(title, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)

@Composable
private fun Toggle(title: String, body: String?, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.titleMedium)
            body?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
        }
        Switch(checked = checked, onCheckedChange = onChange)
    }
}
