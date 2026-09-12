package com.sakshyam.agribot.featurescan

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

@Composable
fun OnboardingScreen(onOnboardingComplete: () -> Unit) {
    Column(Modifier.fillMaxSize().safeDrawingPadding()) {
        FarmerHeader(stringResource(R.string.farmer_home_title), stringResource(R.string.farmer_offline)) {
            TextButton(onClick = onOnboardingComplete) { Text(stringResource(R.string.skip_setup)) }
        }
        Column(
            Modifier.weight(1f).verticalScroll(rememberScrollState()).padding(24.dp),
            verticalArrangement = Arrangement.spacedBy(24.dp),
        ) {
            Text(stringResource(R.string.farmer_scan_heading), style = MaterialTheme.typography.headlineLarge, fontWeight = FontWeight.Bold)
            Text(stringResource(R.string.farmer_scan_intro), style = MaterialTheme.typography.bodyLarge)
            listOf(
                R.string.walk_and_scan to R.string.farmer_side_description,
                R.string.farmer_findings to R.string.farmer_result_note,
                R.string.farmer_saved_scans to R.string.farmer_no_scans,
            ).forEachIndexed { index, (title, body) ->
                Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                    Text("0${index + 1}", color = MaterialTheme.colorScheme.primary, style = MaterialTheme.typography.titleLarge)
                    Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        Text(stringResource(title), style = MaterialTheme.typography.titleMedium)
                        Text(stringResource(body), style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
            }
        }
        Button(onClick = onOnboardingComplete, modifier = Modifier.fillMaxWidth().padding(20.dp).heightIn(min = 56.dp)) {
            Text(stringResource(R.string.open_agribot), style = MaterialTheme.typography.titleMedium)
        }
    }
}
