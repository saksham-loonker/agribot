package com.sakshyam.agribot.featurescan

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.Eco
import androidx.compose.material.icons.filled.VerifiedUser
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.sakshyam.agribot.designsystem.AgribotTheme
import com.sakshyam.agribot.featurescan.check.CheckPlantScreen
import com.sakshyam.agribot.featurescan.home.HistoryScreen
import com.sakshyam.agribot.featurescan.home.HomeScreen
import com.sakshyam.agribot.featurescan.results.ResultsScreen
import com.sakshyam.agribot.featurescan.settings.SettingsScreen
import com.sakshyam.agribot.featurescan.settings.SettingsViewModel
import com.sakshyam.agribot.featurescan.walk.WalkRowsScreen

/** App root: theme from settings, onboarding gate, and navigation between screens. */
@Composable
fun AgribotApp(settingsVm: SettingsViewModel = hiltViewModel()) {
    val settings by settingsVm.settings.collectAsStateWithLifecycle()
    val s = settings
    AgribotTheme(darkTheme = s?.darkMode ?: false, sunMode = s?.sunMode ?: false) {
        Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
            when {
                s == null -> Unit
                s.showOnboarding -> Onboarding { settingsVm.update { it.copy(showOnboarding = false) } }
                else -> AppNav(defaultStride = s.stepLengthM.toDouble())
            }
        }
    }
}

@Composable
private fun AppNav(defaultStride: Double) {
    val nav = rememberNavController()
    NavHost(nav, startDestination = "home") {
        composable("home") {
            HomeScreen(
                onCheck = { nav.navigate("check") },
                onWalk = { nav.navigate("walk") },
                onSettings = { nav.navigate("settings") },
                onOpenRun = { nav.navigate("results/$it") },
                onSeeAll = { nav.navigate("history") },
            )
        }
        composable("check") { CheckPlantScreen(onBack = { nav.popBackStack() }, onOpenResult = { nav.navigate("results/$it") }) }
        composable("walk") {
            WalkRowsScreen(
                onBack = { nav.popBackStack() },
                onFinished = { id -> nav.navigate("results/$id") { popUpTo("home") } },
                defaultStride = defaultStride,
            )
        }
        composable("results/{runId}", arguments = listOf(navArgument("runId") { type = NavType.StringType })) {
            ResultsScreen(onBack = { nav.popBackStack() })
        }
        composable("history") { HistoryScreen(onBack = { nav.popBackStack() }, onOpenRun = { nav.navigate("results/$it") }) }
        composable("settings") { SettingsScreen(onBack = { nav.popBackStack() }) }
    }
}

@Composable
private fun Onboarding(onDone: () -> Unit) {
    var page by rememberSaveable { mutableIntStateOf(0) }
    val pages = listOf(
        Triple(Icons.Filled.Eco, R.string.onb_1_title, R.string.onb_1_body),
        Triple(Icons.Filled.CameraAlt, R.string.onb_2_title, R.string.onb_2_body),
        Triple(Icons.Filled.VerifiedUser, R.string.onb_3_title, R.string.onb_3_body),
    )
    val (icon, title, body) = pages[page]
    Column(Modifier.fillMaxSize().padding(28.dp), verticalArrangement = Arrangement.spacedBy(20.dp, Alignment.CenterVertically), horizontalAlignment = Alignment.CenterHorizontally) {
        Icon(icon, contentDescription = null, modifier = Modifier.size(88.dp), tint = MaterialTheme.colorScheme.primary)
        Text(stringResource(title), style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold, textAlign = TextAlign.Center)
        Text(stringResource(body), style = MaterialTheme.typography.bodyLarge)
        Button(
            onClick = { if (page < pages.lastIndex) page++ else onDone() },
            modifier = Modifier.fillMaxWidth().height(56.dp).testTag("onboarding_next"),
        ) { Text(stringResource(if (page < pages.lastIndex) R.string.onb_next else R.string.onb_start)) }
    }
}
