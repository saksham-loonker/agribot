package com.sakshyam.agribot

import android.graphics.Bitmap
import androidx.activity.ComponentActivity
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asAndroidBitmap
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.test.captureToImage
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.unit.dp
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.sakshyam.agribot.designsystem.AgribotTheme
import com.sakshyam.agribot.domain.scan.PlantVerdict
import com.sakshyam.agribot.domain.scan.VerdictKind
import com.sakshyam.agribot.featurescan.ui.VerdictCard
import java.io.File
import java.util.Locale
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/** Renders result cards in English and Hindi and saves them for visual review (PNG files in the app files dir, folder "render"). */
@RunWith(AndroidJUnit4::class)
class VerdictCardRenderTest {
    @get:Rule val ui = createAndroidComposeRule<ComponentActivity>()

    private val labels = listOf("Late_blight", "Healthy")
    private val cases = listOf(
        "disease" to PlantVerdict(VerdictKind.DISEASE, 0, 0.86f, 1, 0.08f, 5, 4, 6, false),
        "partial" to PlantVerdict(VerdictKind.DISEASE, 0, 0.78f, 1, 0.70f, 6, 2, 6, true),
        "unsure" to PlantVerdict(VerdictKind.UNSURE, 0, 0.45f, 1, 0.40f, 3, 2, 8, false),
    )

    @Test fun renderEnglishAndHindiCards() {
        var current by mutableIntStateOf(0)
        var lang by mutableStateOf("en")
        ui.setContent {
            val base = LocalContext.current
            val cfg = android.content.res.Configuration(LocalConfiguration.current).apply { setLocale(Locale(lang)) }
            val ctx = base.createConfigurationContext(cfg)
            CompositionLocalProvider(LocalContext provides ctx, LocalConfiguration provides cfg) {
                AgribotTheme {
                    Surface(color = MaterialTheme.colorScheme.background) {
                        Box(Modifier.verticalScroll(rememberScrollState()).padding(12.dp).testTag("card")) {
                            VerdictCard(cases[current].second, labels)
                        }
                    }
                }
            }
        }
        val dir = File(ui.activity.filesDir, "render").apply { mkdirs() }
        for (l in listOf("en", "hi")) for (i in cases.indices) {
            ui.runOnUiThread { lang = l; current = i }
            ui.waitForIdle()
            val bmp = ui.onNodeWithTag("card").captureToImage().asAndroidBitmap()
            File(dir, "${l}_${cases[i].first}.png").outputStream().use { bmp.compress(Bitmap.CompressFormat.PNG, 100, it) }
        }
    }
}
