package com.sakshyam.agribot

import android.Manifest
import androidx.appcompat.app.AppCompatDelegate
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.hasTestTag
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTextReplacement
import androidx.core.os.LocaleListCompat
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import org.junit.After
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class AgribotLaunchUiTest {
    @get:Rule(order = 0)
    val permissions: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)

    @get:Rule(order = 1)
    val ui = createAndroidComposeRule<MainActivity>()

    private fun str(id: Int, vararg args: Any) = InstrumentationRegistry.getInstrumentation().targetContext.getString(id, *args)

    @Before fun reachHome() {
        ui.waitUntil(10_000) { ui.onAllNodes(hasTestTag("onboarding_next")).fetchSemanticsNodes().isNotEmpty() || ui.onAllNodes(hasTestTag("mode_check")).fetchSemanticsNodes().isNotEmpty() }
        repeat(3) {
            if (ui.onAllNodes(hasTestTag("onboarding_next")).fetchSemanticsNodes().isNotEmpty()) ui.onNodeWithTag("onboarding_next").performClick()
            ui.waitForIdle()
        }
        ui.waitUntil(10_000) { ui.onAllNodes(hasTestTag("mode_check")).fetchSemanticsNodes().isNotEmpty() }
    }

    @After fun resetLanguage() {
        ui.runOnUiThread { AppCompatDelegate.setApplicationLocales(LocaleListCompat.getEmptyLocaleList()) }
    }

    @Test fun homeShowsBothModesAndRecentSection() {
        ui.onNodeWithTag("mode_check").assertIsDisplayed()
        ui.onNodeWithTag("mode_walk").assertIsDisplayed()
        ui.onNodeWithText(str(com.sakshyam.agribot.featurescan.R.string.recent_title)).assertIsDisplayed()
    }

    @Test fun checkPlantOpensCameraWithGuidance() {
        ui.onNodeWithTag("mode_check").performClick()
        ui.waitUntil(15_000) { ui.onAllNodes(hasTestTag("hint")).fetchSemanticsNodes().isNotEmpty() }
        ui.onNodeWithTag("hint").assertIsDisplayed()
    }

    @Test fun walkSetupRejectsInvalidNumbers() {
        ui.onNodeWithTag("mode_walk").performClick()
        val plantsLabel = str(com.sakshyam.agribot.featurescan.R.string.plants_per_row)
        ui.waitUntil(10_000) { ui.onAllNodes(hasText(plantsLabel)).fetchSemanticsNodes().isNotEmpty() }
        ui.onNode(hasText(plantsLabel).and(androidx.compose.ui.test.hasSetTextAction())).performTextReplacement("0")
        ui.onNodeWithText(str(com.sakshyam.agribot.featurescan.R.string.invalid_number, "1", "1000")).assertIsDisplayed()
        ui.onNodeWithTag("start_walk").assertIsNotEnabled()
    }

    @Test fun settingsSwitchesLanguageToHindiAndBack() {
        ui.onNodeWithTag("open_settings").performClick()
        ui.waitUntil(10_000) { ui.onAllNodes(hasTestTag("lang_hi")).fetchSemanticsNodes().isNotEmpty() }
        ui.onNodeWithTag("lang_hi").performClick()
        ui.waitUntil(15_000) { ui.onAllNodesWithText("भाषा").fetchSemanticsNodes().isNotEmpty() }
        ui.onNodeWithTag("lang_en").performClick()
        ui.waitUntil(15_000) { ui.onAllNodesWithText("Language").fetchSemanticsNodes().isNotEmpty() }
    }
}
