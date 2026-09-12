package com.sakshyam.agribot

import android.Manifest
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onFirst
import androidx.compose.ui.test.onLast
import androidx.compose.ui.test.performClick
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.rule.GrantPermissionRule
import androidx.lifecycle.Lifecycle
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class AgribotLaunchUiTest {
    @get:Rule(order = 0)
    val permissions = GrantPermissionRule.grant(Manifest.permission.CAMERA)
    @get:Rule(order = 1)
    val ui = createAndroidComposeRule<MainActivity>()

    @Before fun home() {
        ui.waitForIdle()
        if (has("Skip setup")) click("Skip setup")
        if (has("Back") && has("Scan findings")) click("Back")
        if (has("Finish scan")) {
            click("Finish scan")
            ui.onAllNodesWithText("Finish scan").onLast().performClick()
        }
        ui.waitUntil(10_000) { has("Agribot") }
    }
    @Test fun primaryScanActionAndChoicesAreVisible() {
        visible("Check your crop")
        visible("Walk & scan")
        visible("Burst overview")
        visible("Start field scan")
    }
    @Test fun settingsOpensInViewAndDismisses() {
        click("Settings")
        visible("Scan settings")
        click("Done")
        visible("Start field scan")
    }
    @Test fun finishRequiresConfirmationAndPreservesResume() {
        start()
        click("Finish scan")
        visible("Finish this scan?")
        click("Keep scanning")
        visible("Pause")
    }
    @Test fun openingResultsPausesRecording() {
        start()
        click("View findings →")
        visible("Scan findings")
        click("Back")
        visible("Resume")
    }
    @Test fun backgroundingStillPausesScan() {
        start()
        ui.activityRule.scenario.moveToState(Lifecycle.State.CREATED)
        ui.activityRule.scenario.moveToState(Lifecycle.State.RESUMED)
        ui.waitUntil(10_000) { has("Resume") }
        visible("Resume")
    }
    private fun start() {
        click("Walk & scan")
        ui.waitUntil(20_000) { !has("Preparing the camera model. Scanning will be available when ready.") }
        click("Start field scan")
        ui.waitUntil(10_000) { has("Pause") }
    }
    private fun has(text: String) = ui.onAllNodesWithText(text).fetchSemanticsNodes().isNotEmpty()
    private fun click(text: String) = ui.onAllNodesWithText(text).onFirst().performClick()
    private fun visible(text: String) = ui.onAllNodesWithText(text).onFirst().assertIsDisplayed()
}
