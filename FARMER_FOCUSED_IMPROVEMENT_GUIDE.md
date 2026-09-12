# Agribot Farmer-Focused Improvement Guide

> **Goal:** Transform Agribot into a system that is **usable by farmers for their daily purpose**, with an **intuitive UI**, **all necessary features**, and **maximum convenience** for crop monitoring and plant-health decisions.

This document is a living roadmap. It is organized by what a farmer needs to accomplish, then mapped to the concrete code changes required. Every recommendation references the exact files and functions that need to change.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Farmer Personas & Daily Workflows](#2-farmer-personas--daily-workflows)
3. [The Three Interfaces](#3-the-three-interfaces)
4. [Priority 1 — Make the Android App the Primary Farmer Tool](#4-priority-1-make-the-android-app-the-primary-farmer-tool)
5. [Priority 2 — Intuitive UI & Onboarding](#5-priority-2-intuitive-ui--onboarding)
6. [Priority 3 — Complete Feature Set for Daily Use](#6-priority-3-complete-feature-set-for-daily-use)
7. [Priority 4 — Reliability, Offline, & Field Robustness](#7-priority-4-reliability-offline--field-robustness)
8. [Priority 5 — Web Dashboard as a Secondary Companion](#8-priority-5-web-dashboard-as-a-secondary-companion)
9. [Priority 6 — ML Pipeline & Model Improvements](#9-priority-6-ml-pipeline--model-improvements)
10. [Priority 7 — Deployment & Operations](#10-priority-7-deployment--operations)
11. [Priority 8 — Technical Debt & Bug Fixes](#11-priority-8-technical-debt--bug-fixes)
12. [Testing Strategy](#12-testing-strategy)
13. [Phased Implementation Plan](#13-phased-implementation-plan)
14. [Quick-Reference Checklist](#14-quick-reference-checklist)

---

## 1. System Overview

Agribot is a **Raspberry Pi + Android** crop-monitoring system. The Pi runs a camera and an on-device ML model (OpenVINO/TFLite) that scans plants for disease. The Android app records, tracks, and displays results. A local web dashboard provides a secondary view.

**Current architecture (three layers):**

```
┌─────────────────────────────────────────────────────────────┐
│  FARMER INTERFACE LAYER                                     │
│  ┌──────────────────────────┐  ┌────────────────────────┐  │
│  │ Android Native App       │  │ Web Dashboard          │  │
│  │ (Kotlin + Compose)       │  │ (agribot_platform/)    │  │
│  │ Primary farmer tool      │  │ Secondary companion    │  │
│  └──────────────────────────┘  └────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  INFERENCE & DATA LAYER                                     │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 43_low_power_5v3a_experiment.py                        │ │
│  │ Live camera inference, decision gate, data export      │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 44_farmer_one_touch.py                                 │ │
│  │ Orchestrator: starts dashboard + inference + archive │ │
│  └────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ML PIPELINE LAYER                                          │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Scripts 1–42: train, export, benchmark, validate     │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Key files (already read):**

| Layer | Key Files |
|-------|-----------|
| Android app | `agribot_android_app/android/featurescan/SideScanViewModel.kt`, `EnhancedRecordingScreen.kt`, `AgribotScreens.kt`, `DataAnalysisScreen.kt`, `ScanUiState.kt`, `tracking/PlantTracker.kt`, `tracking/PlantTrackingMath.kt` |
| Android ML | `ml/inference/TFLiteClassifier.kt`, `TFLiteDetector.kt`, `FrontCandidateClassifier.kt` |
| Android data | `data/db/Entities.kt`, `Daos.kt`, `RoomRunRepository.kt`, `RoomFieldLayoutRepository.kt`, `FileExportRepository.kt` |
| Android domain | `domain/model/DomainModels.kt`, `ScanConstants.kt`, `PlantDecisionGate.kt`, `SideScanMapper.kt`, `RunSummaryReducer.kt` |
| Web dashboard | `agribot_platform/run_platform.py`, `app.js`, `index.html`, `i18n.js` |
| Android web app | `agribot_android_app/www/app.js`, `index.html`, `i18n.js` |
| Farmer orchestrator | `44_farmer_one_touch.py` |
| Config | `farmer_config.json` |
| Deployment | `deployment/12–20_*.sh` |

---

## 2. Farmer Personas & Daily Workflows

### Persona A: Smallholder Farmer (primary user)
- **Goal:** Walk the field, scan each plant row, identify sick plants, and know exactly where to act.
- **Constraints:** Limited tech literacy, works outdoors in bright sun, may not speak English, phone battery is precious, network is unreliable.
- **Daily workflow:**
  1. Start the scan (one tap).
  2. Walk down a row holding the phone/camera.
  3. See plants appear on screen with health status.
  4. When a sick plant is found, mark it and note the location.
  5. At the end, review results and export/share a report.

### Persona B: Farm Manager / Agronomist
- **Goal:** Oversee multiple fields, review historical data, spot trends, and make treatment decisions.
- **Workflow:** Review past runs, compare disease prevalence across fields, export CSV for analysis.

### Persona C: Field Technician
- **Goal:** Set up and maintain the Pi + camera system in the field.
- **Workflow:** Deploy the Pi, configure Wi-Fi/hotspot, start the system, troubleshoot.

---

## 3. The Three Interfaces

### 3.1 Android Native App (`agribot_android_app/android/`)

**What works well:**
- Full Kotlin + Jetpack Compose multi-module architecture with Hilt DI.
- Plant tracking system (`PlantTracker.kt`, `PlantTrackingMath.kt`) with IoU + cosine similarity + spatial distance matching.
- Bounding box overlay (`EnhancedRecordingScreen.kt`) with plant IDs, confidence, occlusion indicators.
- Sensor integration (accelerometer for tilt/roll/pitch, GPS for speed).
- Two recording modes: Side Scan (walk-and-scan) and Front Row Overview (burst capture).
- Data analysis screen with plant grid, filters, detail panel, export.
- Room database for runs, decisions, field layouts.
- TFLite models bundled in assets.

**What needs improvement (see sections 4–8 below).**

### 3.2 Web Dashboard (`agribot_platform/`)

**What works well:**
- Lightweight Python HTTP server (no external dependencies).
- Displays latest run, field map, plant catalogue, run summary.
- Field layout editor with rows, plant counts, spacing.
- CSV download.
- Offline caching via localStorage.
- 14-language i18n support.

**What needs improvement:**
- Read-only field layout (no delete/edit field).
- No historical run browsing.
- No plant search or filtering.
- No treatment notes or action tracking.
- No dark mode (hard to read in bright sun).
- No plant photo evidence display.

### 3.3 Android Web App (`agribot_android_app/www/`)

**What works well:**
- Connects to Pi's dashboard URL.
- Fallback Pi URLs.
- Request timeout and retry.

**What needs improvement:**
- README still says "Capacitor wrapper" but the app is now fully native — documentation is stale.
- The web app is redundant now that the native app exists — should be deprecated or repurposed as a quick-setup helper.

---

## 4. Priority 1 — Make the Android App the Primary Farmer Tool

### 4.1 One-Tap Start (Zero Configuration)

**Problem:** The farmer must select a mode, then tap "Start." The default field layout is hardcoded.

**Fix:**
- In `SideScanViewModel.kt:start()`, auto-create a default run with the current field layout and immediately begin recording.
- Show a single prominent **"Start Scan"** button on the home screen that does everything.
- Pre-fill the field layout from `farmer_config.json` (already imported in `ensureDefaultLayout()`).

**Files:**
- `featurescan/AgribotScreens.kt` — HomeScreen: make the start button the dominant element.
- `featurescan/SideScanViewModel.kt` — `start()`: simplify to one call.

### 4.2 Clear Visual Feedback During Scanning

**Problem:** The recording screen shows bounding boxes, but the farmer needs to know:
- Which plant is being scanned right now.
- Whether the scan was recorded.
- When to move to the next plant.

**Fix:**
- In `EnhancedRecordingScreen.kt`, add a **"Current Plant" card** that highlights the plant closest to the camera center.
- Add a **countdown timer** for the plant cooldown (already tracked in `cooldownRemainingMs`).
- Add a **success animation** (green flash + sound) when a decision is recorded.
- Show a **progress bar** for the overall field scan (e.g., "Plant 12 of 100 in Row A").

**Files:**
- `featurescan/EnhancedRecordingScreen.kt`
- `featurescan/ScanUiState.kt` — add `fieldProgress: Float` and `currentPlantLabel: String?`.

### 4.3 Intuitive Navigation

**Problem:** Navigation between Home → Recording → Data is functional but not obvious.

**Fix:**
- Add a **bottom navigation bar** with three icons: Home, Scan, Data.
- Add a **persistent "Back to Home"** button on the recording screen.
- Add a **"View Results"** button that appears after a run is completed.

**Files:**
- `featurescan/AgribotScreens.kt` — `AgribotScanApp()`: add bottom nav.
- `featurescan/EnhancedRecordingScreen.kt` — add back button.

### 4.4 Farmer-First Language

**Problem:** The UI uses technical terms like "Side Scan," "Front Row Overview," "Run," "Decision."

**Fix:**
- Rename modes to farmer-friendly terms:
  - "Side Scan" → "Walk & Scan"
  - "Front Row Overview" → "Burst Scan"
- Rename "Run" → "Scan Session"
- Rename "Decision" → "Plant Check"
- Use the i18n system to provide translations for all new terms.

**Files:**
- `featurescan/AgribotScreens.kt` — update all labels.
- `featurescan/EnhancedRecordingScreen.kt` — update labels.
- `featurescan/DataAnalysisScreen.kt` — update labels.

---

## 5. Priority 2 — Intuitive UI & Onboarding

### 5.1 First-Time Onboarding

**Problem:** There is no onboarding. A new farmer sees a technical screen with no guidance.

**Fix:**
- Add a **3-step onboarding flow** shown on first launch:
  1. "Point the camera at a plant row" (illustration).
  2. "Plants appear with green (healthy) or red (sick) boxes."
  3. "Tap a box to mark it, or export results at the end."
- Add a **"Show Tips"** toggle in settings that overlays hints on the recording screen.

**Files:**
- New file: `featurescan/OnboardingScreen.kt`
- `featurescan/AgribotScreens.kt` — add onboarding state.
- `featurescan/ScanUiState.kt` — add `showOnboarding: Boolean`.

### 5.2 Large, Touch-Friendly Controls

**Problem:** Some buttons are small and close together.

**Fix:**
- Ensure all buttons are **minimum 48dp** height (already done in most places).
- Add **more spacing** between buttons (currently 12dp — increase to 16dp).
- Add **haptic feedback** on button presses and plant marks.

**Files:**
- `featurescan/EnhancedRecordingScreen.kt` — increase spacing.
- `featurescan/AgribotScreens.kt` — add haptic feedback.

### 5.3 High-Contrast, Sun-Readable UI

**Problem:** The UI uses Material 3 default colors which can be hard to read in bright sunlight.

**Fix:**
- Add a **dark mode** toggle (the recording screen already uses dark overlays — extend this to the whole app).
- Increase **text contrast** and **font sizes** for outdoor readability.
- Add a **"Sun Mode"** that maximizes brightness and uses high-contrast colors.

**Files:**
- `designsystem/AgribotTheme.kt` — add dark color scheme.
- `featurescan/EnhancedRecordingScreen.kt` — add sun mode toggle.

### 5.4 Visual Plant Health Summary

**Problem:** After a scan, the farmer sees a list of plants but no easy summary.

**Fix:**
- On the Data Analysis screen, add a **summary card** at the top:
  - "Total plants scanned: 45"
  - "Healthy: 38 (84%)"
  - "Sick: 5 (11%)"
  - "Uncertain: 2 (4%)"
- Add a **color-coded field map** showing which rows/plants are sick.

**Files:**
- `featurescan/DataAnalysisScreen.kt` — add summary card (partially done in `PlantStatsSummary`).

---

## 6. Priority 3 — Complete Feature Set for Daily Use

### 6.1 Treatment Notes & Actions

**Problem:** The farmer can mark a plant as "Sick" but cannot add a note about what treatment was applied.

**Fix:**
- Add a **"Treatment Notes"** field to each plant decision.
- Add a **"Treat"** button on the plant detail panel that opens a note editor.
- Track treatment status: "Not Treated," "Treated," "Re-scan Needed."

**Files:**
- `domain/model/DomainModels.kt` — add `treatmentNote: String?` and `treatmentStatus: TreatmentStatus` to `RecordedDecision`.
- `data/db/Entities.kt` — add columns.
- `featurescan/DataAnalysisScreen.kt` — add treatment editor.

### 6.2 Plant Photo Evidence

**Problem:** Evidence frames are saved but not viewable in the app.

**Fix:**
- Add a **"View Evidence"** button on the plant detail panel that shows the saved JPEG.
- Add a **gallery view** of all evidence photos for a scan session.

**Files:**
- `featurescan/DataAnalysisScreen.kt` — add evidence gallery.
- `data/repository/FileEvidenceRepository.kt` — add method to load evidence for a plant.

### 6.3 Field Progress Tracking

**Problem:** The farmer doesn't know how much of the field has been scanned.

**Fix:**
- Add a **progress indicator** showing: "Row A: 12/20 plants scanned" with a progress bar.
- Add a **"Skip Row"** button to move to the next row.
- Add a **"Mark Row Complete"** button.

**Files:**
- `featurescan/ScanUiState.kt` — add `fieldProgress: Map<String, Float>`.
- `featurescan/EnhancedRecordingScreen.kt` — add progress UI.
- `featurescan/SideScanViewModel.kt` — add `skipRow()` and `completeRow()` methods.

### 6.4 Historical Trend Analysis

**Problem:** The farmer can only see the latest scan. No way to track disease progression over time.

**Fix:**
- Add a **"History"** tab on the Data Analysis screen.
- Show a **timeline** of past scans with sick/healthy counts.
- Add a **trend chart** showing disease prevalence over time.

**Files:**
- `featurescan/DataAnalysisScreen.kt` — add history tab.
- `domain/logic/RunHistoryPresenter.kt` — enhance with trend data.

### 6.5 Offline-First Operation

**Problem:** The app depends on the Pi for the web dashboard, but the native app should work fully offline.

**Fix:**
- Ensure all scan data is stored locally in Room (already done).
- Add a **"Sync Later"** banner when the Pi is unreachable.
- Add a **"Share Report"** button that generates a PDF/CSV without needing the Pi.

**Files:**
- `featurescan/AgribotScreens.kt` — add sync status.
- `data/repository/FileExportRepository.kt` — add PDF export.

### 6.6 Multi-Language Support

**Problem:** The Android app has no i18n — it's English only. The web dashboard has 14 languages but the Android app doesn't use them.

**Fix:**
- Port the i18n system from the web dashboard to the Android app.
- Add string resources for all 14 languages.
- Add a **language selector** in settings.

**Files:**
- New: `featurescan/i18n/` — string resources for each language.
- `featurescan/AgribotScreens.kt` — use localized strings.

### 6.7 Plant Search & Filter

**Problem:** With many plants, finding a specific one is hard.

**Fix:**
- Add a **search bar** on the Data Analysis screen.
- Add **filter chips** for health status, date range, row.
- Add **sort options** (by plant number, by confidence, by date).

**Files:**
- `featurescan/DataAnalysisScreen.kt` — add search and advanced filters.

---

## 7. Priority 4 — Reliability, Offline, & Field Robustness

### 7.1 Graceful Degradation When Sensors Are Unavailable

**Problem:** If the accelerometer or GPS is unavailable, the app still tries to use them.

**Fix:**
- In `PlantTracker.kt`, check if sensors are available before registering.
- Show a **"Sensor Unavailable"** warning instead of crashing.
- Make sensor data optional — the app should work without it.

**Files:**
- `featurescan/tracking/PlantTracker.kt` — add sensor availability checks.

### 7.2 Thermal Throttling Management

**Problem:** The app has thermal monitoring (`ThermalPolicy.kt`, `SideScanPerformanceGovernor.kt`) but the farmer isn't notified.

**Fix:**
- Show a **"Device Heating"** warning on the recording screen.
- Automatically reduce FPS when thermal throttling is detected.
- Add a **"Cool Down"** suggestion.

**Files:**
- `featurescan/EnhancedRecordingScreen.kt` — add thermal warning.
- `featurescan/SideScanViewModel.kt` — improve thermal feedback.

### 7.3 Battery Optimization

**Problem:** Continuous camera + inference drains the battery quickly.

**Fix:**
- Add a **"Power Save"** mode that reduces FPS and turns off evidence frames.
- Show **battery percentage** on the recording screen.
- Add a **"Low Battery"** warning at 20%.

**Files:**
- `featurescan/EnhancedRecordingScreen.kt` — add battery indicator.
- `featurescan/SideScanViewModel.kt` — add power save mode.

### 7.4 Robust Error Recovery

**Problem:** If the app crashes or the Pi disconnects, the farmer loses data.

**Fix:**
- Auto-save scan state every 5 seconds.
- Add a **"Recover Interrupted Scan"** option on the home screen.
- Show a **"Last Scan Recovered"** banner when restarting.

**Files:**
- `featurescan/SideScanViewModel.kt` — enhance `recoverInterruptedRun()`.
- `featurescan/AgribotScreens.kt` — add recovery UI.

---

## 8. Priority 5 — Web Dashboard as a Secondary Companion

### 8.1 Historical Run Browsing

**Problem:** The dashboard only shows the latest run.

**Fix:**
- Add a **"History"** section listing all past runs.
- Add a **"Compare Runs"** feature.

**Files:**
- `agribot_platform/app.js` — add run list and history view.
- `agribot_platform/index.html` — add history section.
- `agribot_platform/run_platform.py` — `/api/runs` already lists runs; enhance with more detail.

### 8.2 Dark Mode

**Problem:** The dashboard is light-only, hard to read in bright sun.

**Fix:**
- Add a **dark mode toggle** (persisted in localStorage).
- Use CSS variables for easy theme switching.

**Files:**
- `agribot_platform/styles.css` — add dark theme.
- `agribot_platform/app.js` — add theme toggle.

### 8.3 Plant Photo Evidence Display

**Problem:** Evidence photos are saved but not shown on the dashboard.

**Fix:**
- Add an **"Evidence"** column to the plant catalogue.
- Add a **modal viewer** for evidence photos.

**Files:**
- `agribot_platform/app.js` — add evidence display.
- `agribot_platform/index.html` — add evidence column.

### 8.4 Field Layout Management

**Problem:** Can add/edit fields but cannot delete them.

**Fix:**
- Add a **"Delete Field"** button.
- Add a **"Duplicate Field"** button.

**Files:**
- `agribot_platform/app.js` — add delete/duplicate.
- `agribot_platform/index.html` — add buttons.

### 8.5 Export & Share

**Problem:** CSV download works but no PDF or share options.

**Fix:**
- Add a **"Export PDF"** button.
- Add a **"Share Link"** that generates a shareable URL.

**Files:**
- `agribot_platform/run_platform.py` — add PDF export endpoint.
- `agribot_platform/app.js` — add export buttons.

---

## 9. Priority 6 — ML Pipeline & Model Improvements

### 9.1 Model Versioning & Updates

**Problem:** The model bundle ID is hardcoded as `"agribot-model-bundle-v001"` (see `ScanConstants.kt:30`). The manifest is observed but not used to dynamically load models.

**Fix:**
- Make the model bundle ID **configurable** via settings.
- Add a **"Model Update"** screen that checks for new models.
- Support **multiple crop models** (tomato, pepper, etc.) that can be switched.

**Files:**
- `domain/model/ScanConstants.kt` — already has `DEFAULT_MODEL_BUNDLE_ID`; make it overridable.
- `ml/inference/ModelManifestValidator.kt` — enhance to support model switching.
- `featurescan/SideScanViewModel.kt` — use manifest bundle ID instead of constant.

### 9.2 Confidence Threshold Configuration

**Problem:** Confidence threshold is hardcoded at 0.765 (see `ScanConstants.kt:11`, `PlantDecisionGate.kt:8`).

**Fix:**
- Add a **"Confidence Threshold"** slider in settings.
- Show a **"Low Confidence"** warning when confidence is below the threshold.
- Allow per-crop threshold configuration.

**Files:**
- `domain/model/ScanConstants.kt` — make threshold configurable.
- `domain/logic/ScanSettingsValidator.kt` — validate threshold.
- `featurescan/AgribotScreens.kt` — add threshold slider.

### 9.3 Disease Treatment Recommendations

**Problem:** The app identifies diseases but doesn't suggest treatments.

**Fix:**
- Add a **treatment recommendation database** mapping diseases to treatments.
- Show **"Recommended Treatment"** on the plant detail panel.
- Add **"Treatment History"** to track what was applied.

**Files:**
- New: `domain/model/TreatmentGuide.kt` — disease → treatment mapping.
- `featurescan/DataAnalysisScreen.kt` — add treatment recommendations.

### 9.4 Multi-Crop Support

**Problem:** The model is trained on a specific set of crops (tomato diseases).

**Fix:**
- Add a **"Crop Type"** selector in field setup.
- Bundle **multiple models** for different crops.
- Auto-select the correct model based on the crop type.

**Files:**
- `domain/model/DomainModels.kt` — add `cropType` to `FieldLayout`.
- `ml/inference/TfliteInferenceRepository.kt` — add model switching.

---

## 10. Priority 7 — Deployment & Operations

### 10.1 Simplified Field Setup

**Problem:** Setting up the Pi requires running multiple scripts and configuring nmcli.

**Fix:**
- Create a **"Setup Wizard"** script that guides the technician through:
  1. Wi-Fi configuration.
  2. Hotspot setup.
  3. Camera calibration.
  4. Field layout import.
- Add a **"Quick Deploy"** mode that uses defaults for common setups.

**Files:**
- New: `deployment/00_setup_wizard.sh`
- `44_farmer_one_touch.py` — add wizard subcommand.

### 10.2 Remote Monitoring

**Problem:** The farmer/technician can't check the system status remotely.

**Fix:**
- Add a **"Status API"** that returns system health (CPU, temperature, storage, last scan).
- Add a **"Remote Start/Stop"** endpoint.
- Add a **mobile notification** when a scan completes.

**Files:**
- `agribot_platform/run_platform.py` — add status and control endpoints.
- `44_farmer_one_touch.py` — add remote control support.

### 10.3 Automated Field Cleanup

**Problem:** Old runs and evidence accumulate, filling storage.

**Fix:**
- Add an **auto-archive** feature that moves old runs to a compressed archive.
- Add a **"Clean Up"** button that removes evidence older than 30 days.
- Show **storage usage** on the dashboard.

**Files:**
- `44_farmer_one_touch.py` — enhance `archive` command.
- `agribot_platform/run_platform.py` — add storage info.
- `featurescan/SideScanViewModel.kt` — add cleanup UI.

### 10.4 Network Resilience

**Problem:** The Pi's hotspot or Wi-Fi can be unreliable.

**Fix:**
- Add **automatic network fallback** (hotspot → Wi-Fi → offline).
- Add a **"Network Status"** indicator on both the app and dashboard.
- Add **QR code sharing** of the dashboard URL for easy connection.

**Files:**
- `44_farmer_one_touch.py` — enhance `prepare_access()`.
- `agribot_android_app/www/app.js` — add QR code display.
- `agribot_platform/run_platform.py` — add network status endpoint.

---

## 11. Priority 8 — Technical Debt & Bug Fixes

### 11.1 Memory Management

**Issues confirmed in `VERIFIED_BUG_ANALYSIS.md`:**
- `latestFrame` is never cleared in `SideScanViewModel.kt` (line 89, 598).
- `recentFrames` queue adds before checking capacity (line 599–602).

**Fixes already applied (see `FIXES_APPLIED.md`):**
- ✅ `latestFrame = null` in `stop()` and `onAppBackgrounded()`.
- ✅ Frame queue checks capacity before adding.

**Remaining:**
- Clear `latestFrame` when switching modes.
- Add `onCleared()` cleanup for all ViewModels.

**Files:**
- `featurescan/SideScanViewModel.kt` — add mode-switch cleanup.

### 11.2 Synchronization

**Issue:** `SideScanAnalyzer.kt` lacked synchronization (confirmed in `VERIFIED_BUG_ANALYSIS.md`).

**Fix already applied (see `FIXES_APPLIED.md`):**
- ✅ Added `synchronized(lock)` in `SideScanAnalyzer.kt`.

**Remaining:**
- Ensure all frame processing is single-threaded.
- Add thread-safety to `PlantTracker`.

**Files:**
- `camera/SideScanAnalyzer.kt` — verify fix.
- `featurescan/tracking/PlantTracker.kt` — add synchronization.

### 11.3 Hardcoded Values

**Issue:** Model bundle ID, calibration values, confidence thresholds are hardcoded (confirmed in `VERIFIED_BUG_ANALYSIS.md`).

**Fix already applied (see `FIXES_APPLIED.md`):**
- ✅ Created `ScanConstants.kt` with all constants.
- ✅ Replaced hardcoded values with constants.
- ✅ Added model bundle validation.

**Remaining:**
- Make constants configurable via settings.
- Add calibration UI for Front Overview mode.

**Files:**
- `domain/model/ScanConstants.kt` — add more configurable constants.
- `featurescan/SideScanViewModel.kt` — use settings instead of constants.

### 11.4 Error Handling

**Issue:** Incomplete error handling in frame processing (confirmed in `VERIFIED_BUG_ANALYSIS.md`).

**Fix:**
- Add try-catch around all inference calls.
- Show user-friendly error messages.
- Add automatic retry on failure.

**Files:**
- `featurescan/SideScanViewModel.kt` — enhance error handling in `onAnalysisFrame()`.
- `camera/SideScanAnalyzer.kt` — add error recovery.

### 11.5 Stale Documentation

**Issue:** README says "Capacitor wrapper" but the app is native.

**Fix:**
- Update README to reflect the native architecture.
- Add architecture diagrams.
- Add setup instructions for developers.

**Files:**
- `README.md` — rewrite.
- New: `docs/ARCHITECTURE.md`

### 11.6 Test Coverage

**Issue:** Tests exist but don't cover all workflows.

**Fix:**
- Add integration tests for the full scan workflow.
- Add UI tests for navigation.
- Add performance tests for sustained operation.

**Files:**
- `featurescan/src/test/` — add integration tests.
- `featurescan/src/androidTest/` — add UI tests.

---

## 12. Testing Strategy

### 12.1 Unit Tests
- `PlantTrackingMathTest.kt` — already exists; add more edge cases.
- `PlantDecisionGateTest.kt` — already exists; add boundary tests.
- `SideScanMapperTest.kt` — already exists; add field layout edge cases.

### 12.2 Integration Tests
- Test full scan workflow: start → scan → record → stop → export.
- Test error recovery: app backgrounding, permission denial.
- Test data persistence: restart app, verify data is saved.

### 12.3 Field Tests
- Test in bright sunlight (verify UI readability).
- Test with various phone models (verify performance).
- Test with different field layouts (verify mapping).
- Test with poor network (verify offline operation).

### 12.4 Automated Testing
- Add CI pipeline that runs tests on every commit.
- Add performance regression tests.
- Add accessibility tests (TalkBack, font scaling).

---

## 13. Phased Implementation Plan

### Phase 1 (Weeks 1–2): Foundation & Reliability
- [ ] Fix all confirmed bugs (memory, synchronization, error handling).
- [ ] Add comprehensive error handling and user feedback.
- [ ] Add battery and thermal monitoring to the UI.
- [ ] Update documentation.

### Phase 2 (Weeks 3–4): Farmer-First UI
- [ ] Add onboarding flow.
- [ ] Simplify navigation (bottom nav, one-tap start).
- [ ] Add sun-readable dark mode.
- [ ] Rename technical terms to farmer-friendly language.

### Phase 3 (Weeks 5–6): Complete Feature Set
- [ ] Add treatment notes and photo evidence viewing.
- [ ] Add field progress tracking.
- [ ] Add historical trend analysis.
- [ ] Add multi-language support.

### Phase 4 (Weeks 7–8): ML & Model Improvements
- [ ] Make model bundle ID configurable.
- [ ] Add confidence threshold slider.
- [ ] Add treatment recommendations.
- [ ] Add multi-crop support.

### Phase 5 (Weeks 9–10): Deployment & Polish
- [ ] Create setup wizard.
- [ ] Add remote monitoring.
- [ ] Add automated cleanup.
- [ ] Add network resilience features.
- [ ] Final testing and polish.

---

## 14. Quick-Reference Checklist

### Must-Have for Farmer Usability
- [ ] One-tap start scan
- [ ] Clear visual feedback (current plant, success animation)
- [ ] Simple navigation (home, scan, data)
- [ ] Farmer-friendly language
- [ ] Onboarding flow
- [ ] Sun-readable UI (dark mode, high contrast)
- [ ] Treatment notes per plant
- [ ] Plant photo evidence viewing
- [ ] Field progress tracking
- [ ] Offline operation
- [ ] Multi-language support
- [ ] Battery & thermal monitoring
- [ ] Error recovery

### Must-Have for Technical Completeness
- [ ] Configurable model bundle ID
- [ ] Configurable confidence threshold
- [ ] Model versioning & updates
- [ ] Multi-crop support
- [ ] Historical trend analysis
- [ ] Plant search & filter
- [ ] PDF/CSV export
- [ ] Remote monitoring
- [ ] Automated cleanup
- [ ] Network resilience
- [ ] Comprehensive tests
- [ ] Updated documentation

---

## Appendix A: Key Data Structures

### Decision Record (decisions.jsonl)
Each plant check is recorded as a JSON line with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `sequence` | int | Sequential number within the run |
| `timestamp` | string | ISO timestamp |
| `epoch_time` | float | Unix timestamp |
| `field_id` | string | Field name (e.g., "Field 2") |
| `row_id` | string | Row identifier (e.g., "A") |
| `row_index` | int | Row index (1-based) |
| `plant_column` | int | Column within the row |
| `plant_number` | int | Physical plant number |
| `plant_key` | string | Unique key: "field\|row\|plant" |
| `x_m` | float | X position in meters |
| `y_m` | float | Y position in meters |
| `plant_display` | string | Human-readable: "Field / Row A / Plant 12" |
| `scan_index` | int | Scan index |
| `label` | string | Health label (e.g., "Early_blight", "Healthy") |
| `confidence` | float | Confidence score (0.0–1.0) |
| `status` | string | "ok", "uncertain", "manual", "skipped" |
| `action` | string | "Inspect or treat", "Rescan this plant", "No action" |
| `frames_used` | int | Number of frames used in decision gate |
| `reason` | string | "primary_agreement", "backup_majority", "low_confidence", "disagreement" |
| `late_frames` | int | Number of late frames |
| `gate_avg_latency_ms` | float | Average inference latency |
| `gate_max_latency_ms` | float | Max inference latency |
| `threads` | int | CPU threads used |
| `fps_target` | float | Target FPS |

### Field Layout (field_layout.json)
```json
{
  "active_field_id": "field_2",
  "fields": [
    {
      "id": "field_1",
      "name": "Field 1",
      "active_row_id": "A",
      "row_count": 1,
      "plants_per_row": 100,
      "row_spacing_m": 1.0,
      "plant_spacing_m": 0.5,
      "start_plant": 1,
      "plant_step": 1,
      "plant_cooldown_sec": 2.0,
      "rows": [
        {
          "id": "A",
          "row_id": "A",
          "row_index": 1,
          "plants_per_row": 100,
          "plant_spacing_m": 0.5,
          "row_spacing_m": 1.0,
          "y_m": 0.0
        }
      ]
    }
  ],
  "updated_at": "2026-05-30T19:07:22+05:30"
}
```

### Plant Labels
The system recognizes these plant health labels:

| Label | Description |
|-------|-------------|
| `Early_blight` | Early blight disease |
| `Healthy` | Healthy plant |
| `Late_blight` | Late blight disease |
| `Leaf Miner` | Leaf miner pest damage |
| `Magnesium Deficiency` | Magnesium deficiency |
| `Nitrogen Deficiency` | Nitrogen deficiency |
| `Pottassium Deficiency` | Potassium deficiency |
| `Spotted Wilt Virus` | Spotted wilt virus |
| `Uncertain` | Low confidence / unclear |

---

## Appendix B: File Inventory

### Android Native App (`agribot_android_app/android/`)
```
app/                          # Main application
  src/main/java/com/sakshyam/agribot/
    MainActivity.kt           # Entry point, camera permission
    AgribotApplication.kt     # Hilt application class
  src/main/AndroidManifest.xml
  src/main/assets/
    farmer_config.json        # Default field config
    labels/labels.json        # Model labels
    model_manifest.json       # Model bundle manifest
    models/
      classifier_fastcrop_float32.tflite
      detector_nano_256_raw_float32.tflite

camera/                       # CameraX integration
  CameraPreview.kt            # Camera preview composable
  SideScanAnalyzer.kt         # Frame analysis
  FrameSampler.kt             # FPS control
  Yuv420FrameConverter.kt     # YUV → RGB conversion
  JpegEvidenceEncoder.kt      # Evidence frame encoding

core/                         # Core utilities

data/                         # Data layer (Room + repositories)
  db/
    AgribotDatabase.kt        # Room database
    Daos.kt                   # Data access objects
    Entities.kt               # Database entities
  repository/
    FileEvidenceRepository.kt  # Evidence frame storage
    FileExportRepository.kt    # CSV/JSONL/PDF export
    RoomFieldLayoutRepository.kt  # Field layout CRUD
    RoomRunRepository.kt       # Run & decision storage
    RepositoryMappers.kt       # Entity ↔ Domain mapping
  settings/
    ScanSettingsDataStore.kt  # Settings persistence

domain/                       # Domain logic (pure Kotlin)
  logic/                      # Business logic (30+ files)
    PlantDecisionGate.kt      # Confidence gate
    SideScanMapper.kt         # Plant position mapping
    RunSummaryReducer.kt      # Run summary computation
    FieldLayoutEditor.kt      # Layout editing
    FrontBurstProcessor.kt    # Front overview burst processing
    RunHistoryPresenter.kt    # History presentation
    DiagnosticsPresenter.kt   # Diagnostics
    ExportSerializer.kt       # Export formatting
    ScanSettingsValidator.kt  # Settings validation
    ThermalPolicy.kt          # Thermal management
    ...
  model/
    DomainModels.kt           # All domain models
    ScanConstants.kt          # Centralized constants

ml/                           # ML inference
  inference/
    TFLiteClassifier.kt       # Plant health classifier
    TFLiteDetector.kt         # Plant detector (YOLO)
    FrontCandidateClassifier.kt
    FrontCandidateCropper.kt
    DetectorOutputParser.kt
    DetectorPostprocessor.kt
    ModelAssetDigestVerifier.kt
    ModelUnavailableException.kt
    TfliteInferenceRepository.kt
  preprocessing/
    MaskCropPreprocessor.kt   # Preprocessing
    RgbImage.kt               # RGB image wrapper

design-system/                # Compose theme
  AgribotTheme.kt

featurescan/                  # Scanning UI (main farmer interface)
  AgribotScreens.kt           # Navigation & screens
  EnhancedRecordingScreen.kt  # Recording UI with bounding boxes
  DataAnalysisScreen.kt       # Data viewing
  SideScanViewModel.kt        # Main ViewModel
  ScanUiState.kt              # UI state
  FrontBurstCapturePolicy.kt  # Burst capture logic
  SideScanPerformanceGovernor.kt  # Performance management
  DeviceHealthStatusPresenter.kt  # Health status
  FrontReviewPresenter.kt     # Front review logic
  tracking/
    PlantTracker.kt           # Plant tracking system
    PlantTrackingMath.kt      # Tracking math utilities
```

### Web Dashboard (`agribot_platform/`)
```
run_platform.py               # HTTP server
app.js                        # Frontend logic
index.html                    # Layout
styles.css                    # Styling
i18n.js                       # 14-language i18n
service-worker.js             # Offline support
manifest.webmanifest          # PWA manifest
import_legacy_jsonl.py        # Data import utility
```

### Python ML Pipeline (root)
```
1_download.py                 # Download datasets
2_prepare.py                  # Prepare data
3_train.py                    # Train model
4_calibrate_threshold.py      # Calibrate confidence
4_infer.py                    # Run inference
5_download_plantvillage.py    # Download PlantVillage
6_prepare_clf.py              # Prepare classifier
7_train_clf.py                # Train classifier
8_eval_clf.py                 # Evaluate classifier
9_infer_clf.py                # Classifier inference
10_export_runtime_models.py   # Export models
16_openvino_retry.py          # OpenVINO export
30_run_runtime_suite.py       # Runtime test suite
31_summarize_runtime_results.py
32_optimize_openvino_deployment.py
33_prepare_fast_crop_clf.py
34_benchmark_tomato_crop_disease.py
35_prepare_tomato_crop_disease_yolo.py
36_prepare_combined_crop_detector_yolo.py
37_train_deployment_fastcrop_classifier.py
38_sweep_classifier_confidence_gate.py
39_package_pi_deployment.py
40_pi_end_to_end_validation.py
41_pi_tomato_crop_disease_validation.py
42_pi_preflight_no_inference.py
43_low_power_5v3a_experiment.py  # Live inference (key file)
44_farmer_one_touch.py        # Farmer orchestrator
45_build_android_apk.py       # APK build
46_prepare_android_model_bundle.py
47_android_model_benchmark.py
48_android_release_readiness.py
49_validate_android_field_export.py
```

### Deployment (`deployment/`)
```
12_performance_10fps_benchmark.sh
13_performance_10fps_live.sh
14_start_field_platform.sh
15_farmer_one_touch.sh
16_install_farmer_autostart.sh
17_archive_field_cleanup.sh
18_enable_direct_phone_hotspot.sh
19_connect_configured_wifi.sh
19_return_to_airtel_wifi.sh
20_install_final_field_system.sh
```

---

## Appendix C: Configuration (`farmer_config.json`)

```json
{
  "dashboard_port": 8080,
  "camera": "picamera2",
  "field_id": "Field 2",
  "row_id": "B",
  "start_plant": 1,
  "plant_step": 1,
  "plant_cooldown_sec": 2.0,
  "field_layout": {
    "active_field_id": "field_2",
    "fields": [
      {
        "id": "field_1",
        "name": "Field 1",
        "row_count": 1,
        "plants_per_row": 100,
        "row_spacing_m": 1.0,
        "plant_spacing_m": 0.5,
        "rows": [
          { "id": "A", "plants_per_row": 100, "plant_spacing_m": 0.5, "row_spacing_m": 1.0, "y_m": 0.0 }
        ]
      },
      {
        "id": "field_2",
        "name": "Field 2",
        "row_count": 3,
        "plants_per_row": 20,
        "row_spacing_m": 1.4,
        "plant_spacing_m": 0.45,
        "rows": [
          { "id": "A", "plants_per_row": 20, "plant_spacing_m": 0.45, "row_spacing_m": 1.4, "y_m": 0.0 },
          { "id": "B", "plants_per_row": 15, "plant_spacing_m": 0.35, "row_spacing_m": 1.4, "y_m": 1.4 },
          { "id": "C", "plants_per_row": 22, "plant_spacing_m": 0.5, "row_spacing_m": 1.4, "y_m": 2.8 }
        ]
      }
    ]
  },
  "fps": 10.0,
  "threads": 4,
  "cpu_affinity": "0,1,2,3",
  "classifier": "runtime_exports/classifier_openvino_model",
  "data_root": "agribot_inference_data",
  "network": {
    "mode": "hotspot",
    "hotspot_ssid": "Agribot_Field",
    "hotspot_password": "",
    "hotspot_ip": "10.42.0.1/24"
  }
}
```

---

*This document is a comprehensive roadmap for transforming Agribot into a farmer-ready tool. It should be updated as features are implemented and new insights are gained from field testing.*
