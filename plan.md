# Agribot Android Direct Replacement Plan

## Document Purpose

This file is the implementation-ready plan for converting Agribot from a Raspberry Pi oriented field system into a native Android app that directly replaces the Pi. The target app must run the phone camera, model inference, decision gating, field mapping, run logging, exports, and farmer controls on the phone itself. It must not depend on a Raspberry Pi, an external server, internet access, cloud inference, or remote dashboards for normal field operation.

The current repo already contains:

- Raspberry Pi runtime scripts, including `inference_rpi.py`, `pi_plant_loop.py`, `43_low_power_5v3a_experiment.py`, and `44_farmer_one_touch.py`.
- Current model assets under `models/`, including classifier and detector `.pt` weights.
- Pi field configuration in `farmer_config.json`.
- A local dashboard under `agribot_platform/`.
- A Capacitor Android wrapper under `agribot_android_app/` that currently connects to a Pi dashboard and does not run inference on the phone.

The new Android app should keep the good parts of the Pi system: one-touch start, confidence-gated decisions, row and plant mapping, JSONL/CSV-style export compatibility, farmer-friendly controls, local/offline operation, and performance discipline. It should replace the Pi-specific pieces: Python runtime, OpenVINO target, systemd autostart, Pi hotspot management, Pi HTTP dashboard dependency, and Picamera2 camera path.

## Capability Statement

Agribot Android gives a farmer or field operator a one-button phone app that records crop health using the phone camera, runs all plant detection and disease classification locally on the phone CPU, maps each decision to a field, row, side, and plant position, and stores/export results without a Raspberry Pi. It supports two production recording modes:

1. Side Scan Mode: the phone is carried or mounted along a row and captures close side images like the Pi system was designed to do.
2. Front Row Overview Mode: the phone camera is positioned in front of the crop at about one meter distance and captures a wider image where rows on both sides of the field corridor can be seen.

## Non-Negotiable Product Requirements

- Direct Pi replacement: no Raspberry Pi required for capture, inference, mapping, storage, or export.
- Local CPU default: inference must run locally on the phone CPU by default. GPU/NPU delegates may be optional accelerators, but the app must remain usable on CPU.
- Offline by default: no network required for normal scanning.
- One-touch operation: the main screen must have a large Start button and clear Stop/Pause controls.
- Two recording modes: Side Scan Mode and Front Row Overview Mode must be first-class modes, not hidden settings.
- Model included: the APK or signed local model bundle must include the required model artifacts and labels.
- Data ownership: all scans, frames selected for saving, summaries, field layouts, and exports live on the phone unless the user explicitly exports them.
- Compatibility: Android exports should preserve the Pi decision schema as much as possible so current analysis/dashboard code can consume Android runs.
- Farmer-safe UX: all critical controls should be simple, visible, and recoverable in the field.
- Clear uncertainty: the app must mark uncertain decisions instead of inventing confident disease labels.
- Field durability: failed runs, app restarts, low battery, thermal throttling, or permission loss must not corrupt previous run data.

## Pre-Development Decisions

These decisions are fixed for the first native Android implementation so a developer can start without asking for clarification.

| Area | Decision |
| --- | --- |
| Native app location | Build the direct replacement as native Kotlin code inside the existing Android project path, replacing Capacitor as the production runtime. Keep the old Capacitor/WebView behavior only as reference material unless a separate legacy viewer is explicitly needed later. |
| Package name | Keep `com.sakshyam.agribot` for continuity. |
| Minimum SDK | Use `minSdk 26` for the native replacement. The old Capacitor wrapper used 23, but the new CameraX/foreground-service/TFLite runtime should standardize on 26 unless a real target phone requires older support. |
| Target SDK | Use the current Android target supported by the installed toolchain; as of this repo, the wrapper targets SDK 35. |
| ABI | Ship `arm64-v8a` only for v1. Add `armeabi-v7a` only if a named target device requires it. |
| Device target | The minimum real-device requirement is any Android 8+ arm64 phone with rear camera, 4 GB RAM, and enough CPU to sustain 5 selected FPS in Side Scan. The reference class is a mid-range 6-8 GB RAM Android phone. Before field pilot, name and benchmark the exact physical phone used. Lack of a named phone does not block development. |
| Camera | Use rear normal/wide camera as default. Do not use ultrawide automatically because its distortion harms geometry; expose ultrawide only as an advanced/manual camera choice after transform tests pass. |
| Orientation | Default field operation is portrait. Landscape must not break transforms, but portrait is the supported v1 operating posture. |
| Runtime library | Use raw TensorFlow Lite Interpreter APIs for v1, CPU/XNNPACK default. Do not use TFLite Task Library for the first implementation. Do not use ONNX Runtime or OpenVINO on Android for v1. |
| Dependency injection | Use Hilt from scratch. It is not already present in the Capacitor project. |
| Module structure | Create Gradle modules from the start: `:app`, `:domain`, `:data`, `:camera`, `:ml`, at least one `:feature-scan`, and `:design-system`. |
| Model packaging | Embed the v1 classifier model and labels in app assets or an install-time model bundle. No OTA model download infrastructure in v1. |
| Model integrity | Verify SHA-256 hashes from `model_manifest.json` before loading a model. Signature verification is not required for embedded v1 assets, but the manifest must be ready for signed bundles later. |
| Required v1 mode order | Build Side Scan Mode first and field-validate it before Front Row Overview. The final release is not complete until both modes exist. |
| Side Scan detector | Disabled by default. Side Scan v1 is classifier-only with mask crop, matching the Pi fast deployment path. |
| Front Overview detector | Required for Front Row Overview. `detector_nano_256.pt` is the first candidate but must pass Android CPU benchmarks after conversion; if it fails, replace or retrain a smaller detector. |
| OpenCV | Do not add OpenCV for Side Scan. Front Overview Burst MVP must first use CameraX frames, TFLite detector boxes, guide rails, and Kotlin geometry. Add OpenCV only in Phase 6 if row/corridor geometry fails acceptance without contour/homography support. |
| Internet permission | Do not request `INTERNET` in the production native direct-replacement app. The old wrapper needed it for Pi HTTP access; native v1 is offline-only. |
| Evidence frames | Save JPEG evidence for sick and uncertain decisions by default. Do not save every raw frame. Cap evidence storage at 500 MB or 2,000 frames per run, whichever comes first, then warn and continue decisions without new evidence unless the user frees space. |
| Auto-advance | Enabled by default in Side Scan. After each emitted decision, advance to the next plant after `plant_cooldown_sec`, default `2.0`. Retake rolls back/replaces the current plant decision without corrupting sequence history. |
| Manual correction audit | Every manual override writes a correction event with timestamp, original label/status/confidence, new label/status, optional reason, and local operator display name if configured; otherwise `operator = local_user`. |
| Run ID format | Android-origin runs use `android_YYYYMMDD_HHMMSS`. Imported Pi runs keep their source ID when safe. |
| Export compatibility | Preserve Pi core keys exactly in JSONL/CSV. Android may add fields, but must not remove or rename existing Pi-compatible fields. |
| Localization | Start native v1 with English. Migrate existing `i18n.js` strings only after encoding and wording audit. Additional languages are not a blocker for Side Scan MVP. |
| Signing and distribution | Field pilot uses internal sideload debug/release APKs. Play Store/AAB and production keystore management are release-phase tasks, not blockers for development. |

## Current Pi Behavior To Preserve

The Android app should preserve these behavior contracts from the current Pi codebase:

### Inference Defaults

From `inference_rpi.py` and `RUNTIME_TESTING.md`:

- Classifier confidence threshold: `0.765`.
- High-confidence decision threshold: `0.765`.
- Side-scan live crop mode: `mask`.
- Crop scale: `0.55`.
- Crop pad: `0.05`.
- Max edge: `256`.
- Side-scan live loop can run classifier-only with detector disabled for speed.
- Detector may be used for richer detection, but live Pi deployment currently favors classifier-only fast crop.

### Decision Gate

The current `PlantDecisionGate` uses:

- Two primary frames.
- One backup frame.
- Emit immediately when the first two primary frames agree and both exceed high confidence.
- If primary frames disagree or are low confidence, consume a backup frame.
- Emit `Uncertain` if the backup cannot resolve the result.

Android should port this gate exactly first, then only change it behind tests and versioned configuration.

### Field Mapping

The Pi runtime maps each emitted decision to:

- `field_id`
- `row_id`
- `row_index`
- `plant_column`
- `plant_number`
- `plant_key`
- `x_m`
- `y_m`
- `scan_index`
- `scan_pass`
- `field_complete`
- `planned_total_plants`
- `plant_display`

Android should preserve these keys for Side Scan Mode and extend them for Front Row Overview Mode with side and geometry metadata.

### Run Output

The Pi writes:

- `metadata.json`
- `summary.json`
- `decisions.jsonl`
- `events.csv`
- `latest_run.json`
- `field_layout.json`

Android should store the same logical records in Room and export the same JSONL/CSV structure.

### `farmer_config.json` Migration Mapping

When importing or seeding from the current Pi config, map fields as follows:

| Pi config field | Android destination | Notes |
| --- | --- | --- |
| `dashboard_port` | ignored for native runtime | No local HTTP dashboard in production native app. |
| `camera` | diagnostics only | Native app selects CameraX rear normal camera by default. |
| `field_id` | active `FieldLayout.name` fallback | Use only when `field_layout.active_field_id` is absent. |
| `row_id` | active `FieldLayout.activeRowId` fallback | Use only when selected field lacks `active_row_id`. |
| `start_plant` | `FieldLayout.startPlant` | Default `1`. |
| `plant_step` | `FieldLayout.plantStep` | Default `1`. |
| `plant_cooldown_sec` | `FieldLayout.plantCooldownSec` and Side Scan auto-advance setting | Default `2.0`. |
| `field_layout.active_field_id` | active field selector | Must select one imported `FieldLayout`. |
| `field_layout.fields[]` | `field_layouts` rows | Preserve `id`, `name`, `field_id`, `active_row_id`, row defaults, and `updated_at`. |
| `field_layout.fields[].rows[]` | `field_rows` rows | Preserve `row_id`, `row_index`, `plants_per_row`, `plant_spacing_m`, `row_spacing_m`, and `y_m`. |
| `fps` | recording profile default | Side Scan default remains `5`; Pi `10` can seed performance profile only. |
| `threads` | ML CPU thread setting | Default auto profile may override after benchmark. |
| `cpu_affinity` | ignored | Android does not pin Linux CPU cores directly in v1. |
| `classifier` | imported model source note | Android uses TFLite assets from `model_manifest.json`, not OpenVINO path. |
| `data_root` | ignored for native storage | Android uses app-private storage and export bundles. |
| `network` | ignored for native runtime | No hotspot/Wi-Fi management in the phone replacement. |

### Dashboard Concepts

The existing dashboard surfaces:

- Latest run.
- Decision counts.
- Sick count.
- Uncertain count.
- Average/max confidence.
- Recent decisions.
- Field map.
- Field layout editor.
- CSV download.
- Multilingual labels through `i18n.js`.

Android should rebuild these natively in Compose, not rely on a WebView for core operation.

## Recommended Direction

Build a native Android app in Kotlin with Jetpack Compose, CameraX, Room, DataStore, Hilt, and raw TensorFlow Lite Interpreter model execution. Treat the existing Capacitor app as reference material and source of UI/data expectations, not as the final implementation surface.

### Why Native Android

- CameraX gives direct access to Preview, ImageAnalysis, ImageCapture, lifecycle binding, backpressure, and rotation transforms.
- Kotlin coroutines and Flow are a strong fit for camera/inference/run-state streaming.
- Room gives durable local run history and exportable records.
- A native app can manage CPU threads, thermal state, permissions, file sharing, and model memory mapping more reliably than a WebView.
- A native app avoids keeping local HTTP cleartext access and Pi URL logic as a core dependency.

### What To Keep From The Current Android Folder

- Package/app name can continue as `com.sakshyam.agribot` unless there is a naming reason to change it.
- Existing visual language and language strings can inform the native UI.
- Existing debug APK helper can be replaced or adapted to Gradle native builds.
- Salvage from Capacitor: product strings, field layout logic, dashboard concepts, and CSS/UI behavior. There are no Compose screens/ViewModels to port.

### What To Replace

- Replace Capacitor-only `MainActivity extends BridgeActivity` with a native Kotlin `MainActivity`.
- Remove Pi URL connection as the primary app mode.
- Remove cleartext HTTP as a requirement for production.
- Move dashboard state from browser localStorage to Room/DataStore.
- Move field layout editing from browser JS to Compose screens and ViewModels.

## App Modes

## Mode 1: Side Scan Mode

### Purpose

Side Scan Mode is the direct Android equivalent of the Pi live scan. The camera looks at plants from the side as the operator moves down a row. The app samples frames, runs local inference, gates the result, maps the result to the next plant position, and tells the operator what to do.

### Operator Workflow

1. Open app.
2. Confirm or select field.
3. Select Side Scan Mode.
4. Select row or choose auto-continue across rows.
5. Press Start.
6. Aim phone at the side of the current plant.
7. App collects two or three selected frames.
8. App emits Healthy, disease label, or Uncertain.
9. App shows a clear result, haptic feedback, and next plant instruction.
10. Operator moves to next plant.
11. App continues until Stop or field complete.

### Camera Position

- Phone is held or mounted beside the row.
- Target distance should be configurable and guided, initially 20 to 60 cm from the plant/leaf area.
- Camera can be rear-facing by default.
- Support portrait as the default field orientation. Landscape is allowed only after coordinate transforms and overlays pass rotation tests.

### Capture Strategy

- Use CameraX `Preview` for live view.
- Use CameraX `ImageAnalysis` for inference frames.
- Use CameraX `ImageCapture` only for optional saved evidence frames or manual snapshots.
- Bind Preview and ImageAnalysis in one CameraX session so dimensions and transforms can be understood together.
- Use `STRATEGY_KEEP_ONLY_LATEST` style behavior to avoid backlogs. If inference is slower than camera frame rate, old frames should be dropped.
- Start with target analysis resolution near 640x480 or 1280x720, then downscale/crop to model input.
- Run selected-frame inference at 5 FPS baseline and 10 FPS performance target, matching the Pi profiles.

### Side Scan Inference Flow

1. Receive `ImageProxy` from CameraX.
2. Convert YUV/RGBA to RGB in a tested conversion path.
3. Apply rotation and crop transform so the analysis image matches the visible preview.
4. Downscale to `maxEdge = 256`.
5. Apply crop strategy:
   - `mask`: color/vegetation mask crop equivalent to Pi `mask_crop`.
   - `center`: center crop fallback.
   - `none`: full resized image.
  6. Run mask-crop parity check against `inference_rpi.py` `mask_crop` implementation on golden test images.
  7. Run classifier on local CPU.
  8. Convert top class and confidence into a normalized label record.
  9. Send prediction into the two-primary plus one-backup decision gate.
  10. Emit a plant decision when the gate returns a result.
  11. Persist decision and update live UI.

### Side Scan Field Mapping

Use the current row-major mapper:

- `field_id`: active field name.
- `row_id`: active row such as A, B, C.
- `row_index`: 1-based row index.
- `plant_column`: 1-based position inside row.
- `plant_number`: configured start plus `plant_column * plant_step`.
- `x_m`: plant column multiplied by row plant spacing.
- `y_m`: row y-position from field layout.
- `plant_key`: stable string key.
- `scan_pass`: increments if scan continues beyond planned positions.
- `field_complete`: true when scan index exceeds planned total.

### Side Scan UX Requirements

- Main button states: Start, Pause, Resume, Stop.
- Large live status: current field, row, plant number, and next plant.
- Large latest decision tile: Healthy, disease, or Uncertain.
- Confidence shown as percent, but not overemphasized.
- A visible "Retake current plant" action.
- A visible "Skip current plant" action.
- A visible "Mark manually" action with Healthy, Sick, Uncertain, and notes.
- Optional sound/haptic:
  - Healthy: short neutral tick.
  - Sick: stronger alert.
  - Uncertain: distinct prompt to rescan.
- If inference is late, show a small status such as "Phone is hot or slow - reducing capture rate".

### Side Scan Acceptance Criteria

- Starts capture and inference from a single Start button.
- Runs without Pi, network, or server.
- Emits decisions with the same gate logic as Pi.
- Stores every decision in Room and exportable JSONL/CSV.
- Sustains 5 selected FPS on target mid-range phone without frame backlog.
- Keeps average emitted decision latency acceptable for plant-by-plant walking.
- Handles app pause, permission denial, camera disconnect, low battery, and thermal warnings gracefully.

## Mode 2: Front Row Overview Mode

### Purpose

Front Row Overview Mode is a new field-corridor mode. The camera is placed or held in front of the crop at about one meter distance, facing down the row/corridor, so the image can see plants or crop rows on both sides. It should produce mapped plant/row health decisions from a wider scene instead of a close side image of one plant.

This mode is more technically demanding than Side Scan Mode because it must solve geometry, row assignment, scale variation, plant detection, possible occlusion, perspective distortion, and multiple plants per image.

### Operator Workflow

1. Open app.
2. Select field.
3. Select Front Row Overview Mode.
4. Choose field corridor configuration:
   - Left row and right row IDs.
   - Approximate distance from camera to nearest plants.
   - Row spacing and plant spacing.
   - Camera height and tilt profile if known.
5. Stand or place phone about one meter before the plant/corridor.
6. Align the on-screen guide so left and right rows are inside the row rails.
7. Press Start or Capture Burst.
8. App captures a burst of frames, detects visible plants/canopy areas, assigns detections to left/right rows and plant positions, classifies disease crops, and stores decisions.
9. App shows a map of visible row segments and confidence/coverage.
10. Operator moves to next corridor segment and repeats.

### Camera Position

Initial recommended setup:

- Rear camera.
- Phone held at roughly 0.8 m to 1.2 m from nearest plants.
- Camera pointed forward/down the row corridor.
- The frame should contain both left and right side rows when possible.
- Operator should align two guide rails on screen with the left and right crop rows.

The app must not assume perfect geometry. It should grade capture quality and ask for retake when row visibility is too low.

### Front Overview Capture Profiles

Start with two profiles:

- Burst Capture: 5 to 9 frames over 1 to 2 seconds, then aggregate. This is the safest first implementation.
- Continuous Corridor Scan: inference runs continuously while the operator walks or moves the phone. This should come after Burst Capture works.

### Front Overview Inference Flow

1. Capture analysis frame or burst frames.
2. Apply rotation and camera transform.
3. Run row/plant detection:
   - Use a YOLO-style crop/plant detector exported to TensorFlow Lite.
   - Vegetation segmentation or color masks are allowed only as secondary assists, not as the sole source of truth.
4. Estimate row geometry:
   - Use detected plant centers, vegetation masks, or row guide inputs.
   - Fit two row rails/centerlines using robust regression or RANSAC.
   - Assign detections to left or right row based on signed distance to row rails.
5. Estimate plant position:
   - Use configured plant spacing.
   - Use perspective depth ordering from image y-coordinate and row line geometry.
   - Use optional manual "nearest plant number" for each segment.
   - Produce `plant_column_estimate`, `plant_number`, and `position_confidence`.
6. Crop each plant/canopy region.
7. Run disease classifier on each crop.
8. Aggregate across burst frames:
   - Merge detections by row side and approximate plant index.
   - Vote or average confidence across repeated detections.
   - Use the same high-confidence/uncertain policy at plant level.
9. Persist decisions and geometry diagnostics.

### Front Overview Geometry Model

The app should store a `FrontCaptureCalibration` per field or session:

- `camera_height_m`
- `camera_distance_to_nearest_row_m`
- `camera_tilt_degrees`
- `row_spacing_m`
- `plant_spacing_m`
- `left_row_id`
- `right_row_id`
- `nearest_left_plant_number`
- `nearest_right_plant_number`
- `guide_rail_left_px`
- `guide_rail_right_px`
- `calibration_quality`

The first version can avoid full metric reconstruction by requiring the operator to enter or confirm nearest visible plant number and by using relative row/plant ordering. Later versions can add homography and camera intrinsics.

### Front Overview Row Assignment

Each detection should produce:

- `bbox_px`
- `center_px`
- `row_side`: `left`, `right`, or `unknown`
- `row_id`
- `plant_column_estimate`
- `plant_number`
- `plant_position_confidence`
- `geometry_reason`

If row assignment is ambiguous, the decision should be saved as `status = uncertain` with `row_side = unknown` rather than mapped confidently to the wrong plant.

### Front Overview Quality Gates

Reject or warn on a capture when:

- Fewer than two row rails or row regions can be found and operator guide rails are not set.
- Detected plant boxes are too small for disease classification.
- Motion blur is high.
- Exposure is too dark or highlights are blown out.
- The row-side assignment confidence is below threshold.
- The plant index estimate conflicts with existing decisions in the current segment.
- The model finds disease but plant mapping is ambiguous.

### Front Overview UX Requirements

- Mode setup screen with clear left/right row selection.
- On-screen alignment rails.
- Distance guide such as "Move closer", "Move back", or "Rows not visible".
- Capture Burst button.
- Live coverage overlay with boxes and left/right row colors.
- Review screen before committing if mapping confidence is low.
- "Assign left/right manually" correction tool.
- "Merge with previous segment" and "New segment" controls.
- Export includes geometry diagnostics.

### Front Overview Acceptance Criteria

- Captures a one-meter front/corridor image or burst without Pi.
- Detects multiple visible plant/crop regions.
- Assigns detections to left/right rows when geometry is clear.
- Classifies each crop locally on CPU.
- Marks ambiguous geometry as uncertain instead of guessing.
- Stores all front-mode decisions with geometry fields.
- Allows manual correction before export.

## Architecture

## Target Project Structure

Recommended native Android layout:

```text
agribot_android_app/
  android/
    app/
      src/main/
        java/com/sakshyam/agribot/
          AgribotApplication.kt
          MainActivity.kt
          di/
          core/
          domain/
          data/
          ml/
          camera/
          feature/
            home/
            scan/
            fieldsetup/
            runhistory/
            export/
            diagnostics/
          design/
        assets/
          models/
          labels/
          model_manifest.json
```

Create these Gradle modules from the start:

```text
:app
:core
:domain
:data
:ml
:camera
:feature-scan
:feature-fieldsetup
:feature-history
:design-system
```

### Dependency Rules

Multi-module is required from the start: at minimum `app`, `domain`, `data`, `camera`, `ml`, and feature modules. Single-module implementation is not accepted for the native replacement. Hilt is not in the Capacitor project; set it up from scratch in the native app.

- `domain` contains pure Kotlin models, repository interfaces, use cases, and decision gate logic.
- `data` implements repositories using Room, DataStore, file export, and model manifest loading.
- `ml` owns model loading, preprocessing, inference, postprocessing, and benchmarks.
- `camera` owns CameraX binding and frame conversion.
- `feature-*` modules own Compose screens and ViewModels.
- `app` wires dependencies, navigation, permissions, and application lifecycle.

No domain object should depend on Android framework classes, CameraX classes, Room annotations, or TensorFlow Lite interpreter types.

## Core Domain Model

### Enums And Value Types

```kotlin
enum class RecordingMode {
    SIDE_SCAN,
    FRONT_ROW_OVERVIEW,
}

enum class RunState {
    IDLE,
    PERMISSION_REQUIRED,
    READY,
    STARTING,
    RECORDING,
    PAUSED,
    STOPPING,
    COMPLETED,
    FAILED,
}

enum class DecisionStatus {
    OK,
    UNCERTAIN,
    MANUAL,
    SKIPPED,
}

enum class PlantHealthAction {
    NONE,
    INSPECT_OR_TREAT,
    RESCAN,
}
```

### Field Layout

```kotlin
data class FieldLayout(
    val id: FieldId,
    val name: String,
    val activeRowId: RowId,
    val rows: List<FieldRow>,
    val startPlant: Int,
    val plantStep: Int,
    val plantCooldownSec: Double,
    val updatedAt: Instant,
)

data class FieldRow(
    val id: RowId,
    val rowIndex: Int,
    val plantsPerRow: Int,
    val plantSpacingM: Double,
    val rowSpacingM: Double,
    val yM: Double,
)
```

### Inference Prediction

```kotlin
data class FramePrediction(
    val label: String,
    val confidence: Float,
    val rawLabel: String,
    val isDisease: Boolean,
    val isUncertain: Boolean,
    val latencyMs: Double,
    val modelVersion: String,
)
```

### Plant Decision

```kotlin
data class PlantDecision(
    val label: String,
    val confidence: Float,
    val status: DecisionStatus,
    val framesUsed: Int,
    val reason: String,
)
```

### Recorded Decision

```kotlin
data class RecordedDecision(
    val id: DecisionId,
    val runId: RunId,
    val sequence: Int,
    val timestamp: Instant,
    val mode: RecordingMode,
    val fieldId: String,
    val rowId: String?,
    val rowSide: RowSide?,
    val rowIndex: Int?,
    val plantColumn: Int?,
    val plantNumber: Int?,
    val plantKey: String?,
    val xM: Double?,
    val yM: Double?,
    val bboxPx: BoundingBox?,
    val geometryConfidence: Float?,
    val label: String,
    val confidence: Float,
    val status: DecisionStatus,
    val action: PlantHealthAction,
    val framesUsed: Int,
    val reason: String,
    val lateFrames: Int,
    val gateAvgLatencyMs: Double,
    val gateMaxLatencyMs: Double,
    val modelVersion: String,
)
```

## Use Cases

Implement use cases in the domain layer:

- `StartScanRunUseCase`
- `PauseScanRunUseCase`
- `ResumeScanRunUseCase`
- `StopScanRunUseCase`
- `ObserveActiveRunUseCase`
- `ObserveLatestDecisionUseCase`
- `ObserveRunSummaryUseCase`
- `ObserveFieldLayoutUseCase`
- `SaveFieldLayoutUseCase`
- `RunSideFrameInferenceUseCase`
- `RunFrontOverviewInferenceUseCase`
- `ApplyPlantDecisionGateUseCase`
- `MapSideScanPositionUseCase`
- `MapFrontOverviewDetectionsUseCase`
- `RecordDecisionUseCase`
- `ExportRunCsvUseCase`
- `ExportRunJsonlUseCase`
- `ExportRunBundleUseCase`
- `RunModelBenchmarkUseCase`
- `ValidateModelBundleUseCase`

## Repositories

Domain interfaces:

```kotlin
interface RunRepository {
    suspend fun createRun(config: RunConfig): RunId
    suspend fun markRunCompleted(runId: RunId)
    suspend fun appendDecision(decision: RecordedDecision)
    fun observeRun(runId: RunId): Flow<Run>
    fun observeLatestRun(): Flow<Run?>
    fun observeRunSummary(runId: RunId): Flow<RunSummary>
}

interface FieldLayoutRepository {
    fun observeLayouts(): Flow<List<FieldLayout>>
    fun observeActiveLayout(): Flow<FieldLayout>
    suspend fun saveLayout(layout: FieldLayout)
    suspend fun setActiveLayout(fieldId: FieldId)
}

interface InferenceRepository {
    suspend fun loadModel(profile: ModelProfile): ModelLoadResult
    suspend fun classifySideFrame(frame: ImageFrame, config: InferenceConfig): FramePrediction
    suspend fun analyzeFrontOverview(frame: ImageFrame, config: FrontOverviewConfig): FrontOverviewResult
    fun observeModelStatus(): Flow<ModelStatus>
}

interface ExportRepository {
    suspend fun exportCsv(runId: RunId): ExportedFile
    suspend fun exportJsonl(runId: RunId): ExportedFile
    suspend fun exportBundle(runId: RunId): ExportedFile
}
```

## Android Runtime Architecture

### Scan State Machine

```text
Idle
  -> PermissionRequired
  -> Ready
  -> Starting
  -> Recording
  -> Paused
  -> Recording
  -> Stopping
  -> Completed

Any state -> Failed when unrecoverable camera/model/storage failure occurs.
Failed -> Ready after user acknowledges and the blocker is fixed.
```

### Camera Pipeline

```text
CameraX Preview + ImageAnalysis
  -> FrameSource
  -> FrameNormalizer
  -> Mode-specific Analyzer
  -> Inference Engine
  -> Decision Gate / Front Aggregator
  -> Mapper
  -> Run Writer
  -> UI StateFlow
```

### Concurrency Model

- Use `viewModelScope` for UI-triggered scan commands.
- Use a scan-scoped `CoroutineScope` created by the scan controller.
- Do not use `GlobalScope`.
- Use a single CameraX analyzer executor or coroutine channel with capacity 1.
- Use `Dispatchers.Default` for CPU inference and preprocessing.
- Use `Dispatchers.IO` for Room/file export.
- Use `StateFlow` for durable UI state.
- Use `SharedFlow` for one-time UI effects such as haptic feedback, permission prompt, export complete, and error snackbar.
- Cancel active scan scope on Stop.
- Always close `ImageProxy` in `finally`.

### Backpressure

The analyzer must never build an unbounded queue. Use one of:

- CameraX keep-latest backpressure strategy.
- A conflated coroutine channel.
- Atomic latest-frame slot consumed by a single inference worker.

If inference is slower than the target selected FPS:

- Drop frames.
- Keep the UI responsive.
- Count late frames.
- Reduce selected FPS or resolution when thermal/performance policy requires it.

## Technology Stack

### Android

- Kotlin.
- Jetpack Compose for UI.
- CameraX for camera preview, analysis, and capture.
- Room for local database.
- DataStore for small settings.
- Kotlin coroutines and Flow for runtime state.
- Hilt for dependency injection.
- WorkManager only for non-camera background tasks such as export cleanup or deferred report packaging. Do not use WorkManager for live camera scanning.

### ML Runtime

- Raw TensorFlow Lite Interpreter execution on Android.
- CPU default using XNNPACK-style optimized CPU execution when available.
- Use raw TFLite Interpreter for v1. Do not use Task Library in the first implementation.
- Optional GPU delegate profile behind a setting or benchmark gate, not required for field operation.
- Memory-mapped model loading from assets or app-private model bundles.
- Native C++ or optimized Kotlin postprocessing for detector NMS if Kotlin is too slow.

### Computer Vision Support

Start with lightweight in-app preprocessing:

- YUV/RGB conversion.
- Resize.
- Center crop.
- Vegetation/mask crop parity with Pi.
- Bounding box transforms.

Do not add OpenCV for Side Scan. Do not add OpenCV for the Front Overview Burst MVP. Use CameraX frames, TFLite detector boxes, guide rails, and Kotlin geometry first. Add OpenCV only in Front Overview production hardening after a written benchmark shows the no-OpenCV approach misses the row-assignment acceptance gates.

## Model Conversion Plan

## Current Model Inputs

The repo currently has PyTorch model files under `models/`:

- `classifier.pt`
- `classifier_deploy_fastcrop.pt`
- `detector.pt`
- `detector_nano_256.pt`
- `combined_crop_detector_s_256.pt`
- `tomato_crop_detector_s_256.pt`
- `tomato_crop_detector_m_512_precise.pt`
- `tomato_crop_detector_x_768_finetune.pt`

The Pi runtime prefers the calibrated fast-crop classifier and OpenVINO export, but `runtime_exports/` is ignored and not present in this checkout. Android needs its own export lane.

  Verified classifier labels (from both `.pt` files): `0 Early_blight`, `1 Healthy`, `2 Late_blight`, `3 Leaf Miner`, `4 Magnesium Deficiency`, `5 Nitrogen Deficiency`, `6 Pottassium Deficiency`, `7 Spotted Wilt Virus`.

## Android Model Targets

Create these Android artifacts:

```text
android_model_exports/
  classifier/
    classifier_fastcrop_float32.tflite
    classifier_fastcrop_float16.tflite
    classifier_fastcrop_int8.tflite
    labels.json
    preprocessing.json
    evaluation.json
  detector/
    detector_nano_256_float32.tflite
    detector_nano_256_float16.tflite
    detector_nano_256_int8.tflite
    labels.json
    postprocessing.json
    evaluation.json
  bundle/
    agribot_model_bundle_v001.zip
    manifest.json
    manifest.sig
```

Do not commit large generated model files unless the repo intentionally moves to Git LFS or release assets.

## Model Manifest

`model_manifest.json` is a new Android-specific format, aligned with labels, preprocessing, hashes, thresholds, and source model metadata. It does not need to match an existing Pi or dashboard manifest format.

Each app build or model bundle must include:

```json
{
  "bundle_id": "agribot-model-bundle-v001",
  "created_at": "2026-06-08T00:00:00Z",
  "models": {
    "classifier": {
      "file": "classifier_fastcrop_int8.tflite",
      "source": "models/classifier_deploy_fastcrop.pt",
      "input_width": 224,
      "input_height": 224,
      "color_space": "RGB",
      "normalization": "ultralytics_classify_default",
      "labels_file": "labels.json",
      "confidence_threshold": 0.765,
      "high_confidence_threshold": 0.765
    },
    "detector": {
      "file": "detector_nano_256_int8.tflite",
      "source": "models/detector_nano_256.pt",
      "input_width": 256,
      "input_height": 256,
      "postprocess": "yolo_nms"
    }
  },
  "compatibility": {
    "min_android_sdk": 26,
    "cpu_default": true,
    "requires_network": false
  }
}
```

## Conversion Steps

1. Freeze current PyTorch baseline:
   - Record source model paths.
   - Record label order.
   - Record preprocessing parameters.
   - Run current Python accuracy evaluation on `clf_dataset/test`.

  Detector status: use `detector_nano_256.pt` as the first Android detector candidate. Convert and benchmark it before building Front Overview. If it misses the detector gates below, replace it with a smaller retrained detector before continuing Front Overview.

 2. Export classifier to TensorFlow Lite formats:
   - Float32 for reference.
   - Float16 for optional accelerator profile.
   - Int8 for CPU size/speed profile if accuracy holds.
3. Export detector to TensorFlow Lite formats:
   - Start with nano detector for mobile CPU.
   - Keep detector optional in Side Scan Mode.
   - Require detector for Front Row Overview Mode.
4. Build calibration dataset:
   - Include healthy and disease classes.
   - Include side-scan close crops.
   - Include front-overview crops at one-meter distance.
   - Include lighting, blur, and field backgrounds.
5. Validate conversion drift:
   - Compare PyTorch outputs to TFLite outputs on same images.
   - Track top-1 agreement, confidence delta, emitted gate accuracy, coverage.
6. Benchmark on Android:
   - CPU threads 1, 2, 4.
   - Analysis resolutions 640x480 and 1280x720.
   - Side classifier-only.
   - Side detector-plus-classifier.
   - Front detector-plus-classifier.
7. Select production profile:
   - Choose fastest profile that meets accuracy/coverage gates.
   - Keep fallback profile if int8 accuracy drifts.

## Model Acceptance Gates

Classifier:

- Top-1 TFLite vs PyTorch agreement: target >= 99 percent on calibration/eval set.
- Raw Top-1 accuracy on `clf_dataset/test` must be >= `95.0%` before Side Scan pilot.
- Gated emitted accuracy must be >= `97.0%` with coverage >= `96.0%` on the comparable side-scan validation set before Side Scan pilot.
- Mean confidence drift vs PyTorch must be <= `0.03`; p95 absolute confidence drift must be <= `0.08`.
- Android CPU average classifier inference latency must be <= `160 ms`; p95 must be <= `200 ms` on the named target phone.

Detector:

- `detector_nano_256.pt` is the first candidate for Front Row Overview. It is required to pass the Android CPU detector gates here before Front Overview exits Phase 5. It remains disabled in Side Scan by default.
- Front Overview detector acceptance before implementation moves past Phase 5: recall >= `0.85` on labeled front-overview plant/crop regions, precision >= `0.75`, image-level plant-hit rate >= `0.95`, and Android CPU detector latency p95 <= `300 ms` at the chosen analysis resolution.
- False positives on soil/background must average <= `2` extra candidate crops per accepted burst after NMS/filtering.
- Reject candidate boxes before classification when box area is below `1.0%` of analysis frame area, shortest side is below `24 px`, or aspect ratio is outside the configured plant/canopy range.

End-to-end:

- Side Scan Mode must reproduce Pi behavior on the same still images or replay video within the classifier parity gates above.
- Front Overview Mode must report coverage and uncertainty honestly: ambiguous row assignment saves `status = uncertain`, `row_side = unknown`, and `reason = ambiguous_geometry`; it must not guess a row/plant number when geometry confidence is below `0.70`.

## Inference Implementation Details

### Classifier Preprocessing

Define a single `ClassifierPreprocessor` and test it against Python:

- Input color: RGB.
- Resize method: match training/export expectations.
- Crop: side mode uses mask/center/full before classifier resize.
- Normalization: match Ultralytics/TFLite export exactly.
- Output labels: from model metadata or bundled labels file.

### Detector Postprocessing

Define a `DetectorPostprocessor`:

- Decode output tensor shape from model manifest.
- Apply confidence threshold.
- Convert boxes from model coordinates to analysis-frame coordinates.
- Apply non-max suppression.
- Filter boxes by minimum area and aspect ratio.
- Return sorted detections by confidence.

### Decision Gate Port

Port `PlantDecisionGate` as pure Kotlin and test all cases:

- primary high-confidence agreement emits first label.
- primary low-confidence waits for backup.
- backup majority emits label.
- all low-confidence emits `Uncertain`.
- tied labels emit `Uncertain`.
- disease vs healthy disagreement emits `Uncertain` unless backup creates majority.

### Front Overview Aggregation

Create `FrontOverviewAggregator`:

- Input: list of detections/classifications from a burst.
- Group by row side and estimated plant index.
- Merge boxes with IoU or nearest-center matching.
- Aggregate class labels with high-confidence rules.
- Emit one decision per plant group.
- Preserve per-frame evidence for diagnostics.

## CameraX Implementation Plan

### Camera Use Cases

- `Preview`: always visible during scan.
- `ImageAnalysis`: feeds inference.
- `ImageCapture`: saves optional evidence frame, manual snapshots, and front overview burst key frame.

### Analyzer Settings

- Use `ImageAnalysis` because it provides CPU-accessible images for image processing and ML.
- Use output format deliberately:
  - Start with YUV for lower overhead if conversion is optimized and tested.
  - Consider RGBA if it reduces code complexity and the performance hit is acceptable.
- Use one analyzer at a time per mode.
- Bind all concurrent use cases in a single lifecycle bind call.
- Inspect actual returned analysis resolution at runtime and record it in diagnostics.

### Rotation And Coordinate Mapping

The app must correctly map:

- Sensor coordinates.
- Analysis buffer coordinates.
- PreviewView display coordinates.
- Model input coordinates.
- Detection boxes back to preview overlay and export schema.

Create tests for coordinate transforms using known image sizes and rotations.

### Foreground And Background Policy

The normal scan should run while the app is visible and the user has pressed Start.

If later requirements demand scanning while the app is not visible:

- Use a camera foreground service only after explicit user action.
- Show an ongoing notification.
- Declare the correct foreground service type and permissions.
- Respect Android while-in-use camera permission restrictions.
- Do not attempt boot-start camera scanning as a production assumption.

## Local Data And Storage

## Room Tables

### `field_layouts`

- `id`
- `name`
- `active_row_id`
- `start_plant`
- `plant_step`
- `plant_cooldown_sec`
- `created_at`
- `updated_at`
- `is_active`

### `field_rows`

- `id`
- `field_layout_id`
- `row_id`
- `row_index`
- `plants_per_row`
- `plant_spacing_m`
- `row_spacing_m`
- `y_m`

### `scan_runs`

- `run_id`
- `mode`
- `field_layout_id`
- `started_at`
- `completed_at`
- `state`
- `note`
- `model_bundle_id`
- `camera_id`
- `analysis_width`
- `analysis_height`
- `target_fps`
- `app_version`

### `decisions`

- `id`
- `run_id`
- `sequence`
- `timestamp`
- `mode`
- `field_id`
- `row_id`
- `row_side`
- `row_index`
- `plant_column`
- `plant_number`
- `plant_key`
- `x_m`
- `y_m`
- `label`
- `confidence`
- `raw_label`
- `status`
- `action`
- `frames_used`
- `reason`
- `late_frames`
- `gate_avg_latency_ms`
- `gate_max_latency_ms`
- `model_version`
- `manual_override`
- `notes`

### `front_geometry`

- `decision_id`
- `capture_segment_id`
- `bbox_left`
- `bbox_top`
- `bbox_right`
- `bbox_bottom`
- `center_x`
- `center_y`
- `row_side`
- `row_assignment_confidence`
- `plant_position_confidence`
- `geometry_reason`
- `camera_distance_m`
- `camera_height_m`
- `camera_tilt_degrees`

### `run_events`

- `id`
- `run_id`
- `timestamp`
- `type`
- `payload_json`

Examples: started, paused, resumed, stopped, permission_lost, model_loaded, thermal_warning, export_created, manual_override.

Manual override events must use this payload shape:

```json
{
  "decision_id": "decision_123",
  "operator": "local_user",
  "reason": "leaf partly hidden; corrected after visual inspection",
  "original": {
    "label": "Uncertain",
    "status": "uncertain",
    "confidence": 0.42
  },
  "updated": {
    "label": "Healthy",
    "status": "manual",
    "confidence": 1.0
  }
}
```

### `model_bundles`

- `bundle_id`
- `installed_at`
- `classifier_file`
- `detector_file`
- `labels_hash`
- `manifest_json`
- `signature_status`
- `active`

## Export Contract

Android exports should produce:

```text
agribot_run_<run_id>/
  metadata.json
  summary.json
  decisions.jsonl
  events.csv
  field_layout.json
  model_manifest.json
  diagnostics.json
  evidence_frames/
```

Evidence frames are saved as JPEG for sick and uncertain decisions by default, plus manual snapshots. Do not save every raw frame. Stop adding new evidence frames after 500 MB or 2,000 evidence frames per run, whichever comes first, and continue recording decisions with `evidence_status = "storage_cap_reached"`.

## JSONL Decision Example

```json
{
  "run_id": "android_20260608_145500",
  "sequence": 12,
  "timestamp": "2026-06-08T14:55:21+05:30",
  "mode": "SIDE_SCAN",
  "field_id": "Field 2",
  "row_id": "B",
  "row_index": 2,
  "plant_column": 12,
  "plant_number": 12,
  "plant_key": "Field 2|B|12",
  "x_m": 3.85,
  "y_m": 1.4,
  "label": "Healthy",
  "confidence": 0.918234,
  "status": "ok",
  "action": "None",
  "frames_used": 2,
  "reason": "primary_agreement",
  "late_frames": 0,
  "gate_avg_latency_ms": 84.2,
  "gate_max_latency_ms": 88.1,
  "model_version": "agribot-model-bundle-v001"
}
```

Front mode extends with:

```json
{
  "mode": "FRONT_ROW_OVERVIEW",
  "row_side": "left",
  "bbox_px": [120, 220, 260, 420],
  "geometry_confidence": 0.82,
  "plant_position_confidence": 0.76,
  "geometry_reason": "row_rail_fit_and_spacing_estimate"
}
```

## UI Plan

## Main Home Screen

Primary elements:

- Large Start button.
- Mode segmented control:
  - Side Scan.
  - Front Overview.
- Active field and row/corridor selector.
- Latest run summary.
- Model status chip.
- Camera permission status.
- Storage/battery/thermal status.

No landing-page style content. This is a field tool; first screen should be operational.

## Recording Screen

Common:

- Full-screen camera preview.
- Top status bar: mode, field, row/corridor, run timer.
- Large current target: "Row B / Plant 12" or "Left A + Right B segment".
- Start/Pause/Stop controls.
- Latest decision panel.
- Small performance strip: FPS, latency, late frames.

Side Scan:

- Current plant target.
- Retake current plant.
- Skip current plant.
- Move next/previous plant.
- Auto-advance toggle.

Front Overview:

- Left/right row guide rails.
- Capture Burst button.
- Box overlay and row-side colors.
- Review mapping button.
- Commit segment button if review is required.

## Field Setup Screen

- Field list.
- Add/edit field.
- Row count.
- Plants per row.
- Row spacing.
- Plant spacing.
- Per-row overrides.
- Active row.
- Front Overview corridor setup:
  - left row.
  - right row.
  - nearest plant numbers.
  - camera distance guide.

## Run History Screen

- List runs by date, field, mode, sick count, uncertain rate.
- Open run detail.
- Export CSV.
- Export JSONL bundle.
- Delete run with confirmation.

## Run Detail Screen

- Summary metrics.
- Disease labels chart.
- Field map.
- Decision table.
- Filters: sick, uncertain, manual, skipped.
- Front Overview geometry overlay for front-mode runs.
- Manual correction tools.

## Diagnostics Screen

- App version.
- Model bundle version.
- Camera ID and supported sizes.
- Current analysis resolution.
- CPU thread profile.
- Latest benchmark result.
- Thermal/battery warnings.
- Storage use.
- Permission status.
- Export logs.

## Settings Screen

Settings should be grouped by risk:

Basic:

- Language.
- Default mode.
- Active field.
- Save evidence frames on sick decisions.
- Export location.

Advanced:

- Confidence threshold.
- High-confidence gate.
- Selected FPS.
- Max edge.
- Crop mode.
- CPU threads.
- Detector enabled in Side Scan.
- Optional GPU delegate toggle, disabled by default until benchmarked.

Every advanced setting should have reset-to-default.

## Accessibility And Field Usability

- Responsive layout targeting at least `360dp` width, `48dp` minimum touch targets, scalable text, and responsive overlays.
- Large touch targets.
- High contrast outdoors.
- No tiny-only controls for Start/Stop.
- Text labels plus icons.
- Haptics for key result states.
- Works with gloves as much as possible through large buttons.
- Localization strings exist in `i18n.js`, but need audit/repair before native use. Start with English; then migrate validated Hindi/Bengali/Telugu/Marathi/etc. strings.
- No reliance on internet fonts.

## Security, Privacy, And Safety

## Data Security

- Store data in app-private storage by default.
- Use Android Storage Access Framework for user-selected exports.
- Do not upload images or decisions unless a future explicit sync feature is built.
- Avoid `INTERNET` permission in the production direct-replacement app unless a real network feature remains.
- Remove production dependence on cleartext HTTP.
- Validate imported model bundles and field-layout files.
- Never hardcode secrets.

## Model Bundle Integrity

V1 model handling:

- Models are embedded in app assets or an install-time local bundle.
- OTA model downloads are not part of v1.
- Every app build must include `model_manifest.json`.
- Manifest must include SHA-256 of model files and label files.
- App must verify SHA-256 before loading a model.
- Signature verification is not required for embedded v1 assets.
- If later local model import is added, failed validation must keep the previous working model active.

## Permission Requirements

- `CAMERA` runtime permission.
- Storage permissions should be avoided by using app-private storage and Storage Access Framework.
- Foreground service camera permission only if the app supports user-visible background camera scanning.

## Safety Policy

- This app gives plant-health screening, not guaranteed agronomy diagnosis.
- UI should use action wording like "Inspect or treat" rather than overclaiming certainty.
- Uncertain and manual states must be preserved.
- Manual corrections should be auditable.

## Performance Plan

## Baseline Targets

Side Scan CPU baseline:

- Selected frame schedule: 5 FPS.
- Average per-frame inference target: <= 160 ms.
- Hard per-frame ceiling: <= 200 ms for selected frames.
- Decision emit after 2 to 3 selected frames.
- No analyzer backlog.

Side Scan performance profile:

- Selected frame schedule: 10 FPS on capable phones.
- Average per-frame inference target: <= 90 ms.
- Automatic fallback to 5 FPS if thermal or late-frame count rises.

Front Overview baseline:

- Burst of 5 to 9 frames over 1 to 2 seconds.
- Segment processing target: <= 3 seconds for first production version.
- Continuous mode can come later after burst mode is stable.

Performance reference from Pi runtime (`RUNTIME_TESTING.md`): raw frame accuracy `95.66%`, gated emitted accuracy `97.32%` with `96.96%` coverage, `12.53 FPS` / `80 ms` avg latency normal profile, `114 ms` low-power, `73 ms` at 10 FPS profile.

## Thermal Policy

Collect:

- Average latency.
- Max latency.
- Late frames.
- Device thermal status where available.
- Battery level.
- Run duration.

Actions:

- A selected frame is `preferred_late` when end-to-end analysis latency exceeds `180 ms`.
- A selected frame is `hard_late` when end-to-end analysis latency exceeds `200 ms`.
- If `hard_late` frames are >= `5` in any rolling window of `30` selected frames, reduce Side Scan target FPS from `10` to `5` or from `5` to `3`.
- If rolling average latency is > `160 ms` for `30` selected frames in Side Scan, reduce selected FPS by one profile step.
- If still late after FPS reduction, reduce analysis resolution one profile step.
- If still late, keep Side Scan classifier-only and leave detector disabled.
- On Android thermal status `MODERATE`, force low-power profile: Side Scan `5 FPS`, front burst only, no GPU delegate.
- On Android thermal status `SEVERE` or worse, pause scan after the current decision group, persist run state, and show a resume-after-cooldown prompt.
- Never silently drop decisions without showing status.

## Memory Policy

- Memory-map model files.
- Reuse input/output buffers.
- Avoid allocating bitmaps for every frame.
- Keep only latest frame in active inference queue.
- Save evidence frames only on selected decisions and only if enabled.
- Cap export bundle size or warn user before export.

## Testing Strategy

Use test-first development for all critical logic. The implementation should not ship with only a manually tested APK.

## Unit Tests

Pure Kotlin:

- Decision gate.
- Field layout normalization.
- Row ID generation.
- Side scan mapper.
- Front row assignment math.
- Confidence/action mapping.
- Run summary reducer.
- Export serializers.
- Model manifest parser.
- Settings validation.

ML preprocessing:

- YUV/RGB conversion test fixtures.
- Resize/crop parity with Python.
- Mask crop parity with Python on known images.
- Label mapping.
- Detector postprocessing/NMS with fixed tensors.

## Integration Tests

- Room migrations.
- Run creation and decision append.
- Export creates valid CSV/JSONL.
- Model bundle validation.
- Camera frame analyzer can be fed synthetic frames.
- ViewModels update StateFlow correctly for scan state changes.

## Instrumentation Tests

- Permission flow.
- Home screen Start button.
- Side Scan start/pause/resume/stop.
- Field layout editor.
- Run history export.
- Manual correction flow.
- Front Overview review flow.

## Golden Dataset Tests

Create a mobile golden test folder:

```text
android_test_assets/
  side_scan/
    healthy/
    disease/
    uncertain/
  front_overview/
    clear_two_rows/
    occluded/
    low_light/
    ambiguous_geometry/
  expected/
    side_scan_expected.json
    front_overview_expected.json
    front_overview_manifest.jsonl
```

`front_overview_manifest.jsonl` uses one JSON object per fixture:

```json
{
  "fixture_id": "front_clear_two_rows_001",
  "source_file": "front_overview/clear_two_rows/front_clear_two_rows_001.jpg",
  "mode": "FRONT_ROW_OVERVIEW",
  "reviewer": "local_reviewer",
  "reviewed_at": "2026-06-08",
  "camera_pose": {
    "distance_m": 1.0,
    "height_m": 1.1,
    "tilt_degrees": 15
  },
  "expected": [
    {
      "bbox_px": [120, 220, 260, 420],
      "row_side": "left",
      "row_id": "A",
      "plant_number": 12,
      "label_group": "plant_candidate",
      "geometry_status": "clear"
    }
  ],
  "acceptance": {
    "row_side_required": true,
    "plant_number_required": true,
    "allow_unknown": false
  }
}
```

Golden tests should verify:

- Expected labels.
- Confidence within tolerance.
- Row/plant mapping.
- Uncertain behavior.
- Front geometry assignment.

## Device Test Matrix

Development can start without a named phone, but field pilot cannot. The implementation must support this matrix:

- Emulator: unit/UI smoke only, no performance claims.
- Baseline real phone: Android 8+ arm64, 4 GB RAM, rear normal camera.
- Reference real phone: mid-range Android 12+ arm64, 6-8 GB RAM.
- Modern real phone: Android 14+ arm64 for target SDK and thermal behavior checks.

Before Phase 3 exit, record the exact baseline and reference device models in the benchmark report.

## Coverage Target

- Overall code coverage target: 80 percent or better for domain/data/ml logic.
- UI coverage should focus on critical flows rather than chasing line coverage.
- ML numeric behavior should be protected by golden tests and benchmark reports.

## Verification Commands

Expected local commands after implementation:

```powershell
cd agribot_android_app\android
.\gradlew.bat testDebugUnitTest
.\gradlew.bat connectedDebugAndroidTest
.\gradlew.bat lintDebug
.\gradlew.bat assembleDebug
```

If Gradle modules are added:

```powershell
.\gradlew.bat :domain:test
.\gradlew.bat :data:test
.\gradlew.bat :ml:test
.\gradlew.bat :app:assembleDebug
```

## CI/CD Plan

## GitHub Actions

Create Android CI:

- JDK setup.
- Android SDK setup.
- Gradle cache.
- Unit tests.
- Lint.
- Detekt/ktlint if added.
- Assemble debug.
- Upload APK artifact.

Required before production release:

- Emulator instrumentation smoke test.
- Dependency vulnerability scan.
- Signed release build using encrypted CI secrets or a documented local signing handoff.

## Release Build

Release build requirements:

- Minification enabled after model/runtime code is stable.
- Proguard rules for TensorFlow Lite, CameraX, Room, and Hilt.
- Signed APK/AAB.
- Versioned model bundle.
- Release notes including model version and known limits.

## Migration Plan

## Phase 0: Current-State Audit

Deliverables:

- Record current Pi behavior and thresholds from `RUNTIME_TESTING.md`, `inference_rpi.py`, and `43_low_power_5v3a_experiment.py`.
- Record current model labels and preprocessing from the `.pt` artifacts and Python mask crop implementation.
- Record the default target profile: min SDK 26, arm64-v8a, rear normal camera, raw TFLite Interpreter, Side Scan classifier-only default.
- Record Pi-dashboard compatibility as export compatibility only, not a production WebView dependency.

Exit criteria:

- Written mapping of Pi script behavior to Android features.
- Written model export path.
- Written target device profile and benchmark report template.

## Phase 1: Native Android Foundation

Deliverables:

- Native Kotlin/Compose app replaces `BridgeActivity`.
- Camera permission flow.
- Home screen with mode selection and Start button.
- Field layout storage with Room/DataStore.
- Run history skeleton.
- Basic export skeleton.

Exit criteria:

- APK builds.
- App opens to operational home screen.
- Field layout can be created and saved.
- Unit tests pass for domain models and field layout.

## Phase 2: Model Export And Mobile Inference

Deliverables:

- Export classifier to TensorFlow Lite artifact.
- Export detector to TensorFlow Lite artifact.
- Add model manifest and labels.
- Android classifier inference wrapper.
- Android detector wrapper and NMS.
- Python vs Android golden comparison.

Exit criteria:

- Classifier runs on Android CPU.
- Detector runs on Android CPU.
- Conversion drift report exists.
- Model benchmark screen or command path exists.

## Phase 3: Side Scan Mode MVP

Deliverables:

- CameraX preview and ImageAnalysis.
- Side scan preprocessing parity.
- CPU classifier inference.
- Kotlin `PlantDecisionGate`.
- Row-major mapper.
- Room run writer.
- Live recording screen.
- CSV/JSONL export matching Pi schema.

Exit criteria:

- One-button side scan works without Pi.
- Decisions are emitted and mapped.
- Exports can be opened by existing tooling.
- Unit/integration/instrumentation tests pass.
- 5 FPS selected-frame target verified on target phone.

## Phase 4: Side Scan Production Hardening

Deliverables:

- Retake/skip/manual correction.
- Pause/resume recovery.
- Thermal and late-frame policy.
- Evidence-frame capture on sick/uncertain decisions.
- Run summary and field map.
- Localization.
- Accessibility pass.

Exit criteria:

- App survives app pause/resume and process restart without losing completed run data.
- Farmer can complete a row using only the phone.
- Export contains all expected metadata and decisions.

## Phase 5: Front Row Overview Burst Mode

Deliverables:

- Front mode setup screen.
- Row guide overlay.
- Detector-based multi-plant detection.
- Left/right row assignment.
- Burst aggregation.
- Review and correction screen.
- Geometry diagnostics in export.

Exit criteria:

- Captures one-meter front overview burst.
- Assigns clear detections to left/right rows.
- Marks ambiguous captures uncertain.
- Supports manual correction before commit.
- Field pilot data can be reviewed and exported.

## Phase 6: Front Row Overview Production Hardening

Deliverables:

- Better row geometry estimation.
- Segment merge support.
- Continuous corridor scan experiment.
- More robust occlusion/blur/lighting handling.
- Front-mode validation dataset and benchmark report.

Exit criteria:

- Front mode produces useful row-side maps under realistic field conditions.
- Quality gates prevent bad confident mappings.
- Performance is acceptable on target phones.

## Phase 7: Field Pilot

Deliverables:

- Pilot protocol.
- Device checklist.
- Field checklist.
- Failure logging.
- Export review.
- Model error analysis.

Exit criteria:

- At least one full field or row set scanned in Side Scan Mode.
- At least several front-overview segments scanned and reviewed.
- Known failure modes documented.
- Next model/data collection needs identified.

## Phase 8: Release

Deliverables:

- Signed release APK/AAB.
- Model bundle manifest.
- User guide.
- Field troubleshooting guide.
- CI green.
- Security/privacy review complete.

Exit criteria:

- Release build runs on target devices.
- No Pi required.
- User can start, scan, stop, review, and export.

## Detailed Implementation Backlog

## App Foundation

- Replace Capacitor activity with Kotlin `MainActivity`.
- Add Compose Material 3 theme.
- Add navigation graph.
- Add DI.
- Add Room database.
- Add DataStore settings.
- Add file export provider.
- Add app-private directories:
  - `models/`
  - `runs/`
  - `exports/`
  - `evidence_frames/`

## Domain

- Add value classes for IDs.
- Add field layout normalization.
- Add row ID generation.
- Add side mapper.
- Add decision gate.
- Add run summary reducer.
- Add export DTOs.

## Camera

- Add CameraX dependencies.
- Implement `CameraController`.
- Implement Preview composable.
- Implement analysis frame stream.
- Implement frame conversion and rotation.
- Add overlay coordinate transforms.
- Add camera diagnostics.

## ML

- Add model manifest parser.
- Add model asset loader.
- Add classifier interpreter.
- Add detector interpreter.
- Add preprocessing.
- Add NMS.
- Add inference benchmarks.
- Add model status reporting.

## Side Scan

- Add Side Scan ViewModel.
- Add side recording screen.
- Connect frame stream to classifier.
- Connect decision gate.
- Connect mapper.
- Append decisions to Room.
- Add haptic/sound effects.
- Add retake/skip/manual.

## Front Overview

- Add setup flow.
- Add guide rail overlay.
- Add burst capture.
- Add detector-based plant grouping.
- Add row-side assignment.
- Add classifier on each crop.
- Add review UI.
- Add geometry export.

## History And Export

- Add run list.
- Add run detail.
- Add field map.
- Add CSV export.
- Add JSONL export.
- Add bundle export.
- Add share/open exported file.

## Diagnostics

- Add model benchmark.
- Add camera support screen.
- Add storage usage.
- Add thermal/battery status.
- Add app logs export.

## Algorithm Specifications

## Side Scan Pseudocode

```text
onStartSideScan(config):
  runId = createRun(config)
  mapper = SideScanMapper(fieldLayout, startScanIndex)
  gate = PlantDecisionGate(highConf = config.highConfidence)
  startCamera(mode = SIDE_SCAN)

onFrame(imageProxy):
  if not shouldSampleFrame(targetFps):
    close imageProxy
    return

  frame = normalizeFrame(imageProxy)
  prediction = classifier.classifySideFrame(frame, config)
  gate.add(prediction)

  if gate.hasDecision:
    position = mapper.position(scanIndex)
    decision = gate.decision
    record = buildRecordedDecision(runId, decision, position, gate.frames)
    save(record)
    emitUiEffect(decision)
    scanIndex += 1
    gate.reset()
    wait plantCooldownSec if auto-paced
```

## Front Overview Pseudocode

```text
onCaptureFrontBurst(config):
  frames = captureBurst(count = config.burstFrameCount)
  detectionsByFrame = []

  for frame in frames:
    normalized = normalizeFrame(frame)
    detections = detector.detectPlants(normalized)
    geometry = estimateRows(detections, config.guideRails, config.fieldGeometry)
    classified = []
    for detection in detections:
      crop = cropDetection(normalized, detection)
      classification = classifier.classify(crop)
      mapped = assignToRowAndPlant(detection, geometry, config)
      classified.add(mapped + classification)
    detectionsByFrame.add(classified)

  plantGroups = aggregateAcrossFrames(detectionsByFrame)
  decisions = []
  for group in plantGroups:
    decision = decideFromGroup(group)
    if group.geometryConfidence < threshold:
      decision.status = UNCERTAIN
      decision.reason = "ambiguous_geometry"
    decisions.add(decision)

  if any decision requires review:
    showReview(decisions)
  else:
    save(decisions)
```

## Risk Register

## High Risks

- TensorFlow Lite conversion changes classifier accuracy or label order.
- Detector on Android CPU may be too slow for continuous Front Overview.
- Front Overview geometry may be unreliable without calibration or row guides.
- Field lighting and motion blur can produce false disease or false healthy decisions.
- APK size may become large if multiple models or OpenCV are bundled.
- Thermal throttling can make 10 FPS unrealistic on lower-end phones.

## Mitigations

- Keep classifier-only Side Scan as first production mode.
- Benchmark multiple model precisions.
- Use burst mode for Front Overview before continuous scan.
- Add quality gates and review before committing uncertain geometry.
- Save model manifest and exact preprocessing.
- Add device-specific performance profiles.
- Use optional evidence-frame saving for error analysis.

## Resolved Implementation Decisions

This section records decisions that would otherwise become implementation blockers. They are resolved for v1.

1. Target phones: develop against the device profile in `Pre-Development Decisions`; before field pilot, record exact physical baseline and reference phones in benchmark output.
2. Distribution: pilot through internal sideload APKs; production signing and Play Store/AAB work happen in release phase.
3. Internet: native v1 is offline-only and must not request `INTERNET`.
4. Pi dashboard: preserve export compatibility with `agribot_platform/`; do not keep Pi dashboard or WebView as the production runtime.
5. Classifier labels: `0 Early_blight`, `1 Healthy`, `2 Late_blight`, `3 Leaf Miner`, `4 Magnesium Deficiency`, `5 Nitrogen Deficiency`, `6 Pottassium Deficiency`, `7 Spotted Wilt Virus`.
6. Front Overview unit of work: detect individual plant/crop regions, assign them to left/right rows of a walking corridor, classify each crop, and aggregate per mapped plant.
7. Evidence retention: JPEG for sick and uncertain decisions by default, plus manual snapshots; cap at 500 MB or 2,000 evidence frames per run.
8. Model updates: embedded/install-time models for v1; no OTA download infrastructure.
9. Mask crop: port `inference_rpi.py::mask_crop` exactly and prove parity with golden tests before Side Scan exit.
10. Detector gate: Front Overview cannot pass Phase 5 unless detector recall, precision, hit rate, false positive, and latency gates in `Model Acceptance Gates` pass.
11. Manual evidence capture: Side Scan MVP must support manual snapshot capture in addition to automatic sick/uncertain evidence capture.
12. Front golden assets: use the `android_test_assets/front_overview/` structure in this plan; each asset must include source image/video, expected detections, expected row side, expected plant mapping or `unknown`, and reviewer initials/date.

## Industry-Scale Acceptance Criteria

The app is ready for real deployment only when all of these are true:

- A farmer can use the phone alone to complete a Side Scan run without Pi or laptop.
- The app starts from one button and gives clear Stop/Pause/Resume controls.
- The model runs locally on CPU.
- The app stores durable run data and exports it.
- Side Scan output is schema-compatible with Pi decisions.
- Front Overview supports one-meter corridor captures with left/right row assignment and uncertainty handling.
- The app has a field layout editor.
- The app has run history and export.
- The app has diagnostics for model, camera, latency, battery, and storage.
- Unit, integration, and instrumentation tests cover critical logic.
- Golden dataset tests verify model/preprocessing behavior.
- Benchmarks prove selected FPS and latency on target devices.
- Security review confirms no hardcoded secrets, no unnecessary network dependency, and safe local storage/export.
- Release build is signed and reproducible.

## Handoff Instructions For Future AI Agents

When implementing this plan:

1. Re-read this `plan.md`, `README.md`, `RUNTIME_TESTING.md`, `inference_rpi.py`, `43_low_power_5v3a_experiment.py`, `44_farmer_one_touch.py`, `farmer_config.json`, `agribot_platform/run_platform.py`, and `agribot_android_app/README_APK.md`.
2. Do not treat the current Capacitor app as already meeting the requirement. It is a Pi dashboard client, not a Pi replacement.
3. Start with native Android foundation and Side Scan Mode before Front Overview.
4. Port and test the decision gate exactly.
5. Preserve the Pi export schema unless there is a documented reason to extend it.
6. Make CPU inference the default.
7. Mark uncertain cases honestly.
8. Do not delete or revert existing dirty work unless explicitly instructed.
9. Use test-first development for domain, mapping, preprocessing, and export logic.
10. Verify every completion claim with current commands and, for app work, emulator or real-device evidence.

## Primary External References To Verify During Implementation

- CameraX Image Analysis: https://developer.android.com/media/camera/camerax/analyze
- CameraX configuration and resolution behavior: https://developer.android.com/media/camera/camerax/configuration
- Android foreground services overview: https://developer.android.com/develop/background-work/services/fgs
- Android foreground service types and camera restrictions: https://developer.android.com/develop/background-work/services/fgs/service-types
- TensorFlow Lite Android GPU delegate guidance: https://developers.google.com/edge/litert/android/gpu
- TensorFlow Lite delegates overview: https://developers.google.com/edge/litert/performance/delegates
