# AGRIOT ANDROID APP — COMPLETE REDESIGN PROMPT

## GOAL
Completely overhaul the Agribot Android app to fix ALL UI, plant detection, tracking, and sensor integration issues. The app must be farmer-friendly, intuitive, and work correctly without bugs.

---

## 1. PLANT DETECTION & TRACKING (CRITICAL)

### 1.1 Bounding Boxes
- Every detected plant MUST have a visible bounding box with:
  - **8f stroke width** (thick, visible on camera preview)
  - **20% opacity fill** matching the health color
  - **Unique plant ID** label (e.g., "plant_1", "plant_2") at top-left of box
  - **Confidence score** (e.g., "85%") below the plant ID
  - **Occlusion indicator** ("OCCLUDED") for plants not seen in current frame
- Bounding box coordinates MUST be scaled from analysis frame pixels to screen pixels:
  - `scaleX = canvasWidth / frameWidth`
  - `scaleY = canvasHeight / frameHeight`
  - Apply scale to ALL box coordinates (left, top, right, bottom)

### 1.2 Plant Tracking Algorithm
- Use **IoU (Intersection over Union)** for spatial matching (weight: 0.4)
- Use **cosine similarity** for feature-based re-identification (weight: 0.3)
- Use **spatial distance** for proximity matching (weight: 0.3)
- Match threshold: **0.35f** (minimum score to match)
- Feature vector (7 dimensions): [centerX, centerY, width, height, confidence, aspectRatio, areaRatio]
- **Occlusion handling**: Plants not seen in a frame are marked occluded (not immediately removed)
- **Stale removal**: Plants not seen for 5+ seconds are removed
- **Unique IDs**: Format `"plant_N"` where N starts at 1 and increments

### 1.3 Auto-Detection with Manual Override
- **Automatic**: When a run is active, every frame is processed:
  - Object detection runs on each frame
  - Plants are tracked across frames with unique IDs
  - Closest plant to camera center is auto-recorded as a decision
  - Cooldown between auto-records (configurable, default 2 seconds)
- **Manual override**: Farmer can manually mark any plant as Healthy/Sick/Uncertain
- **Stop recording**: Auto-detection stops immediately when run is stopped/paused

---

## 2. SENSOR INTEGRATION (CRITICAL)

### 2.1 Accelerometer (tilt/roll/pitch)
- Use `Sensor.TYPE_ACCELEROMETER` with low-pass filter (alpha = 0.8)
- Calculate orientation using `SensorManager.getRotationMatrix` + `getOrientation`
- **Tilt**: `sqrt(orientation[1]² + orientation[2]²)`
- **Roll**: `orientation[2]` (Z-axis rotation)
- **Pitch**: `orientation[1]` (X-axis rotation)
- Update sensor data every 200ms (5 times per second)
- Use sensor data for 3D position mapping: `map2DTo3D()` converts 2D bbox to 3D coordinates

### 2.2 GPS Speed (speedometer)
- Use `LocationManager` with `GPS_PROVIDER` (fallback to `NETWORK_PROVIDER`)
- Track speed via `Location.hasSpeed()` and `Location.speed`
- Update speed every 500ms
- Display speed in m/s on the sensor overlay
- Handle `SecurityException` gracefully (speed = 0 if permission denied)

### 2.3 Sensor Data Display
- Show in top-right overlay with dark semi-transparent background
- Display: Tilt (°), Roll (°), Pitch (°), Speed (m/s)
- Show accuracy percentage based on sensor stability
- Accuracy color: Green (>80%), Yellow (>50%), Red (<50%)

---

## 3. UI/UX DESIGN (CRITICAL)

### 3.1 Screen Structure
**Two distinct modes:**
1. **Recording Mode**: Live camera with bounding boxes, sensor data, plant count, controls
2. **Data Viewing Mode**: Plant grid, filters, detailed plant info, export

**Navigation:**
- Home screen → "Start Scan" button → Recording screen
- Recording screen → "View Data" button → Data viewing screen
- Data viewing screen → "Back to Recording" button → Recording screen
- Any screen → "Back to Home" button → Home screen

### 3.2 Recording Screen Layout (EnhancedRecordingScreen)
```
┌─────────────────────────────────────┐
│ 📱 Sensors (top-right overlay)      │
│   Tilt: 5.2°  Roll: 1.1°  Pitch: 0.3°│
│   Speed: 0.5 m/s  Accuracy: 85%     │
│                                     │
│ 🌱 Plants: 12 (top-left overlay)     │
│                                     │
│   [CAMERA PREVIEW FILLS SCREEN]     │
│   ┌─────────────┐                   │
│   │  plant_1    │                   │
│   │  85%        │  ← Bounding box   │
│   │  ┌───────┐  │     with label    │
│   │  │       │  │                   │
│   │  └───────┘  │                   │
│   └─────────────┘                   │
│                                     │
│                                     │
│                                     │
│                                     │
│                                     │
│                                     │
│                                     │
│ ┌─────────────────────────────────┐ │
│ │ [Back to Home] [View Data]      │ │
│ │                                 │ │
│ │ [Capture Burst] [Stop]          │ │  ← Mode-specific controls
│ │                                 │ │
│ │ [Healthy] [Sick] [?]            │ │  ← Manual override (side scan)
│ │                                 │ │
│ │ [Export Data] [Clear Tracking]  │ │
│ └─────────────────────────────────┘ │
└─────────────────────────────────────┘
```

**Key layout rules:**
- Camera preview fills entire screen (`fillMaxSize`)
- Bounding boxes drawn on top of camera (scaled to screen coordinates)
- Sensor overlay: top-right, 12dp padding, 12dp rounded corners, dark semi-transparent background
- Plant count overlay: top-left, 12dp padding, 12dp rounded corners, dark semi-transparent background
- Bottom controls: Column at bottom with 16dp padding, 12dp spacing between rows
- Bottom controls have **bottom padding** to avoid overlapping with bounding box labels
- All buttons: 48dp height minimum, 14sp text, proper spacing

### 3.3 Home Screen Layout
```
┌─────────────────────────────────────┐
│ Agribot                             │  ← Header
│                                     │
│ ┌─────────────────────────────────┐ │
│ │ Recording Mode                  │ │  ← Card: Side Scan / Front Overview
│ │ [● Side Scan] [○ Front Overview]│ │
│ └─────────────────────────────────┘ │
│                                     │
│ ┌─────────────────────────────────┐ │
│ │ System Status                   │ │  ← Card: Field, Model, Camera, etc.
│ │ Field: My Farm                  │ │
│ │ Model: Ready                    │ │
│ │ Camera: Ready                   │ │
│ └─────────────────────────────────┘ │
│                                     │
│ ┌─────────────────────────────────┐ │
│ │ Field Layout                    │ │  ← Card: Row selection, plant count
│ └─────────────────────────────────┘ │
│                                     │
│ ┌─────────────────────────────────┐ │
│ │ Run History                     │ │  ← Card: Recent runs
│ └─────────────────────────────────┘ │
│                                     │
│ ┌─────────────────────────────────┐ │
│ │ Diagnostics                     │ │  ← Card: App version, model, etc.
│ └─────────────────────────────────┘ │
│                                     │
│ ┌─────────────────────────────────┐ │
│ │ [ START SIDE SCAN ]             │ │  ← Big button at bottom
│ └─────────────────────────────────┘ │
│                                     │
│ [View Plant Data] [Export CSV]      │  ← Secondary actions
│                                     │
│ *AI-generated, may make mistakes    │  ← Footer
└─────────────────────────────────────┘
```

**Key layout rules:**
- All sections wrapped in Cards with 16dp rounded corners
- Main "Start" button at bottom: 64dp height, full width, prominent
- Secondary actions below start button in a Row
- Footer text at very bottom: "*AI-generated, may make mistakes"
- Scrollable content above the fixed bottom buttons

### 3.4 Data Viewing Screen Layout
```
┌─────────────────────────────────────┐
│ [Healthy] [Sick] [Uncertain] [All]  │  ← Filter chips
│                                     │
│ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐     │
│ │P1   │ │P2   │ │P3   │ │P4   │     │  ← Plant grid (full height)
│ │85%  │ │Sick │ │?    │ │85%  │     │
│ └─────┘ └─────┘ └─────┘ └─────┘     │
│                                     │
│ ┌─────────────────────────────────┐ │
│ │ Plant Details                   │ │  ← Detail panel (sticky bottom)
│ │ ID: plant_1                     │ │
│ │ Health: Healthy                 │ │
│ │ Confidence: 85%                 │ │
│ │ Last seen: 2s ago               │ │
│ │ Position: (1.2m, 0.5m, 1.5m)    │ │
│ └─────────────────────────────────┘ │
│                                     │
│ *AI-generated, may make mistakes    │  ← Footer
└─────────────────────────────────────┘
```

**Key layout rules:**
- Filter chips at top: ALL, HEALTHY, SICK, UNCERTAIN, OCCLUDED
- Plant grid fills available height (not fixed 400dp)
- Each plant card: 16sp plant ID, 14sp health status, colored border
- Selected plant: highlighted with cyan border (30% opacity)
- Detail panel: sticky at bottom, 16dp rounded corners
- Footer text at bottom

---

## 4. CODE STRUCTURE (CRITICAL)

### 4.1 SideScanViewModel.kt
- `onAnalysisFrame()`: **FIRST LINE** must be `if (!_state.value.isRecording) return`
- SIDE_SCAN mode: use `detectFrontFrame()` + `plantTracker.processFrameDetections()` for bounding boxes
- FRONT_ROW_OVERVIEW mode: use `detectFrontFrame()` + `plantTracker.processFrameDetections()`
- Both modes: auto-record closest plant, then arm cooldown
- `updateSensorData()`: include `speedMps` from `plantTracker.currentSpeedMps`

### 4.2 PlantTracker.kt
- Implement `SensorEventListener` for accelerometer
- Implement `LocationListener` for GPS speed
- `startTracking()`: register accelerometer + location listeners
- `stopTracking()`: unregister all listeners
- `processFrameDetections()`: match using IoU + cosine similarity + spatial distance
- `map2DTo3D()`: use tilt/roll/pitch for 3D position mapping
- `clearAll()`: clear plants, reset ID counter

### 4.3 EnhancedRecordingScreen.kt
- Accept `frameWidth` and `frameHeight` for coordinate scaling
- `PlantBoundingBoxOverlay`: scale all coordinates from frame to screen
- Bottom controls with padding to avoid overlapping with box labels
- Mode-specific controls (Front: Capture Burst/Stop; Side: Mark/Stop/Health/Sick/?)
- Sensor overlay: top-right with speed display
- Plant count overlay: top-left

### 4.4 AgribotScreens.kt
- `RecordingScreen`: use `EnhancedRecordingScreen` for BOTH modes
- `HomeScreen`: cards with 16dp rounded corners, big start button at bottom
- `DataAnalysisScreen`: full-height grid, filter chips, sticky detail panel
- Footer: "*AI-generated, may make mistakes" on all screens

### 4.5 ScanUiState.kt
- Add `speedMps: Float` to `SensorData`
- Keep `analysisWidth` and `analysisHeight` as `Int?`
- Keep `trackedPlants`, `plantFilter`, `selectedPlantId`, `isDataMode`

---

## 5. CONSTRAINTS & FORBIDDEN PATTERNS

### 5.1 Compose BOM 2024.12.01 / compiler 1.5.15
- **DO NOT** use `androidx.compose.material3.icons.*` (NOT available)
- **DO NOT** use `BorderStroke` (NOT available) — use `Modifier.background()` with shape
- **DO NOT** use `DrawScope.nativeCanvas` (NOT available) — use `Box` + `Text` + `IntOffset`
- `IntOffset` requires `Int` — use `.roundToInt()` before `.coerceAtLeast(0)`

### 5.2 Kotlin 1.9.25, JVM target 17
- Use `kotlin.math.abs` not `Math.abs`
- Use `kotlin.math.max` not `Math.max`
- Use `kotlin.math.cos`, `kotlin.math.sin`, `kotlin.math.sqrt`

### 5.3 Android permissions
- Camera permission: `Manifest.permission.CAMERA`
- Location permission: `android.permission.ACCESS_FINE_LOCATION` (for speed)
- Handle permission denials gracefully

### 5.4 Forbidden patterns
- NO processing frames when `isRecording == false`
- NO using `classifySideFrame` for SIDE_SCAN mode (use `detectFrontFrame` instead)
- NO hardcoded coordinate values without scaling
- NO overlapping UI elements
- NO fixed-height grids (use `fillMaxHeight`)
- NO `BorderStroke` or Material 3 icons

---

## 6. VERIFICATION CHECKLIST

Before declaring complete, verify:
- [ ] `:feature-scan:compileDebugKotlin` succeeds with NO errors
- [ ] `:feature-scan:testDebugUnitTest` passes ALL tests (49+)
- [ ] `:app:assembleDebug` builds APK successfully
- [ ] APK installs on device (or download URL works)
- [ ] Bounding boxes are visible and correctly positioned
- [ ] Plant IDs are unique and displayed
- [ ] Confidence scores are shown
- [ ] Sensor data (tilt, roll, pitch, speed) is displayed
- [ ] No frames processed when not recording
- [ ] Auto-detection works when recording starts
- [ ] Manual override buttons work
- [ ] UI elements don't overlap
- [ ] Navigation between screens works
- [ ] Footer "*AI-generated, may make mistakes" on all screens
- [ ] No compilation warnings about unused variables
