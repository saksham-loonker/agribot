# Agribot

Agribot is a crop monitoring and plant-health inference project for farmers. It includes a native
offline Android app for field scanning, TensorFlow Lite models for tomato crop disease detection,
training/export/benchmark scripts, field deployment automation, and a local dashboard web app.

## Key Paths

- `agribot_android_app/` — Native offline Android app (Kotlin + Jetpack Compose, Hilt DI).
  See `agribot_android_app/README_APK.md` for build and deployment instructions.
- `agribot_platform/` — Local dashboard web app (PWA) with 14-language i18n, served by a Python
  HTTP server (`run_platform.py`).
- `deployment/` — Raspberry Pi setup and field networking scripts.
- `44_farmer_one_touch.py` — Farmer-facing runtime flow: dashboard, live inference, archive helper,
  autostart setup.
- [`AGRIBOT_ACCURACY_AND_FARMER_GUARDRAILS.md`](AGRIBOT_ACCURACY_AND_FARMER_GUARDRAILS.md) — Accuracy,
  fallback, sensor/GPS, map, offline-backend, farmer-UX, governance, and field-readiness contract.
- `RUNTIME_TESTING.md` — Runtime export, validation, and benchmark order.
- `requirements-runtime-pi.txt` — Raspberry Pi CPU runtime stack pins.

## Clone, Build, and Run Locally

The Android app needs the two model assets stored with Git LFS. Install Git
LFS before cloning so a fresh checkout receives the model files rather than
small pointer files:

```bash
git lfs install
git clone https://github.com/saksham-loonker/agribot.git
cd agribot
git lfs pull
```

For Android builds, install JDK 17 and Android SDK platform 35/build-tools
35.0.0, then set `ANDROID_HOME` to the SDK directory. Build and reinstall the
debug APK on a connected device with:

```bash
python 45_build_android_apk.py --variant debug
adb install -r dist/agribot-field-app-debug.apk
```

The first Android build needs internet access to download the Gradle
distribution and Android/Maven build dependencies. After installation, the
native app performs camera processing and inference locally and does not
request Android's `INTERNET` permission. The detailed build, signing, and
device-verification steps are in
[`agribot_android_app/README_APK.md`](agribot_android_app/README_APK.md).

The dashboard runs on Python's standard library and serves local files and
local API routes. To run the Python regression suite, install its optional
image-processing dependencies in a virtual environment:

```bash
python -m venv .venv
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -p "test_*.py"
```

Pi runtime dependencies and setup commands are documented separately in
[`RUNTIME_TESTING.md`](RUNTIME_TESTING.md) and target Raspberry Pi 5/aarch64
with Python 3.13.

## Android App Features

The native Android app (`agribot_android_app/android/`) provides:

- **Walk & Scan mode** — Walk through your field; the app auto-detects plants using on-device
  TensorFlow Lite models and tracks each plant with bounding boxes.
- **Burst Scan mode** — Capture a burst of frames for front-row overview scanning.
- **Plant health detection** — Classifies plants as Healthy, Sick, or Uncertain using trained
  models for Early blight, Late blight, Leaf Miner, nutrient deficiencies, and Spotted Wilt Virus.
- **Treatment recommendations** — Shows recommended treatments (chemical and organic) for each
  detected disease.
- **Treatment notes** — Record treatment status and notes for each plant.
- **Field progress tracking** — Real-time progress bar showing how much of your field has been scanned.
- **Sun-readable dark mode** — High-contrast display for bright sunlight conditions.
- **Battery & thermal monitoring** — Real-time battery level and thermal status display.
- **Confidence threshold slider** — Adjust detection sensitivity to balance false positives vs. misses.
- **Plant search & filter** — Search plants by ID, health status, or treatment notes.
- **PDF export** — Export scan reports as PDF for sharing with agronomists.
- **CSV/JSONL/Bundle export** — Export raw data in multiple formats.
- **Onboarding flow** — First-time farmer-friendly onboarding with step-by-step guidance.
- **Offline operation** — No internet required for field scanning.

## Building the Android App

See `agribot_android_app/README_APK.md` for full build instructions.

Quick start:

```powershell
cd agribot_android_app\android
$env:ANDROID_HOME = "$env:LOCALAPPDATA\Android\Sdk"
.\gradlew.bat :app:assembleDebug --no-daemon --max-workers=1
```

## Local Dashboard

```powershell
python agribot_platform/run_platform.py
```

Then open `http://localhost:8080` in a browser.

The dashboard keeps field-layout writes same-origin/loopback-only by default so
offline local use continues to work. For a shared hotspot or farm network,
configure a long random bearer token before starting the server; the browser
will prompt once and keep it only for the current session:

```bash
export AGRIBOT_PLATFORM_TOKEN="use-a-long-random-local-token"
python agribot_platform/run_platform.py --host 0.0.0.0
```

Optional cross-origin API clients must use a comma-separated allow-list in
`AGRIBOT_PLATFORM_CORS_ORIGINS`. API responses are never service-worker cached;
offline dashboard fallback is explicitly labeled as phone cache rather than
live field data.

## Local Artifacts

The two TFLite files used by the native Android app are distributed through
Git LFS so clean clones can build a working offline inference app. Training
checkpoints, datasets, other exported models, archives, virtual environments,
build outputs, and benchmark runs remain local artifacts.
