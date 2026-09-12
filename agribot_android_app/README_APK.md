# Agribot Native Android APK

This folder contains the native offline Agribot Android app. It replaces the
old Capacitor/Pi HTTP wrapper: camera capture, local TensorFlow Lite inference,
field mapping, run storage, and CSV/JSONL export run on the phone.

The production native app must not request `INTERNET`. Normal field operation
does not require a Raspberry Pi, server, hotspot, or dashboard.

Read [`../AGRIBOT_ACCURACY_AND_FARMER_GUARDRAILS.md`](../AGRIBOT_ACCURACY_AND_FARMER_GUARDRAILS.md)
for the accuracy contract, fallback hierarchy, GPS/distance limits, farmer UX,
offline-backend boundary, and field-readiness gates.

Tracked Capacitor config, package lock, and generated Android shim files have
been removed. `package.json` only exposes native Gradle convenience commands;
do not run `cap sync` for this replacement app.

## Build Requirements

- JDK 17 or newer
- Android SDK platform/build-tools 35
- Android SDK path in `ANDROID_HOME`
- Gradle wrapper under `agribot_android_app/android`

## Model Assets

The runnable classifier and detector assets are stored in Git LFS. Install Git
LFS and run `git lfs pull` if either file is missing or is only a small text
pointer after cloning. Before building an APK, verify the app assets and their
manifest hashes:

```powershell
python 46_prepare_android_model_bundle.py --app-assets-only --json
```

Expected readiness for the current bundle:

- classifier: `models/classifier_fastcrop_float32.tflite`
- detector: `models/detector_nano_256_raw_float32.tflite` with Kotlin-side YOLO NMS
- labels: `0 Early_blight`, `1 Healthy`, `2 Late_blight`, `3 Leaf Miner`,
  `4 Magnesium Deficiency`, `5 Nitrogen Deficiency`,
  `6 Pottassium Deficiency`, `7 Spotted Wilt Virus`

Those two runtime `.tflite` assets are tracked with Git LFS. Training
checkpoints, datasets, and alternate/quantized exports remain ignored; they are
needed only for retraining or rebuilding the wider model-export bundle.

## Build Debug APK

From the repo root:

```powershell
python 45_build_android_apk.py
```

Or directly:

```powershell
cd agribot_android_app\android
$env:ANDROID_HOME = "$env:LOCALAPPDATA\Android\Sdk"
.\gradlew.bat :app:assembleDebug --no-daemon --max-workers=1
```

APK output:

```text
agribot_android_app/android/app/build/outputs/apk/debug/app-debug.apk
dist/agribot-field-app-debug.apk
```

## Build Release APK

Unsigned release verification build:

```powershell
python 45_build_android_apk.py --variant release
```

Signed local pilot release, without committing secrets:

```powershell
$env:AGRIBOT_RELEASE_STORE_FILE = "C:\path\to\agribot-release.jks"
$env:AGRIBOT_RELEASE_STORE_PASSWORD = "<store password>"
$env:AGRIBOT_RELEASE_KEY_ALIAS = "<key alias>"
$env:AGRIBOT_RELEASE_KEY_PASSWORD = "<key password>"
python 45_build_android_apk.py --variant release --require-signed
```

The same values may be supplied as Gradle properties in a local, uncommitted
`~/.gradle/gradle.properties` file.

`45_build_android_apk.py` verifies copied APK metadata with `aapt`: package
`com.sakshyam.agribot`, minSdk `26`, target SDK `35`, `arm64-v8a`, `CAMERA`,
and no `INTERNET`. It also rejects packaged legacy Capacitor/Pi dashboard
assets such as `assets/public/` and `assets/capacitor.config.json`. Debug APKs
must verify with `apksigner` using APK Signature Scheme v2 or newer. Release
APKs verify only when signing credentials are supplied; use `--require-signed`
for any field-distribution build.

Release APK output:

```text
agribot_android_app/android/app/build/outputs/apk/release/app-release.apk
dist/agribot-field-app-release.apk
```

## Verification Commands

```powershell
python -m unittest discover -s tests -p "test_*.py"
cd agribot_android_app\android
$env:ANDROID_HOME = "$env:LOCALAPPDATA\Android\Sdk"
.\gradlew.bat :domain:test :camera:testDebugUnitTest :ml:testDebugUnitTest :data:testDebugUnitTest :feature-scan:testDebugUnitTest :app:testDebugUnitTest :app:assembleDebug :app:lintDebug --stacktrace --no-daemon --max-workers=1
.\gradlew.bat connectedDebugAndroidTest --stacktrace --no-daemon --max-workers=1
```

Check APK metadata:

```powershell
& "$env:ANDROID_HOME\build-tools\35.0.0\aapt.exe" dump badging app\build\outputs\apk\debug\app-debug.apk
```

The native APK should show package `com.sakshyam.agribot`, `sdkVersion:'26'`,
`targetSdkVersion:'35'`, `native-code: 'arm64-v8a'`, and no
`android.permission.INTERNET`.

Generate the consolidated local readiness report after building the dist APKs:

```powershell
python 48_android_release_readiness.py --require-development-ready
```

This writes `dist/android_release_readiness.json`. `development_ready` covers
the offline model manifest, Pi-compatible label order, debug APK metadata, and
debug signing. `field_ready` is stricter and remains false until the release APK
is signed and a physical-phone benchmark meets the target latency gates.

Use [FIELD_PILOT_RUNBOOK.md](FIELD_PILOT_RUNBOOK.md) for the physical-phone
benchmark, Side Scan field run, Front Overview review run, real export
collection, and troubleshooting procedure needed to make `field_ready` true.

## Model Benchmark Report

Run the embedded Android model benchmark and pull a JSON report:

```powershell
python 47_android_model_benchmark.py
```

The report is written to:

```text
dist/embedded_model_benchmark.json
```

Emulator results are smoke evidence only. Before field pilot, run the same
command on the named `arm64-v8a` target phone and keep the report with the APK.
The Side Scan classifier target is average latency <= `160 ms` and p95 <=
`200 ms`; the Front Overview detector target is p95 <= `300 ms` on the selected
phone.

Field-pilot benchmark gate for a connected physical phone:

```powershell
python 47_android_model_benchmark.py --device <adb-serial> --require-physical --expected-model "<phone model>" --enforce-target-limits --output dist/embedded_model_benchmark_target_phone.json
python 49_validate_android_field_export.py dist/agribot_real_field_export.zip --require-real-field --output dist/android_field_export_validation.json
python 48_android_release_readiness.py --benchmark-report dist/embedded_model_benchmark_target_phone.json --field-export-bundle dist/agribot_real_field_export.zip --require-field-ready
```

Use `adb devices` to get `<adb-serial>`. The command fails if the report is
from an emulator, the physical device does not advertise `arm64-v8a`, the
device model does not match, or any target latency limit is missed. The field
export validator fails if the real run bundle is missing required JSON/CSV
files, row geometry, completed-run metadata, or JPEG evidence for sick,
uncertain, or manual decisions.

If Gradle reports `Instrumentation run failed due to failed to attach`, install
the debug APKs and run the focused export test directly:

```powershell
$env:ANDROID_HOME = "$env:LOCALAPPDATA\Android\Sdk"
$adb = "$env:ANDROID_HOME\platform-tools\adb.exe"
& $adb install -r agribot_android_app\android\app\build\outputs\apk\debug\app-debug.apk
& $adb install -r agribot_android_app\android\app\build\outputs\apk\androidTest\debug\app-debug-androidTest.apk
& $adb shell am instrument -w -r -e class com.sakshyam.agribot.AgribotRoomExportIntegrationTest com.sakshyam.agribot.test/androidx.test.runner.AndroidJUnitRunner
```

To pull the latest app-private export ZIP from a debug build and validate it:

```powershell
$zipPath = (& $adb shell run-as com.sakshyam.agribot find files/exports -maxdepth 1 -name '*.zip' | Select-Object -Last 1).Trim()
cmd /c """$adb"" exec-out run-as com.sakshyam.agribot cat $zipPath > dist\android_app_export_instrumented.zip"
python 49_validate_android_field_export.py dist\android_app_export_instrumented.zip --output dist\android_app_export_instrumented.validation.json
python 48_android_release_readiness.py --require-development-ready --field-export-bundle dist\android_app_export_instrumented.zip --output dist\android_release_readiness.with_app_export.json
```

Interrupted Side Scan runs are recovered on app restart. If the latest Room run
was still `RECORDING`, the app marks it `PAUSED`, restores the active run and
next plant index, and records a `recovered_interrupted_run` event before the
operator resumes or exports.

## Current Readiness Gates

Development APK readiness requires:

- `python 46_prepare_android_model_bundle.py --app-assets-only --json` reports `"ready": true`.
- Unit/lint/debug build verification passes.
- `connectedDebugAndroidTest` passes on the emulator.
- Debug APK metadata shows package `com.sakshyam.agribot`, minSdk `26`, target
  SDK `35`, `arm64-v8a`, `CAMERA`, and no `INTERNET`; debug signing verifies
  with APK Signature Scheme v2 or newer.
- `python 47_android_model_benchmark.py` writes
  `dist/embedded_model_benchmark.json`.
- `python 48_android_release_readiness.py --require-development-ready` writes
  `dist/android_release_readiness.json` with development and field blockers.

Field-pilot readiness additionally requires:

- A signed release APK built with `python 45_build_android_apk.py --variant release --require-signed`.
- A benchmark report from the named real `arm64-v8a` target phone. Emulator
  latency is only a smoke test and must not be used for field FPS claims.
- A real app export ZIP from captured field crops passes
  `python 49_validate_android_field_export.py dist/agribot_real_field_export.zip --require-real-field`.
- `python 48_android_release_readiness.py --benchmark-report dist/embedded_model_benchmark_target_phone.json --field-export-bundle dist/agribot_real_field_export.zip --require-field-ready` passes.
- A field review using real captured crops, because synthetic and emulator
  tests cannot prove agronomic accuracy or sustained thermal behavior.
