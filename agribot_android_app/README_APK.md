# Agribot Android app (v2)

Offline tomato-leaf checking for farmers. Two modes:

- **Check a plant**: point the camera at one plant. The app finds leaves, classifies each one, and votes over several
  leaves and frames to give one answer per plant.
- **Walk rows**: walk along the rows. Each plant is counted once, gets its own result, and appears on a field map.

No internet permission, no cloud inference. All results, photos and exports stay on the phone until the user shares them.

## Models (in `app/src/main/assets/`, Git LFS)

| File | Model | Licence | Input |
|---|---|---|---|
| `models/leaf_detector.tflite` | YOLOv9-T leaf detector (LibreYOLO) | MIT | 1x320x320x3, letterbox, /255 |
| `models/disease_classifier_0.tflite` | DINOv2 ViT-S/14 classifier, 9 classes (8 conditions + Other) | Apache-2.0 | 1x224x224x3, ImageNet mean/std |
| `model_manifest.json` | Contract: shapes, labels, thresholds, calibration, SHA-256 | | |

The app reads every model constant from `model_manifest.json`; nothing about the model is hard-coded. The training,
export and evaluation scripts live in [`../model_training/v2`](../model_training/v2). Held-out results measured on the
shipped `.tflite` files are recorded in `model_manifest.json` under `heldout_results`.

After cloning, fetch the models: `git lfs install && git lfs pull`. Without them the app shows "models could not be
loaded" and the contract test skips the hash check.

## Build

Requirements: JDK 17, Android SDK 35 (platform, build-tools 35.0.0), `local.properties` with `sdk.dir=...`.

```bash
cd agribot_android_app/android
./gradlew :app:assembleDebug            # app/build/outputs/apk/debug/app-debug.apk
./gradlew :app:assembleRelease          # unsigned unless signing is configured (below)
```

Release signing reads `AGRIBOT_RELEASE_STORE_FILE`, `AGRIBOT_RELEASE_STORE_PASSWORD`, `AGRIBOT_RELEASE_KEY_ALIAS` and
`AGRIBOT_RELEASE_KEY_PASSWORD` from Gradle properties, the environment, or an ignored `signing.properties`.

## Tests

```bash
./gradlew :domain:test :ml:testDebugUnitTest :data:testDebugUnitTest :camera:testDebugUnitTest \
          :feature-scan:testDebugUnitTest :app:testDebugUnitTest :app:lintRelease
./gradlew :app:connectedDebugAndroidTest      # needs a device or emulator
python3 -m unittest discover -s tests -p "test_*.py"   # from the repository root
```

- `ml` golden parity tests: Kotlin preprocessing and decision maths against the Python reference
  (`model_training/v2/final/canon.py`).
- `AgribotModelGoldenTest` (on device): the shipped `.tflite` files reproduce the reference logits, labels and
  detections, and "Check a plant" on static images gives the reference verdict. Logcat tag `AgribotBench` reports
  latency.
- UI tests cover onboarding, both modes, input validation and switching to Hindi.

Emulators on Apple silicon advertise SVE2 they cannot execute; the app detects emulators and uses LiteRT's built-in
kernels there. Real phones use XNNPACK. Measure speed only on real phones.

Use [FIELD_PILOT_RUNBOOK.md](FIELD_PILOT_RUNBOOK.md) for the physical-phone pilot.

## Latency work and pending device checks (2026-10-02)

The original merged app at `1b0ca6dc5c20a0651548880123557ea83431349c` measured 592.2, 625.0 and
464.2 ms per `analyze()` on an RMX5101 / Android 16 / SM7750 phone (mean 560.47 ms, one classified
leaf). These are the original app's measurements, not results for the optimization changes below.
The preserved baseline checkout and reports are under `verification-20261002/agribot/`.

The current checkout reduces camera and preprocessing allocations: direct RGBA plane reads with
fixed rotation strides; reusable detector/classifier tensor storage; direct letterbox writes;
bounded per-thread resize-weight caching; cached direct-buffer views; and one physical inference
thread with serialized interpreter access. Every externally retained frame and inference output
still owns its storage. Models, preprocessing arithmetic, camera resolution, leaf limits, consensus
and thermal pacing are preserved.

Production defaults remain CPU/XNNPACK with the original 2–4-thread policy. Independent detector
and classifier CPU thread counts, exact-precision GPU delegates and NNAPI are available for the
developer benchmark; no accelerated backend has been selected for production. GPU creation,
invocation and cleanup require the same physical thread. Unsupported or inaccurate accelerator
candidates are logged as `REJECTED`, rather than counted as speed improvements.

Local validation: 34 ML JVM tests and 9 camera JVM tests passed. The phone was disconnected during
the subsequent tuning run at the user's request, so final on-device parity, speed, camera behavior
and sustained performance are **pending**. Do not infer a measured speedup or release readiness.

When the USB phone is available again, use the existing JDK/SDK and run from `android/` (PowerShell
requires the quotes around the dotted Gradle property):

```powershell
$env:JAVA_HOME = 'C:\Program Files\Eclipse Adoptium\jdk-17.0.16.8-hotspot'
$env:ANDROID_HOME = "$env:LOCALAPPDATA\Android\Sdk"
$env:ANDROID_SERIAL = '<serial from adb devices -l>'
.\gradlew.bat :app:connectedDebugAndroidTest '-Pandroid.testInstrumentationRunnerArguments.class=com.sakshyam.agribot.AgribotRuntimeBenchmarkTest,com.sakshyam.agribot.AgribotCameraBenchmarkTest' --no-daemon --max-workers=1
```

The runtime benchmark validates golden logits, labels and detector boxes for each option; timed
calls must retain the CPU reference's leaf count and crop labels. It reports ten warm samples with
mean/median/p95 and component averages under `AgribotTune`. `AgribotCameraBench` separately compares
the original and optimized 1280×960 RGBA conversion paths with identical output checks. Neither
benchmark measures live-camera FPS or field accuracy. Save the per-test `logcat-*.txt` files from
`app/build/outputs/androidTest-results/connected/` before another run overwrites them.

Then compare the preserved baseline and optimized APK with the unchanged `AgribotModelGoldenTest`
three times each under comparable charge, thermal and compilation conditions. Select thread/backend
settings only after the measured candidate passes parity, and rerun the full device suite and both
camera modes on the final APK. The full suite now includes the two developer benchmark tests, so
the expected count is 16. The original 14-test release checks can be isolated with:

```powershell
.\gradlew.bat :app:connectedDebugAndroidTest '-Pandroid.testInstrumentationRunnerArguments.notClass=com.sakshyam.agribot.AgribotRuntimeBenchmarkTest,com.sakshyam.agribot.AgribotCameraBenchmarkTest' --no-daemon --max-workers=1
```

Host build evidence is under `dist/latency-20261002/`; final phone results remain pending. Release
signing and human/Hindi/agronomy review are still separate requirements.
