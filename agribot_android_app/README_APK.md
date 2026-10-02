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
