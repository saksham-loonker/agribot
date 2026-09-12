# Agribot Android Field Pilot Runbook

This runbook is the field procedure for proving the native Android app is ready
for pilot use without a Raspberry Pi, local server, internet access, or cloud inference.
It is intentionally strict: emulator runs and `android_test_*` export bundles are
development smoke evidence only.

## September 7 bug-fix build

The settings header opens a dismissible, scrollable dialog. Scan mode is on the
home screen; optional GPS/motion details and advanced runtime controls are collapsed.
Sensor activity is no longer displayed as an accuracy percentage in recording.

Tracks require repeated label observations before showing an initial diagnosis.
Three uncertain classifications clear a stale diagnosis. Duplicate timestamps do
not add evidence; detector-only tracks are excluded from confirmed counts, and
position history is bounded to 120 observations per live track.

The crop pipeline rejects effectively achromatic input before disease inference.
This is a narrow quality guard, not a trained non-plant detector: colored lab parts
can still pass, while grayscale or desaturated plant photos may be rejected. Use
clear color close-ups and review uncertain results. No model weights or calibration
parameters were retrained for this build. The existing Healthy acceptance policy
is preserved; uncertainty is never relabeled Healthy to inflate accuracy.

The 90% field-accuracy target is unverified. Validation must include labeled healthy,
diseased, and non-plant inputs from the target camera, with false positives, false
negatives and abstentions counted separately. Existing multi-frame consensus is
not proof of independent camera angles or reliable tractor plant identities.
Walking calibration must not be used as tractor travel distance.

Current build evidence is saved under `dist/bugfix-verification.log`; the focused
pre-fix reproductions are in `dist/bugfix-red.log`. Device installation and physical
testing were removed from this task at the user's request.

## Required Artifacts

### September 8 farmer interface build

The home screen now keeps Start visible, provides two descriptive scan choices,
and links to recent saved scans even when camera tracks have expired. Row plans
are editable secondary information, not measured plant locations. Settings and
the first-run guide support scrolling and larger text; new farmer-facing copy
has English and Hindi resources.

Recording separates the camera image from controls, uses a side-by-side layout
in landscape, pauses before showing results, and confirms finishing a scan.
Manual entry is explicitly a new observation. Last saved predictions are labeled
as saved observations, not displayed as a diagnosis of the current camera view.

Results use mutually exclusive categories (looks healthy, check plant, check
again), exclude unobserved positions, and provide search, filtering, optional row
plans, and export/open/share actions for the selected saved run. Late-loading
results cannot overwrite a subsequently selected run.

Build log: `dist/farmer-ui-build.log`. The launch/navigation instrumentation is
compiled, but no physical-device UI or camera validation is claimed. The user
requested build only: do not install this artifact as part of this task.

- `dist/agribot-field-app-release.apk`
- `dist/android_release_readiness.with_app_export.json`
- A physical-phone benchmark JSON, normally
  `dist/embedded_model_benchmark_target_phone.json`
- A real field export ZIP, normally `dist/agribot_real_field_export.zip`

## Phone Checklist

Before leaving for the field:

- Use an Android 8+ arm64 phone with rear camera and at least 4 GB RAM.
- Disable battery saver for the pilot run.
- Charge the phone to at least 80 percent.
- Confirm at least 2 GB free storage.
- Install the signed release APK, not a debug-only APK.
- Put the phone in airplane mode or otherwise keep the run offline.
- Open Agribot once and grant only the camera permission.
- Confirm the diagnostics panel shows the model bundle as ready.
- Confirm the app has no `INTERNET` permission using the APK verifier or device
  app-info screen.

## Install And Benchmark

Install the release APK on the physical phone:

```powershell
$env:ANDROID_HOME="$env:LOCALAPPDATA\Android\Sdk"
& "$env:ANDROID_HOME\platform-tools\adb.exe" install -r dist\agribot-field-app-release.apk
```

Run the physical-phone benchmark:

```powershell
python 47_android_model_benchmark.py --output dist\embedded_model_benchmark_target_phone.json
```

The benchmark is acceptable only when the report identifies a non-emulator
device, advertises `arm64-v8a`, uses package `com.sakshyam.agribot`, and reports
model bundle `agribot-model-bundle-v001`.

## Sensor And GPS Acceptance

Run this check before interpreting any plant map or distance number:

1. Open the diagnostics panel and start a fresh scan with GPS reference enabled.
2. Keep the phone still for a few seconds. The first GPS fix may correctly show
   `0 m`; this is the reference anchor, not a failure.
3. Walk at least 10 to 20 metres in a straight line, then stop. Confirm that
   distance, speed, motion-event status, source, and quality update together.
4. If Activity Recognition is requested, grant it for the pilot. If it is
   denied, continue only if the diagnostics panel shows an acceleration-based
   estimate rather than an unexplained zero.
5. Tap the measurement reset action, wait for the next reference point, and
   repeat the walk. Resetting the walk must not erase confirmed plant history.
6. Repeat once with GPS disabled or unavailable. The app must either show an
   explicitly estimated phone-sensor path or say that distance is unavailable;
   it must not silently turn a missing value into `0 m`.
7. When GPS is available, verify the map card shows a GPS reference path with
   an anchor first and additional points only after meaningful movement. Confirm
   the provider/point count is visible in diagnostics and that the path is
   labelled as a reference path, not as exact plant positions.
8. When GPS is disabled, stale, inaccurate, or the phone has no validated
   direction, verify the map says `distance only`/motion-only and does not draw
   a synthetic 2D track. A plant may be counted with a confirmed camera track,
   but it must not receive a fabricated field coordinate.
9. Stop or pause the run, export the bundle, and confirm the run event log
   contains a bounded `gps_path_snapshot` with local east/north coordinates and
   no raw latitude/longitude fields.

Record the phone model, Android version, GPS accuracy, distance source, and any
zero/stale interval in the pilot report. GPS is a path/envelope reference; it is
not evidence of an individual plant's physical width or height. Until camera
calibration is accepted, plant size must remain relative or unknown.

### Optional walking calibration

Do this before the first scan when phone-only distance matters:

1. Mark a straight 5–20 metre distance that can be measured reliably.
2. Open Settings → Walking distance calibration and enter the marked distance.
3. Tap Start walking calibration, walk the complete line at the same pace used
   for scanning, and tap Finish calibration only after reaching the end.
4. Accept the result only if the app reports a calibrated stride. A short walk,
   missing sensor events, an unrealistic stride, or backgrounding must leave the
   previous profile unchanged and explain how to repeat the check.
5. Confirm the diagnostics panel shows the saved stride and `CALIBRATED` status.
   The value is an operator/phone estimate, not a substitute for GPS field
   reference or camera/reference-object calibration of plant size.

## Side Scan Pilot

Use Side Scan first. This is the primary Pi replacement workflow.

1. Open Agribot.
2. Select Side Scan.
3. Confirm the active field, active row, plant count, and cooldown.
4. Tap Start Side Scan.
5. Walk or hold the phone beside the row at roughly 20 to 60 cm from leaf area.
6. Wait for each emitted decision before moving to the next plant.
7. Use Retake when the operator knows the latest plant was poorly framed.
8. Use Skip only for plants that cannot be captured.
9. Use Snapshot when a human-visible condition needs evidence even before an
   automatic sick or uncertain decision.
10. Stop after a full row or full planned field segment.

Acceptance evidence:

- At least one real row segment produces multiple mapped decisions.
- Sick, uncertain, manual, retake, skip, and snapshot paths are exercised when
  the field conditions allow them.
- Exported `decisions.jsonl` keeps Pi-compatible keys.
- Evidence JPEGs are present for sick, uncertain, and manual snapshot cases
  unless the run legitimately hit the storage cap.

## Front Row Overview Pilot

Run Front Row Overview after Side Scan is proven on the same phone.

1. Select Front Overview.
2. Choose left and right row IDs.
3. Set nearest left and right plant numbers.
4. Hold or place the phone about 1 m before the corridor.
5. Align both rows inside the guide rails.
6. Tap Start Front Overview, then Capture Burst.
7. Review all burst decisions before commit.
8. Correct wrong row-side, plant, or label assignments manually.
9. Mark ambiguous geometry as uncertain instead of guessing.
10. Confirm review and export the run.

Acceptance evidence:

- The export contains row-side and front-geometry metadata.
- Ambiguous captures are marked uncertain.
- Manual review/correction events are present when corrections were made.
- The field map can be reviewed from run history.

## Export And Readiness

Create a real field export from the app after the run, then save or share it as:

```text
dist/agribot_real_field_export.zip
```

Run the release readiness gate with the physical-phone benchmark and real field
export:

```powershell
python 48_android_release_readiness.py --benchmark-report dist\embedded_model_benchmark_target_phone.json --field-export-bundle dist\agribot_real_field_export.zip --require-field-ready --output dist\android_release_readiness.field.json
```

The app is field-ready only when this command exits successfully and reports:

```json
{
  "development_ready": true,
  "field_ready": true
}
```

## Troubleshooting

- Camera permission denied: grant camera permission in Android app settings,
  reopen Agribot, and verify the diagnostics panel.
- Model not ready: rebuild or reinstall the APK and rerun the readiness script;
  do not field-pilot with a missing or mismatched model manifest.
- Phone slow or hot: pause, let the phone cool, and resume only when diagnostics
  no longer show thermal warnings. Severe or critical thermal status should
  pause scanning after the current decision group.
- Storage cap reached: continue decisions, but expect `evidence_status` to show
  `storage_cap_reached` until space is freed.
- Export missing evidence: verify the run did not hit the evidence cap and that
  sick, uncertain, or manual snapshot cases occurred.
- Front Overview geometry ambiguous: retake with both rows inside guide rails or
  mark the segment uncertain.

## What Not To Claim

- Do not claim field readiness from an emulator benchmark.
- Do not claim field readiness from `android_test_*` exports.
- Do not claim Front Overview agronomic accuracy from synthetic or scaffold
  assets alone.
- Do not use a Pi dashboard, local HTTP server, internet access, or cloud model
  as part of normal field operation.
