# Field pilot runbook (physical phones)

Use this before any release to farmers. Do not claim field readiness from an emulator benchmark.
Do not claim field readiness from `android_test_*` exports or from the golden images.

The app runs fully offline, without a Raspberry Pi, internet access or cloud inference. The APK has no `INTERNET`
permission. At install, grant only the camera permission; location and step counting are optional and only used by
Walk rows.

## 1. Device check (once per phone model)

1. Install the release APK and open it once.
2. Run the on-device golden test against the phone:
   `./gradlew :app:connectedDebugAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=com.sakshyam.agribot.AgribotModelGoldenTest`
3. Record the `AgribotBench` logcat line (`adb logcat -s AgribotBench`). Target: `analyze()` under 250 ms on an
   8 GB phone, so "Check a plant" decides within about 3 seconds.
4. All golden tests must pass. A failure means the phone's runtime does not reproduce the validated model; stop.

## 2. Check a plant pilot

1. Choose at least 30 plants whose condition an agronomist has confirmed, covering every condition present.
2. For each plant: open **Check a plant**, hold the phone 20–40 cm from the leaves in daylight, wait for the result,
   tap **Save result**.
3. Note the agronomist's diagnosis next to each saved result.
4. Count results as correct, "Not sure — look again" or wrong. "Not sure" is acceptable; a confident wrong disease is
   the error that matters.
5. Repeat in morning, midday and evening light, and in Hindi.

## 3. Walk rows pilot

1. Measure the step length (walk 10 steps, measure, divide by 10) and set it in **Settings**.
2. Pick one row with a known plant count. Start **Walk rows**, enter rows, plants per row and spacing.
3. Walk slowly with the phone pointed at the plants. Check that the plant number on screen follows the real plant.
   Use **Next plant** / **Previous** to correct any drift, and note how often a correction was needed.
4. Tap **Finish**. On the results screen, check that every plant appears once on the field map and that the
   "need attention" list matches what the agronomist sees.
5. Tap **Share all data and photos (ZIP)** and keep the file as `dist\agribot_real_field_export.zip`.

## 4. Export validation

```bash
python 49_validate_android_field_export.py dist\agribot_real_field_export.zip --require-real-field
```

The validator checks the export structure and that every decision comes from the bundled model (bundle ID from
`model_manifest.json`).

## 5. Sign-off

Release needs a named human reviewer's sign-off, covering:

- Field accuracy from section 2 (per condition, with the number of plants).
- Device timing from section 1 for each phone model.
- Hindi text reviewed by a native speaker; treatment guidance reviewed by an agronomist.
- Data licences (Tomato-Village, FieldPlant, PlantDoc, Mendeley tomato leaves) cleared for the intended use.
