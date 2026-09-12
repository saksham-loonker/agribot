# Runtime Testing Order

## 0. Raspberry Pi runtime environment

For a fresh Pi or a rebuilt home directory, create a dedicated virtual
environment before running any of the runtime scripts.

```bash
sudo apt update
sudo apt install -y python3-venv
python3 -m venv ~/agribot_venv
source ~/agribot_venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-runtime-pi.txt
```

This installs the packages needed for the runtime scripts, benchmarks, and the
OpenVino Retry flow. TFLite export may still install additional extras when you
explicitly use the TFLite path.

If the install is interrupted, rerun the same command and wait for a final
`Successfully installed ...` line before testing imports. On Raspberry Pi, the
largest download is usually the CPU-only PyTorch wheel. The requirements file
pins the correct `aarch64` CPU wheels directly so pip does not try to install
CUDA packages.

If a previous install failed with `OSError: [Errno 28] No space left on device`,
clear the broken environment and pip cache first:

```bash
rm -rf ~/agribot_venv ~/.cache/pip
python3 -m venv ~/agribot_venv
source ~/agribot_venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir -r requirements-runtime-pi.txt
```

## 1. Export Pi deployment runtime variants

```powershell
python 10_export_runtime_models.py --runtimes ncnn openvino
```

Artifacts are written to `runtime_exports/`.
Export status is also written to `dataset/runtime_tracking/export_results.csv`.

## 2. Accuracy tests

Run these in order to compare the exported CPU runtimes against the current
PyTorch model:

```powershell
python 11_test_runtime_pt.py
python 13_test_runtime_ncnn.py
python 14_test_runtime_openvino.py
```

Default test set: `clf_dataset/test`

Reports are written to:

- `runtime_reports/accuracy/pt/`
- `runtime_reports/accuracy/ncnn/`
- `runtime_reports/accuracy/openvino/`

Each non-PT runtime report includes PT agreement numbers so conversion drift is easy to spot.
Accuracy summary rows are written to `dataset/runtime_tracking/accuracy_results.csv`.

## 3. Pipeline benchmarks

Classifier-only benchmark:

```powershell
python 21_benchmark_runtime_pt.py
python 23_benchmark_runtime_ncnn.py
python 24_benchmark_runtime_openvino.py
```

Full detect + classify benchmark:

```powershell
python 21_benchmark_runtime_pt.py --with-detector
python 23_benchmark_runtime_ncnn.py --with-detector
python 24_benchmark_runtime_openvino.py --with-detector
```

The benchmark reports include average FPS, average latency, max selected-frame
latency, frames above 180 ms, frames above the hard 200 ms ceiling, and pass/fail
flags for the Raspberry Pi deployment target.

Benchmark reports are written to:

- `runtime_reports/benchmark/pt/`
- `runtime_reports/benchmark/ncnn/`
- `runtime_reports/benchmark/openvino/`

Benchmark summary rows are written to `dataset/runtime_tracking/benchmark_results.csv`.

## 4. Run everything automatically

```powershell
python 30_run_runtime_suite.py
```

This runs export, accuracy tests, classifier-only benchmarks, full-pipeline benchmarks, and final summarization.

Final CSV outputs:

- `dataset/runtime_tracking/export_results.csv`
- `dataset/runtime_tracking/accuracy_results.csv`
- `dataset/runtime_tracking/benchmark_results.csv`
- `dataset/runtime_tracking/runtime_leaderboard.csv`
- `dataset/runtime_tracking/best_runtime_summary.csv`
- `dataset/runtime_tracking/deployment_recommendation.json`
- `dataset/runtime_tracking/suite_runs.csv`

The deployment recommendation only considers `pt`, `ncnn`, and `openvino`.
It selects the fastest runtime that preserves PyTorch accuracy within 0.5
percentage points, reaches at least 96.0% frame-level accuracy, prefers the
strict 97.2% frame-level target when available, stays at or above 5 FPS, and
keeps selected-frame latency at or below 200 ms.

## 5. Live 5 FPS plant decision loop

After choosing the runtime, run the loop with the selected exported classifier
and detector paths from `deployment_recommendation.json`.

```powershell
python pi_plant_loop.py --camera 0 --with-detector \
  --clf-model runtime_exports/classifier_openvino_model \
  --det-model runtime_exports/detector_openvino_model
```

Use the NCNN paths instead if NCNN is selected:

```powershell
python pi_plant_loop.py --camera 0 --with-detector \
  --clf-model runtime_exports/classifier_ncnn_model \
  --det-model runtime_exports/detector_ncnn_model
```

The loop samples selected frames at 5 FPS. It emits after two high-confidence
matching primary frames. If those two frames disagree or either is low
confidence, it uses the third backup frame. If the three-frame window is still
ambiguous, it emits `Uncertain`.

Current calibrated fast deployment settings for the Pi 5 CPU path:

- classifier: `runtime_exports/classifier_openvino_model`
- detector: disabled for the live loop
- crop: `--crop-mode mask --crop-scale 0.55 --crop-pad 0.05`
- confidence gate: `--conf 0.765 --high-conf 0.765`

On the full `clf_dataset/test` split this produced `95.66%` raw frame accuracy
at `12.53 FPS` with `80 ms` average latency. With the confidence gate at
`0.765`, emitted frame accuracy was `97.32%` with `96.96%` coverage.

## 6. Low-power 5V/3A experiment

Use this when testing from a 20,000 mAh 5V/3A power bank. It keeps the same
calibrated classifier, crop, confidence gate, and 5 FPS selected-frame schedule,
but disables the detector, uses the validated two-thread low-power profile,
pins two CPU cores, increases process niceness, and paces frames instead of
running flat out.

```powershell
python 43_low_power_5v3a_experiment.py \
  --mode benchmark \
  --test clf_dataset/test \
  --clf-model runtime_exports/classifier_openvino_model \
  --threads 2 \
  --fps 5 \
  --max-edge 256 \
  --crop-mode mask \
  --crop-scale 0.55 \
  --crop-pad 0.05 \
  --conf 0.765 \
  --high-conf 0.765 \
  --nice 5 \
  --cpu-affinity 0,1 \
  --sample-power-every 0
```

The report is written to
`runtime_reports/low_power/low_power_benchmark_report.json`.

Validated full-set result on the 20,000 mAh 5V/3A bank:

- `461` images
- `5.00` wall FPS
- `114 ms` average latency
- `118 ms` max latency
- `0/461` frames above `200 ms`
- `57.0%` estimated CPU duty cycle
- `97.32%` emitted accuracy
- `96.96%` coverage
- `throttled=0x0` at start and end

Live low-power run:

```powershell
python 43_low_power_5v3a_experiment.py \
  --mode live \
  --camera picamera2 \
  --camera-width 640 \
  --camera-height 480 \
  --picamera-raw-width 2304 \
  --picamera-raw-height 1296 \
  --clf-model runtime_exports/classifier_openvino_model \
  --threads 2 \
  --fps 5 \
  --max-edge 256 \
  --crop-mode mask \
  --crop-scale 0.55 \
  --crop-pad 0.05 \
  --conf 0.765 \
  --high-conf 0.765 \
  --nice 5 \
  --cpu-affinity 0,1
```

## 7. Fan/heatsink 10 FPS performance profile

With the fan/heatsink and 5V/5A supply, the original accurate `224` OpenVINO
classifier sustains 10 FPS without lowering accuracy:

```powershell
python 43_low_power_5v3a_experiment.py \
  --mode benchmark \
  --test clf_dataset/test \
  --clf-model runtime_exports/classifier_openvino_model \
  --threads 4 \
  --fps 10 \
  --cpu-affinity 0,1,2,3 \
  --max-edge 256 \
  --crop-mode mask \
  --crop-scale 0.55 \
  --crop-pad 0.05 \
  --conf 0.765 \
  --high-conf 0.765 \
  --sample-power-every 0 \
  --progress-every 0
```

Full-set result:

- `461` images
- `10.01` wall FPS
- `73 ms` average latency
- `78 ms` max latency
- `0/461` frames above `200 ms`
- `97.32%` emitted accuracy
- `96.96%` coverage
- ended at `60.4'C` and `throttled=0x0`

This is the recommended performance profile when the fan/heatsink and 5V/5A
supply are attached.

CSI camera smoke test passed with:

```powershell
python 43_low_power_5v3a_experiment.py \
  --mode live \
  --camera picamera2 \
  --camera-width 640 \
  --camera-height 480 \
  --picamera-raw-width 2304 \
  --picamera-raw-height 1296 \
  --clf-model runtime_exports/classifier_openvino_model \
  --threads 4 \
  --fps 10 \
  --cpu-affinity 0,1,2,3 \
  --max-live-frames 30
```

The attached camera is detected as `imx708` with modes:

- `1536x864` at `120 fps`, central crop
- `2304x1296` at `56 fps`, full sensor crop
- `4608x2592` at `14 fps`, full sensor crop

Use the `2304x1296` raw mode for live inference so the camera keeps the wider
field of view while outputting a cheap `640x480` RGB frame to the model path.

Live inference now also writes organized local data for the farmer platform:

- root: `agribot_inference_data/`
- per-run folder: `agribot_inference_data/runs/<run_id>/`
- files: `metadata.json`, `summary.json`, `decisions.jsonl`, `events.csv`
- latest pointer: `agribot_inference_data/latest_run.json`

The legacy `--jsonl` path is still written for compatibility.

Historical rejected variants before cooling/power fix:

- `8 FPS`, `2` threads, `224`: full-set run reached only `7.71` wall FPS and throttled.
- `192` and `160` OpenVINO exports: reached 10 FPS but lost too much accuracy.
- `208` OpenVINO export: screening run reached 10 FPS, but emitted accuracy was only `96.05%` at the calibrated gate and the full sweep destabilized the Pi.

## 8. Farmer field platform

The local dashboard is in `agribot_platform/` and uses only Python's standard
library plus static HTML/CSS/JS. Start it on the Pi:

```bash
cd ~/agribot
source ~/agribot_venv/bin/activate
python3 agribot_platform/run_platform.py --host 0.0.0.0 --port 8080 --data-root agribot_inference_data
```

It prints a phone URL such as:

```text
http://192.168.1.25:8080
```

The phone browser stores the most recent loaded run in local storage and the
service worker cache, so the farmer can reopen the platform and still see the
last loaded data even if the page was closed. The Capacitor Android app uses the
same API contract and keeps its own local cache on the phone.

The dashboard and Android app only ask the farmer for the field details that
matter during a scan:

- field name
- active row
- row count
- plants per row

Plant numbering, plant step, row spacing, plant spacing, and move pause are
kept as internal defaults so the farmer does not have to fill extra setup
fields.

Saving this writes `agribot_inference_data/field_layout.json` and updates
`farmer_config.json` for future one-touch runs.

No internet is required, but the phone still needs a local network path to the
Pi. The final field-safe default is the Pi's direct hotspot, so the system works
even when the farm has no router:

```bash
./deployment/18_enable_direct_phone_hotspot.sh
```

Then connect the phone to:

```text
Wi-Fi: Agribot_Field
Pass : configured on-device via AGRIBOT_HOTSPOT_PASSWORD (not committed)
URL  : http://10.42.0.1:8080
```

To connect the Pi to any available Wi-Fi instead, either run:

```bash
./deployment/19_connect_configured_wifi.sh
```

or provide the values non-interactively:

```bash
AGRIBOT_WIFI_SSID='FarmWifi' AGRIBOT_WIFI_PASSWORD='wifi-password' ./deployment/19_connect_configured_wifi.sh
```

Deployment helper:

```bash
./deployment/14_start_field_platform.sh
```

## 9. One-touch farmer mode

Use this for the actual farmer workflow. It starts both the dashboard and live
inference, writes field-mapped data, and is the command used by the autostart
service:

```bash
cd ~/agribot
source ~/agribot_venv/bin/activate
python3 44_farmer_one_touch.py run --config farmer_config.json
```

Convenience helper:

```bash
./deployment/15_farmer_one_touch.sh
```

Install once so the Pi starts Agribot automatically on boot:

```bash
./deployment/16_install_farmer_autostart.sh
```

For the final field system, prefer the all-in-one installer. It installs both
boot services and defaults to direct hotspot mode without switching the current
SSH connection immediately:

```bash
./deployment/20_install_final_field_system.sh
```

To make the Pi prefer a specific Wi-Fi while still falling back to hotspot when
that Wi-Fi is unavailable:

```bash
AGRIBOT_NETWORK_MODE=auto \
AGRIBOT_WIFI_SSID='FarmWifi' \
AGRIBOT_WIFI_PASSWORD='wifi-password' \
./deployment/20_install_final_field_system.sh
```

Set `AGRIBOT_START_NOW=1` only when you intentionally want the setup command to
switch the Pi network immediately. Otherwise the new access mode takes effect
on the next boot.

The farmer map is based on a sequential row scan. The farmer starts at the
configured row and plant number, points the camera at one plant, waits for the
decision, then moves to the next physical plant during the move pause. Each
saved decision includes:

- `field_id`
- `row_id`
- `plant_number`
- `plant_display`, for example `Field 1 / Row A / Plant 12`
- `action`: `Inspect or treat`, `Rescan this plant`, or `No action`

Update `farmer_config.json` before the run to change the field, row, starting
plant, FPS, or camera source.

To archive training/benchmark files that are not needed for field operation:

```bash
./deployment/17_archive_field_cleanup.sh
```

The archive is written under `field_archives/` and verified before originals are
removed. It does not archive the deployed model, platform, or live inference
data.

## 10. Android APK wrapper

The Android app lives in `agribot_android_app/`. It is a Capacitor wrapper for
the Pi dashboard/API; it does not run inference on the phone. The default Pi URL
is the direct hotspot address `http://10.42.0.1:8080`; the app also tries
`http://192.168.1.25:8080` for lab/router testing, and the farmer can change it
inside the app.

Prepare/build from the repo root:

```bash
python3 45_build_android_apk.py
```

This machine now has a local Android toolchain under `.toolchains/`, and the
helper uses it automatically. On a fresh machine, install JDK 21+ and Android
SDK/build-tools first. After those are installed, build:

```bash
cd agribot_android_app/android
./gradlew assembleDebug
```

APK output:

```text
agribot_android_app/android/app/build/outputs/apk/debug/app-debug.apk
dist/agribot-field-app-debug.apk
```

Install on a connected phone:

```bash
adb install -r agribot_android_app/android/app/build/outputs/apk/debug/app-debug.apk
```

The generated Android manifest is configured for local `http://` access to the
Pi, and the phone app caches the latest loaded run in local storage.

## 10.1 Farmer platform and field mapping

The farmer-facing data lives under `agribot_inference_data/` on the Pi:

- `field_layout.json` stores all fields, row counts, row overrides, active field,
  and spacing.
- `runs/<run_id>/decisions.jsonl` stores every plant decision with field, row,
  plant number, confidence, timestamp, and map coordinates.
- `latest_run.json` points the app/platform to the most recent run.

Start the full boot-style controller manually:

```bash
cd ~/agribot
source ~/agribot_venv/bin/activate
python3 44_farmer_one_touch.py run --config farmer_config.json
```

Install or refresh automatic startup:

```bash
cd ~/agribot
sudo ~/agribot_venv/bin/python3 44_farmer_one_touch.py install-autostart --start-now
```

The platform URL is printed by `44_farmer_one_touch.py status`. When the Pi is
running its own hotspot, the default phone URL is:

```text
http://10.42.0.1:8080
```

When the Pi is on the lab/router network, use the Pi address, for example:

```text
http://192.168.1.25:8080
```

The field setup screen supports multiple fields. Each field has default rows,
plants per row, row spacing, and plant spacing. Each row can override plant
count and plant spacing. Live inference reloads `field_layout.json`, so changing
the active field or row layout in the platform updates the scan mapper without
manual row-transition code.

## 11. tomato_crop_disease detector benchmark

`tomato_crop_disease` is the current detector benchmark dataset. It has
YOLO-format labels under `tomato_crop_disease/labels`, and the pass target is
`F1 >= 90%` when evaluating strict per-box localization.

This dataset is not an 8-class classifier benchmark. Its labels are detection
boxes and currently use one foreground class (`tomato_crop_disease`). Use
`--target-metric image_hit_rate --target-value 0.90` when the deployment question
is whether at least one valid crop/disease region is found in the image.

Run this after exporting the OpenVINO detector:

```powershell
python 34_benchmark_tomato_crop_disease.py \
  --det-model runtime_exports/detector_openvino_model \
  --imgsz 256
```

Image-level deployment triage:

```powershell
python 34_benchmark_tomato_crop_disease.py \
  --det-model runtime_exports/detector_openvino_model \
  --imgsz 256 \
  --target-metric image_hit_rate \
  --target-value 0.90
```

Outputs:

- `dataset/runtime_tracking/tomato_crop_disease_detector_results.csv`
- `dataset/runtime_tracking/tomato_crop_disease_detector_best.json`

The benchmark is class-agnostic because this dataset's labels currently use
class id `1`, while the existing crop detector is a single-class detector.

## 12. OpenVino Retry experiment

This keeps the baseline OpenVINO rows intact and writes a separate experiment row
to `dataset/runtime_tracking/openvino_retry_results.csv`.

Defaults:

- classifier exported as OpenVINO FP16 at `imgsz=192`
- detector exported as OpenVINO FP16 at `imgsz=192`
- benchmark `max_edge=192`
- export thread pressure limited to 2 threads for better Pi stability

```powershell
python 16_openvino_retry.py
```

If you want to opt into classifier INT8 calibration anyway, use:

```powershell
python 16_openvino_retry.py --classifier-int8
```

To sweep classifier-only accuracy/FPS without spending time on detector export
and detector benchmarking, use:

```powershell
python 16_openvino_retry.py --classifier-only --clf-imgsz 320
```

To run only the detect+classify benchmark path, use:

```powershell
python 16_openvino_retry.py --full-pipeline-only
```

Outputs are written to:

- `runtime_exports/openvino_retry/classifier_openvino_model/`
- `runtime_exports/openvino_retry/detector_openvino_model/`
- `runtime_reports/accuracy/openvino_retry/`
- `runtime_reports/benchmark/openvino_retry/`
- `dataset/runtime_tracking/openvino_retry_results.csv`
