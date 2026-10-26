# Agribot v3 — field Healthy retraining pipeline

This directory implements the retraining and deployment contract described in
[`docs/training/RETRAINING_HANDOFF.md`](../../docs/training/RETRAINING_HANDOFF.md).
It does **not** claim that the shipped TFLite file is fine-tunable. The candidate is
a fresh nine-class head on timm's DINOv2 ViT-S/14 pretrained backbone, trained on
grouped field data, then calibrated on a separate split and exported to the same
Android I/O contract. Inputs are normalised as RGB `0..1` (not `0..255`) before
ImageNet mean/std, matching `canon.py` and the Android app exactly. See
[`RESULTS.md`](RESULTS.md) for the measured outcome; the first candidate was
rejected and the production baseline was retained.

## Why the shipped weights were not fine-tuned

`field_vits2/best.pt` is absent from the repository, LFS objects and releases. The
shipped `disease_classifier_0.tflite` is an inference graph, not a trainable
checkpoint. This run is therefore named **pretrained-backbone retraining** and uses
the shipped TFLite file only as the baseline comparator.

## Data and leakage controls

| Source | Pinned revision | Use |
| --- | --- | --- |
| [Tomato-Village](https://github.com/mamta-joshi-gehlot/Tomato-Village) | `dfc246014558293987b381c1edb49398d9f570ac` | Field training and grouped test |
| [PlantDoc](https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset) | `4730a233a555b30ee98e0879c63ad25d82407455` | Field object crops + field test |
| [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) | `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a` | Evaluation-only seeded diagnostic sample |

PlantDoc is CC BY 4.0. Tomato-Village and PlantVillage terms must be rechecked for
any redistribution of derived assets; this repository commits no source images.
PlantVillage is **evaluation-only**: the exact 255-file seeded sample used by the
2026-10-15 baseline never enters training or calibration.

Tomato-Village filenames repeat for augmented crops of the same camera photo
(`IMG...jpg` and `IMG..._N.jpg`). Splitting by image would leak a photographed
plant across partitions, so the pipeline groups by camera filename and assigns
whole groups to train/selection-validation/calibration/locked-test (70/10/10/10,
stratified by class and source). PlantDoc groups are source photos; each annotated
leaf box becomes a sample but all boxes from a photo stay in one partition.
Duplicate source bytes and duplicate crops are removed before split assertion.
Development and locked-test group sets are asserted disjoint.

## Pipeline

```bash
source ~/.venvs/agribot-train/bin/activate
python model_training/v3/prepare_data.py                 # manifest + 256 px cache
python model_training/v3/train.py --run-name <name> ...  # checkpoint + metrics
python model_training/v3/calibrate.py --checkpoint ...   # calibration split only
python model_training/v3/make_dev_manifest.py            # photo-level dev replay
python model_training/v3/export_litert.py --checkpoint ...
python model_training/v3/make_bundle.py ...
python model_training/v3/app_eval.py --assets <bundle> --manifest <locked/diagnostic>
python model_training/v3/compare.py ...
python model_training/v3/regen_goldens.py --assets <bundle> --install
```

Evaluation is split into two deliberate reports:

* **Historical diagnostic** (`diagnostic_manifest.json`) reproduces the 15 October
  2026 replay of 952 source-test images. It is a comparator, not clean evidence
  for the new model: many Tomato-Village source-test photos share camera groups
  with the new training split. Those rows are flagged `new_model_clean=false`.
* **Locked test** (`locked_test_manifest.json`) contains grouped Tomato-Village
  test photos, PlantDoc source-test photos excluding exact TRAIN duplicates, and
  the evaluation-only PlantVillage sample. It was opened once after model,
  temperature and bias selection.

## Metrics

The app verdict is evaluated, not just classifier top-1. `Healthy recall` is the
fraction of all Healthy-labeled eligible photos yielding a `HEALTHY` verdict;
Unsure, Other and no-leaf outcomes remain in the denominator. `False Healthy` is
reported on supported disease, unsupported tomato conditions and non-tomato inputs
separately and pooled. Other rejection, wrong known labels, no-leaf and Unsure are
reported alongside recognition so a reduction in rejection cannot masquerade as a
gain. Paired bootstrap intervals resample source groups, not crops.

## Artefacts

Caches, checkpoints and raw logs stay outside git under `~/agribot-data/`.
Compact, auditable JSON summaries are copied to `model_training/v3/results/`.
