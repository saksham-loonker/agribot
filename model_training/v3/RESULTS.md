# Agribot v3 field-Healthy retraining — result

Historical October 20 experiment. See [October 24 recovery](RECOVERY_20261024.md)
for original-checkpoint recovery, training-loop repairs and controlled fine-tuning.

**Outcome: candidate rejected; production model retained.** The retrained candidate
improves Tomato-Village field Healthy recall on the locked test, but it also causes
large supported-disease and unsupported-input regressions. It is not deployable and
no production weights, thresholds, or Android assets were changed.

## What was built

- Grouped, leak-controlled training/evaluation manifests for Tomato-Village,
  PlantDoc, and the evaluation-only PlantVillage diagnostic sample.
- A 256 px cache, DINOv2 ViT-S/14 retraining loop (BF16, RTX 5090), calibration,
  LiteRT export with parity checks, Android-contract bundle builder, exact app
  replay, and group-clustered paired bootstrap.
- A critical preprocessing bug was found and fixed before the final candidate:
  earlier training normalized 0–255 pixels, while the app normalizes 0–1. Those
  runs were invalid and are not used below. The corrected pipeline trains and
  validates on `pixel / 255` and matches the app's crop/resize contract.
- No app/Kotlin changes and no production asset changes.

## Data

| Partition | Items | Groups |
| --- | ---: | ---: |
| Train | 9,054 | 3,105 |
| Selection validation | 1,277 | 440 |
| Calibration | 1,315 | 441 |
| Locked test (manifest) | 980 | 707 |
| Locked test eligible | 969 | 696 |
| Historical diagnostic eligible | 941 | — |

Tomato-Village groups are camera filenames (augmented `_N` crops stay together).
PlantDoc groups are source photos. PlantVillage is evaluation-only. The locked test
was opened once after model, temperature, and Other-bias selection.

## Selected candidate

| Field | Value |
| --- | --- |
| Run | `v3_logit_fix_s17` |
| Backbone | `vit_small_patch14_dinov2.lvd142m`, fresh 9-class head |
| Recipe | 25 epochs, 2 warmup, 6 unfrozen blocks, BS 64, logit-adjusted CE, class/source sampling |
| Selected epoch | 8 |
| Calibration | temperature 0.91662, Other bias fixed at production 0.75 |
| Classifier SHA-256 | `fdd8cdadb203b225e0a772443a9d49baa9b09398cb016769432294265602acb9` |
| Size / contract | 86,702,380 bytes; `[1,224,224,3]` float32 → `[1,9]` |
| LiteRT parity | passed; worst max abs logit error `1.86e-5` |

The candidate was selected on the grouped development replay for the Healthy
objective before opening the locked test. Other-bias was held at production 0.75
rather than maximizing a crop-level sweep that over-rewarded rejection.

## Development replay (grouped calib photos; selection only)

| Metric | Baseline | Candidate |
| --- | ---: | ---: |
| Supported app-verdict correctness | 466/484 (96.3%) | 320/484 (66.1%) |
| Healthy recall | 20/26 (76.9%) | 21/26 (80.8%) |
| False Healthy, supported disease | 0 | 5 |
| False Other, supported | 0 | 7 |

The development signal was already a trade: +3.9 pp Healthy, −30.2 pp supported
correctness. The candidate was still taken to the locked test because the Healthy
gain was the primary objective and the supported regression was the key risk to
measure.

## Locked final test (969 eligible photos)

| Metric | Baseline | Candidate | Delta (paired bootstrap) |
| --- | ---: | ---: | ---: |
| Healthy recall | 61/81 (75.3%) | 56/81 (69.1%) | −6.2 pp (95% CI −21.9, +9.4) |
| Supported app-verdict correctness | 619/666 (92.9%) | 423/666 (63.5%) | −29.4 pp (95% CI −36.8, −22.2) |
| Supported-disease correctness | 558/585 (95.4%) | 367/585 (62.7%) | — |
| False Healthy, all non-Healthy | 4/888 (0.5%) | 44/888 (5.0%) | +4.5 pp (one-sided 95% upper +5.8) |
| False Healthy, unsupported tomato | 1/144 (0.7%) | 36/144 (25.0%) | — |
| False Healthy, non-tomato | 2/159 (1.3%) | 1/159 (0.6%) | — |
| Unsupported false-known | 20/303 (6.6%) | 52/303 (17.2%) | +10.6 pp |
| Unsupported explicit Other | 271/303 (89.4%) | 203/303 (67.0%) | −22.4 pp |

Healthy recall by source:

| Source | Baseline | Candidate |
| --- | ---: | ---: |
| Tomato-Village | 8/23 (34.8%) | 22/23 (95.7%) |
| PlantDoc | 3/8 (37.5%) | 6/8 (75.0%) |
| PlantVillage | 50/50 (100%) | 28/50 (56.0%) |

Supported per-class app recall:

| Class | Baseline | Candidate |
| --- | ---: | ---: |
| Early blight | 134/139 (96.4%) | 28/139 (20.1%) |
| Late blight | 140/148 (94.6%) | 89/148 (60.1%) |
| Leaf Miner | 99/103 (96.1%) | 94/103 (91.3%) |
| Magnesium Deficiency | 88/88 (100%) | 83/88 (94.3%) |
| Nitrogen Deficiency | 44/48 (91.7%) | 26/48 (54.2%) |
| Pottassium Deficiency | 8/8 (100%) | 8/8 (100%) |
| Spotted Wilt Virus | 45/51 (88.2%) | 39/51 (76.5%) |
| Healthy | 61/81 (75.3%) | 56/81 (69.1%) |

## Acceptance decision

Predeclared gates from the handoff:

| Gate | Result |
| --- | --- |
| Field Healthy benefit ≥ +10 pp | **Fail**: pooled −6.2 pp; TV +60.9 pp but PV −44.0 pp |
| False Healthy not increased; upper bound ≤ +2 pp | **Fail**: +4.5 pp, upper +5.8 pp; +30 unsupported-tomato false Healthy |
| Other supported macro recall drop ≤ 2 pp / no class >5 pp | **Fail**: supported correctness −29.4 pp; Early blight −76.3 pp |
| Unsupported false-known increase ≤ 2 pp | **Fail**: +10.6 pp |
| Contract/parity | Pass for the standalone export; not integrated |
| Deployment footprint/device latency | Not run; no candidate qualified |

**Verdict: rejected.** The candidate trades field Healthy sensitivity for severe
disease concealment and unsupported-input acceptance, including 30/105
PlantVillage unsupported-disease photos called Healthy. This is a safety failure,
not a threshold-tuning opportunity under the current contract.

## Why this is inconclusive about the field idea

The shipped baseline was developed on the same source families. The locked test is
clean for the new model, but it is **not an independent holdout for the baseline**.
The baseline's near-perfect Tomato-Village and PlantVillage results may reflect
source exposure that the new grouped training split does not have. Conversely, the
candidate's TV Healthy gain is real on these photos but does not transfer to lab
Healthy or unsupported diseases. A larger, independently collected field Healthy
cohort (with agronomist review) is required before a deployment decision.

## Limitations

- Static repeated-frame replay, not a live-camera or device trial.
- No agronomist review of dataset labels; no phone thermal/latency/RAM evidence.
- The PlantDoc Healthy cohort is 8 photos; the PlantVillage sample is 50 Healthy
  and 105 unsupported-disease lab photos. Confidence intervals are wide.
- The candidate is a fresh DINOv2 head, not a fine-tune of the unavailable
  `field_vits2` checkpoint.
- No production assets, manifest, Kotlin code, or goldens were changed.

## Artefacts and reproduction

External artefacts (not committed):

- Runs: `~/agribot-data/runs/v3_logit_fix_s17/` (checkpoint, per-epoch metrics,
  calibration, config, meta).
- Candidate: `~/agribot-data/candidates/v3_logit_fix_s17/` (TFLite, parity,
  bundle).
- Replays: `~/agribot-data/eval/baseline_diagnostic/`, `baseline/`,
  `v3_logit_fix_s17_dev_b075/`, `locked_v3_logit_fix_s17/`,
  `locked_compare.json`.

Compact evidence is committed under `model_training/v3/results/`.

```bash
source ~/.venvs/agribot-train/bin/activate
python model_training/v3/prepare_data.py
python model_training/v3/train.py --run-name v3_logit_fix_s17 --epochs 25 \
  --warmup-epochs 2 --unfreeze-blocks 6 --bs 64 --loss logit_adjusted \
  --sampling class_source --logit-tau 0.5
python model_training/v3/calibrate.py \
  --checkpoint ~/agribot-data/runs/v3_logit_fix_s17/best.pt \
  --run-dir ~/agribot-data/runs/v3_logit_fix_s17
python model_training/v3/export_litert.py \
  --checkpoint ~/agribot-data/runs/v3_logit_fix_s17/best.pt \
  --out ~/agribot-data/candidates/v3_logit_fix_s17
python model_training/v3/app_eval.py --assets <bundle> \
  --manifest ~/agribot-data/work/locked_test_manifest.json --out <dir> --tag <tag>
python model_training/v3/compare.py --baseline <baseline-predictions> \
  --candidate <candidate-predictions> --out <compare.json>
```
