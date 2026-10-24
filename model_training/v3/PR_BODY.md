# Improve field Healthy recognition with validated retraining

## Outcome

**Draft / do not merge.** A retrained candidate was evaluated and rejected. The
production classifier, manifest, Android assets, thresholds, Kotlin code, and
goldens are unchanged.

The candidate improves Tomato-Village Healthy recall on the locked test
(22/23 vs 8/23), but it fails the safety and supported-class gates:

- Locked Healthy recall: 56/81 (69.1%) vs baseline 61/81 (75.3%).
- Supported app-verdict correctness: 423/666 (63.5%) vs 619/666 (92.9%).
- False Healthy, all non-Healthy: 44/888 (5.0%) vs 4/888 (0.5%); paired
  one-sided 95% upper bound on the increase is +5.8 pp (gate ≤ +2 pp).
- False Healthy on unsupported tomato photos: 36/144 (25.0%) vs 1/144 (0.7%).
  This includes 30/105 PlantVillage unsupported-disease photos called Healthy.
- Early blight recall fell from 96.4% to 20.1%; Late blight 94.6% → 60.1%;
  Nitrogen Deficiency 91.7% → 54.2%.

Full evidence: [`model_training/v3/RESULTS.md`](RESULTS.md) and
`model_training/v3/results/`.

## What changed

- Added grouped, leak-controlled manifest/cache preparation for Tomato-Village,
  PlantDoc, and evaluation-only PlantVillage diagnostics.
- Added DINOv2 ViT-S/14 retraining, calibration, LiteRT export with parity,
  Android-contract bundle, exact app replay, and group-clustered bootstrap.
- Fixed a training normalization bug (0–255 vs 0–1) before the final candidate.
- No production inference code or assets changed.

## Checks

- `python -m compileall model_training/v3`: pass.
- `python -m pytest -q tests`: 52 passed, 1 skipped (optional public-media check).
- `./gradlew testDebugUnitTest`: 56 tests, 0 failures.
- Production classifier SHA-256 remains
  `12b9b17194ac5ba779126274d1a475016478e1b07ac6c2170fe51c2ef816af39`.
- Detector SHA-256 remains
  `0a8c9128544726348a8400ec2f481b5985f47a778b601f4025be582eaa30a0d4`.
- Candidate LiteRT parity passed (worst max abs logit error `1.86e-5`).

## Why no deployment

The locked test shows disease concealment and unsupported-input acceptance, not a
safe Healthy gain. The baseline was developed on these source families, so the
locked test is clean for the new model but not independent for the baseline. A
larger, independently collected and agronomist-reviewed field Healthy cohort is
needed before the next attempt.

## Follow-up

- Collect independent field Healthy/lookalike photos and review disputed labels.
- Train with Healthy/lookalike balance while explicitly guarding unsupported
  disease inputs.
- Re-run the predeclared gates on a new locked cohort; do not tune on this one.
