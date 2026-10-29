# App 2.0.1 and training acceptance decision

Date: 24 October 2026. This report supersedes any intermediate claim that the
Healthy adapter was accepted. The original classifier is retained. App fixes
can be deployed independently of a failed model experiment.

## What was recovered and trained

The owner's RTX 5090 PC (`vewmet`) has the original trainable checkpoint at
`/home/genesis/sakshyam-agribot/v2/runs/field_vits2/best.pt`.
Its SHA-256 is `659bd21eceb7ac2ba78abf20b449cb3462f3fb8439778e26f32c6d348e88e35f`.
The recovery report in `model_training/v3/RECOVERY_20261024.md` documents the
trainer repairs and parity with the original inference graph. Genuine fine-tuning
is therefore possible; restarting from a new classification head is unnecessary.

Further eight-epoch experiments used supervised Healthy weighting and distillation
only on non-Healthy training examples that the original teacher classified
correctly. Both sample and class balancing were explored. Head-only adaptation
kept backbone features deterministic; one-block adaptation was also tested.
Candidates that lost disease retention were rejected. Logit-based RL was not used.

The most promising experiment blended the original and adapted heads equally
over one unchanged DINOv2 backbone. It retained the reference distribution when
any calibrated non-Healthy alternative reached 0.4 probability, or when the
adapter would reduce Healthy probability. This conservative policy was selected
using development analysis. It still failed broader regression, so it is not
included in the runtime assets.

## Exact exported-model regression

These are complete static-photo app-policy replays, including detection, crops,
normalization, calibration and repeated-frame plant consensus. Repeated frames
are not independent observations. The original and candidate use the final app
quality-admission policy. All runs completed with zero image-processing errors.

| Set | Raw photos | Eligible scalar labels | Healthy original → adapter | Supported disease correct original → adapter | Disease false Healthy original → adapter |
|---|---:|---:|---:|---:|---:|
| Development selection | 693 | 692 | 20/26 → 23/26 | 446/458 → 446/458 | 0 → 0 |
| Public diagnostic regression | 952 | 941 | 64/80 → 67/80 | 521/558 → 521/558 | 1 → 1 |
| Taiwan additional source | 127 | 127 | 21/22 → 21/22 | 11/20 → 11/20 | 0 → 0 |
| Previously opened grouped test | 980 | 969 | 61/81 → 68/81 | 558/585 → 558/585 | **1 → 4** |

The three additional false Healthy calls are Leaf Miner reference labels. Their
original result was UNSURE. Unchanged disease-correct counts alone would have
missed this regression. Healthy and Miner examples occupy overlapping confidence
ranges, so another threshold tuned to these cases would not establish a reliable
solution. Unsupported-condition false known counts were unchanged in these sets.

These sets must not be pooled: source overlap and differing selection rules exist.
Development was used for selection. Public-source regression can overlap the
training/calibration source distribution. The grouped test was first opened on
20 October; it is no longer blind. The Taiwan data is additional-source material,
but novelty to the original model's training cannot be guaranteed. Taiwan Late
blight accuracy remains 11/20, including six unknown-condition results; this
release does not claim to have solved field generalization or false rejection.

Full denominators, scopes, provenance and errors are in
`model_training/v3/results/production_validation_20261024.json`.

## Retained and rejected artifacts

Retained classifier SHA-256:
`12b9b17194ac5ba779126274d1a475016478e1b07ac6c2170fe51c2ef816af39`.
Detector SHA-256:
`0a8c9128544726348a8400ec2f481b5985f47a778b601f4025be582eaa30a0d4`.
The original labels, canonical RGB normalization, Other bias 0.75, temperature
0.8814700841903687, disabled energy rejection and reference goldens are preserved.

Rejected export SHA-256:
`9d0cbd131f04e549a8c55603ccf5208b2d4a98c7a124ee81787212c4f7e46807`.
It remains outside runtime assets at
`/home/genesis/agribot-data/production_20261024/conservative05/`.
Its self-contained trainable `pytorch_bundle.pt` has SHA-256
`f9d7dabd172a3c0d9df4807447c683d076ea9625df870766e6fa3c979725df2e`;
`export_head_blend.load_export_bundle(path)` restores it without the legacy
checkpoint directory. The run uses EMA weights from
`retention_h3_a2_b0_20261024/last.pt`.

The rejected graph passed random and 13 real-input numerical parity, with real
maximum logit error 9.06e-6, and all four phone golden tests. These checks establish
runtime fidelity, not fitness for deployment. A 100-call CPU proxy measured
13.76 ms original versus 13.78 ms adapter median; it is not phone performance.

The added downloaded-photo phone check initially exposed a fixture decoding
difference: source ICC profiles were copied into PNGs, so Android colour
management changed RGB values relative to PIL's reference. The phone fixtures
now store identical raw RGB pixels without embedded profiles and record RGB
SHA-256 separately from source identity. This correction does not change model
weights, input normalization or the host regression results. It is not evidence
about live-camera or colour-managed photo-import accuracy.

## App changes

- Detail-free black, white and uniform frames request a clearer view and add no
  evidence. Structured dim/soft frames remain eligible with the existing quality
  warning. Blocking every quality warning was tested and rejected because it
  discarded valid disease photos.
- Both scan modes use the same quality admission helper. Rejected frames clear
  stale stability state. Cancellation is propagated, and stale scan/save
  completions cannot overwrite a newer scan's UI state.
- English and Hindi unknown-condition wording no longer asserts that a plant
  must be sick. Isolation advice is conditional on visible illness. Verdict
  headers choose a contrasting foreground instead of white text on every tone,
  fixing the pale unknown-condition header in dark mode.
- Manifest validation rejects unsupported normalization, layout, shapes, label
  mappings, calibration and model paths. LiteRT checks actual float32 tensor
  shapes and rejects nonfinite classifier outputs before evidence accumulation.
- Version 2.0.1, code 3, remains offline, arm64, minimum API 26 and target API 35.
  No database migration or internet permission was introduced.

The connected RMX5101 had an older `com.sakshyam.agribot.field` pilot using a
FastCrop classifier and different input/rejection policy. The rebuilt native
`com.sakshyam.agribot` app is a separate package. Preserve the pilot and its data;
these DINOv2 original-versus-adapter statistics are not a pilot comparison.

## Continuation and release boundaries

SSH alias `agribot-5090` uses the PC's Tailscale address and a dedicated SSH key.
Remote work lives in persistent tmux session `agribot`; laptop Wi-Fi changes do
not kill those jobs. Neither tmux nor Tailscale makes a powered-off or sleeping
machine reachable. No credentials or private keys belong in this repository.

Next model work needs adjudicated Healthy/Miner confusables, source/plant-level
grouping and a fresh field acceptance set. Audit conflicting labels and near
duplicates before adding hard-negative supervision. Retain the original teacher
and deploy only after paired false Healthy, per-disease, unsupported-condition,
export and phone checks pass. Do not relabel the three failed cases without an
expert review or describe another run selected on these sets as a blind test.

The locally installable build is debug-signed. The optimized release APK remains
unsigned unless the owner's release signing configuration is supplied. Dataset,
model licensing and agronomy/Hindi review remain explicit release-owner tasks;
no public release or repository merge is performed automatically.

Final build and device evidence is saved alongside the APKs under
`dist/agribot-2.0.1-20261024/`, mirrored on the PC at
`/home/genesis/agribot-data/production_20261024/app_device_verification.json`,
and summarized in the PR description after verification. CI performs host build
checks; it does not imply that a physical phone is attached to the CI runner.
