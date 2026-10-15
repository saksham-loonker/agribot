# Agribot retraining and Android integration handoff

Prepared: 2026-10-15. Repository: https://github.com/saksham-loonker/agribot.
Audience: an autonomous implementation agent with one local NVIDIA RTX 5090.

## 1. Objective, authorization, and completion

Improve recognition of genuinely healthy tomato leaves in field conditions and
reduce erroneous `Problem not in our list` verdicts on supported conditions,
without increasing disease concealment or losing rejection of unsupported inputs.
Optimize the complete Android verdict, not just classifier top-1 accuracy or the
frequency of Healthy/Other outputs.

The owner requests independent execution: download accessible datasets, install
project-local dependencies, implement missing tooling, choose and run justified
experiments on the provided GPU, export and integrate a qualifying model, test,
push a dedicated branch, and **open a pull request against main without asking
for routine implementation decisions**. A plan alone is not the deliverable.
Complete applicable phases below and record why conditional branches were skipped.
Opening the PR is authorized; merging, publishing a release, paid cloud jobs, and
accessing unavailable private data are outside this handoff. Do not send messages
to people or claim to have reported to the originating agent: the PR and final
report are the durable handback that the owner can bring to that agent.

Autonomy does not make absent data, credentials, agronomy expertise, phone hardware,
or statistical evidence available. Continue independent work through recoverable
failures. If a prerequisite cannot be obtained, deliver the code, actual results,
and a draft PR with the exact blocker. Never fabricate training, clean holdout
performance, device measurements, expert label review, or passing checks. If no
candidate qualifies, retain the current app model and deliver the experiments and
recommendation in the PR. Successful completion can be a justified no-deployment
result; replacing weights at any cost is not the objective.

## 2. Current problem and measured evidence

The owner reports frequent `Problem not in our list` responses from phone scans.
The rough frequency is anecdotal and has not been measured. No camera is needed
for the initial retraining work. A downloaded-image investigation already used
the **exact shipped TFLite assets and app decision rules**. Read
[baseline findings](BASELINE_FINDINGS_20261015.md) and
[evaluation harness](../../tests/model_evaluation/README.md) before changing code.

952 source-labeled images were downloaded; excluding 11 exact PlantDoc TRAIN/TEST
overlaps leaves 941 in the primary diagnostic. Supported-condition outcomes:

| Source | Supported photos | Correct app verdict | False Other | Healthy correct |
| --- | ---: | ---: | ---: | ---: |
| Tomato-Village Variant-A TEST | 461 | 425/461 | 0/461 | 11/22 |
| PlantDoc TEST | 27 | 17/27 | 4/27 | 3/8 |
| PlantVillage sampled tomato images | 150 | 143/150 | 2/150 | 50/50 |

The other 303 primary photos have unsupported source labels. PlantDoc contributes
198 of them: 39 unsupported tomato conditions and 159 non-tomato photos.

Important findings and limitations:

- Tomato-Village Healthy errors: seven Leaf Miner verdicts and four Unsure.
  PlantDoc Healthy errors: two Other and three Unsure. Inspect their labels and
  symptoms before treating these as verified Healthy training positives.
- Author-box PlantDoc crops still yielded Other for 13/60 supported crops,
  including 11/27 Healthy-labeled crops from only eight photos. Detection covered
  53/60 supported boxes at IoU >= 0.5; top-six selection covered 52/60. Classifier,
  crop, domain, and label issues deserve attention alongside detector errors.
- Removing Other bias +0.75 improved PlantDoc supported correctness from 17/27 to
  20/27 and false Other from 4/27 to 2/27, while unsupported inputs receiving known
  labels rose from 20/198 to 26/198. No Healthy-bonus experiment was run.
- Two rejected PlantVillage photos share recorded leaf identifier `GHLB2 Leaf 155`.
  They are correlated. Ten sampled images pixel-match golden fixtures. A pHash
  sensitivity analysis removes 22 additional correlated Tomato-Village photos.
- These sources were used during original model development. The original full
  image inventory, cache, split membership, and trainable checkpoint are not in
  this checkout. This is a source replay, **not an independent holdout**. Removing
  detected duplicates cannot prove independence from the original training.
- Dataset labels have not received a new agronomist review. Static replay repeats
  the same image to exercise consensus; frames are not independent photos and do
  not validate camera conversion, motion, tracking, or thermal behavior.
- At the investigation baseline: 105 JVM tests, 53 Python tests, four physical-device
  golden tests, 13 golden fixtures, and all 952 Python/Kotlin decision comparisons
  passed. These are historical results, not automatic passes for your changes.

The previous detailed notebook, raw logits, CSVs, cached images, and logs live only
in the originating workspace under ignored `dist/model-evaluation-20261015/`.
They are **not carried by git pull**. The findings and harness are checked in;
recreate needed raw outputs rather than assuming that directory exists.

## 3. Exact deployed model and application contract

Use `agribot_android_app/android/app/src/main/assets/model_manifest.json` as the
runtime source of truth. Record its hash and git revision before any changes.

| Component | Current value |
| --- | --- |
| Classifier run recorded in calibration | `field_vits2` |
| Architecture | `vit_small_patch14_dinov2.lvd142m`, DINOv2 ViT-S/14 |
| Classifier asset | `models/disease_classifier_0.tflite` |
| Classifier SHA-256 | `12b9b17194ac5ba779126274d1a475016478e1b07ac6c2170fe51c2ef816af39` |
| Classifier I/O | one NHWC float32 `[1,224,224,3]` input; nine logits |
| Detector | YOLOv9-T, LibreYOLO, NHWC float32 `[1,320,320,3]` |
| Detector asset | `models/leaf_detector.tflite` |
| Detector SHA-256 | `0a8c9128544726348a8400ec2f481b5985f47a778b601f4025be582eaa30a0d4` |
| Detector output | `[1,5,2100]`: x1, y1, x2, y2, score in letterboxed input pixels |
| Detector decoding | confidence 0.35; NMS IoU 0.6; at most 16 detections |
| Classification selection | up to six detected leaf crops in Check a plant |
| Calibration | Other bias +0.75; temperature 0.8814700841903687; energy gate off |

The label order is an API contract, including the existing spelling:

```text
0 Early_blight
1 Healthy
2 Late_blight
3 Leaf Miner
4 Magnesium Deficiency
5 Nitrogen Deficiency
6 Pottassium Deficiency
7 Spotted Wilt Virus
8 Other
```

Classifier preprocessing is RGB -> detector box padded 10%, floor/ceil and bounds
clipping -> centered square of the short side -> canonical resize -> /255 ->
ImageNet mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`.
Canonical shrinking uses area averaging; enlargement uses half-pixel bilinear.
Detector preprocessing is centered letterbox with padding 114 and /255.
Use `model_training/v2/final/canon.py`, the bundled `canon_reference.py`, and
Android `CanonicalImageOps.kt` to verify exact behavior, not generic torchvision
resize defaults. Augmentation may vary during training; evaluation must match.

For member logits z_i, inference computes `log(mean_i softmax(z_i))`, adds Other
bias at index 8, then applies softmax after dividing by temperature. The shipped
ensemble has one member. Positive scalar temperature preserves per-leaf argmax
but changes confidence, averaging behavior, and thresholded plant verdicts.
Calibrate the exact exported/combined representation, including its energy
definition; do not mix raw-logit and log-probability energy thresholds.

`PlantEvidence` averages per-track probabilities and then averages leaves with
capped weights. Observation weight is detectorScore * frameQuality, clamped to
0.05..1.0. Current policy: minimum three frames/five observations; per-leaf weight
cap 3; Healthy >=0.60 with margin >=0.20; disease >=0.50 with margin >=0.15;
Other >=0.50. Partial disease can override Healthy/Other/uncertain evidence when
two distinct leaf tracks each support a disease at >=0.70 with at least two hits.
Check a plant waits for two additional stable frames or stops at 14 frames.
Frame quality weights evidence and supplies hints in this flow; do not invent a
strict quality gate. No detected leaves does **not** trigger whole-frame fallback.
`UNKNOWN_CONDITION` renders as `Problem not in our list`; Unsure is a different
outcome. Preserve these distinctions in metrics.

## 4. Code map and existing workflow hazards

Paths below are relative to the repository root.

| Path | Role / required attention |
| --- | --- |
| `model_training/v2/build_cache.py` | Historical dataset preparation; hardcoded home paths and missing inputs; replace/configure ingestion and splits rather than executing blindly. |
| `model_training/v2/train_field.py` | Historical field training; defaults to MobileNet, not the shipped ViT; already uses CE(logits + log_prior, labels), smoothing 0.05, weighted domain sampling, optional KD and EMA. |
| `model_training/v2/train_clf.py`, `eval_field.py`, `ens_eval.py` | Supporting historical training/evaluation; audit assumptions before reuse. |
| `model_training/v2/final/stage_a.py` | Historical ONNX/calibration; pools val_clean and val_leaky and optimizes coarse source scores. Replace with clean, grouped, app-aware validation. |
| `model_training/v2/final/convert_litert.py` | Preferred existing ViT conversion pattern: disable fused attention, wrap NCHW model with NHWC input, convert with `litert_torch`, check parity. |
| `model_training/v2/final/stage_b.py` | Historical bundler; assumes missing caches/checkpoints, deletes its output directory, uses obsolete top-four mean voting and whole-frame fallback. Do not use its composite metric as current app behavior. Reimplement safely for the new run. |
| `model_training/v2/export_clf.py`, `tflite_check.py` | Older export utilities with architecture/results/shape assumptions; not a complete current deployment pipeline. |
| `model_training/v2/results/final/calibration.json` | Historical selected run/calibration, not trainable weights. |
| `model_training/v2/bundle/` | Mirrored manifest, 13 golden PNGs and reference expectations. |
| `agribot_android_app/android/ml/src/main/java/com/sakshyam/agribot/ml/vision/` | `AgribotLeafVision.kt`, `EnsembleDecision.kt`, `VisionManifest.kt`, `CanonicalImageOps.kt`, detector decoder and runtime. |
| `agribot_android_app/android/domain/src/main/java/com/sakshyam/agribot/domain/scan/PlantEvidence.kt` | Evidence aggregation and `VerdictPolicy`; its thresholds are source defaults, not currently parsed from the model manifest. |
| `agribot_android_app/android/featurescan/src/main/java/com/sakshyam/agribot/featurescan/check/CheckPlantViewModel.kt` | Top-six inference, tracking/evidence use, stop logic. Also inspect walk flow if shared behavior changes. |
| `agribot_android_app/android/featurescan/src/main/java/com/sakshyam/agribot/featurescan/Conditions.kt` | Label-to-condition text mapping; preserve label keys. |
| `agribot_android_app/android/featurescan/src/main/res/values/strings.xml` | Verdict UI strings; verify actual resource location if repository moves. |
| `tests/model_evaluation/` | Download, exact TFLite inference, oracle crops, static app replay, analysis, notebook, Kotlin replay and validation tooling. |
| `.github/workflows/android-native.yml` | Canonical Python/JVM/lint/APK CI checks. |

Historical training loads the entire uint8 image cache onto the GPU: 448x448 RGB
costs roughly 0.60 MB per sample before model/optimizer/activation memory. Do not
assume an entire expanded corpus fits the 5090. Use a streamed DataLoader/pinned
host cache and bounded prefetching, or a measured partial GPU cache.

The evaluation harness was authored on Windows. Before Linux execution fix
`verify_kotlin_replay.py` JDK discovery (`java.exe`/`javac.exe` and `.toolchains`),
stdlib discovery, and output paths; prefer JAVA_HOME/PATH and configured Gradle
cache. Parameterize model bundle, output directory, source revisions, policy,
and run identity. Current outputs and checks contain baseline-specific dates,
counts, bias 0.75 comparisons, and strict physical-device prerequisites. Preserve
the historical reproduction mode, add a general candidate mode, and report
unavailable device checks as NOT RUN, not passing. Include model/split/config
hashes in resume keys so old predictions cannot be reused for a new candidate.
The downloader currently resolves branch names then records revisions: supply the
pinned revisions below explicitly for exact reproduction instead of relying on
future branch heads or an existing local cache.

## 5. Phase 0 — repository and RTX 5090 preflight

1. Read applicable AGENTS.md instructions. Preserve all preexisting user changes.
   In a clean clone, update main with a fast-forward pull, then make a dedicated
   `training/healthy-field-<date>` branch. In a dirty checkout use a separate
   worktree; do not reset, stash without explanation, force-push, or train on main.
2. Run `git lfs install` and `git lfs pull`. Verify asset hashes and that assets are
   actual model bytes, not LFS pointer files. Record baseline revision and hashes.
3. Verify authenticated push and PR creation access early with `gh auth status`
   (or an authenticated GitHub API client). Never print credentials or commit
   tokens. Missing authentication does not prevent local experimentation.
4. Prefer native Linux on the GPU host; WSL2 is acceptable after verifying CUDA
   passthrough. A provided Windows host may train there, but use a compatible
   Linux environment for conversion when required. Do not reuse the originating
   Windows CPU-only virtualenv or install a CPU-only torch wheel by accident.
5. Inspect `nvidia-smi`, available VRAM/RAM/disk, driver, Python, CUDA runtime, and
   CPU cores. Query PyTorch's official current installation selector for a
   compatible CUDA build and confirm Blackwell support with real operations.
   Create isolated training and conversion environments if dependencies conflict;
   pin exact successful versions and save lockfiles. Do not upgrade system drivers
   or change system CUDA installations silently when project-local wheels suffice.
6. Check `torch.cuda.is_available()`, GPU name/capability, matmul, BF16 forward and
   backward on the intended timm model, augmentation ops and optional compile.
   Start eagerly; enable compile only after numerical and throughput checks.
   Resolve torchvision/CUDA ABI problems before launching a long run.
7. Install JDK 17, Android SDK 35/build-tools 35.0.0, and Git LFS. Match Gradle and
   Android dependencies to repository configuration. Keep training and export
   dependencies separate from the Android runtime contract.

Official references to recheck on the GPU host:
[PyTorch installation](https://pytorch.org/get-started/locally/),
[NVIDIA Blackwell compatibility](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html),
[PyTorch to LiteRT](https://developers.google.com/edge/litert/conversion/pytorch/overview).
Compatibility is established by a successful smoke test, not a hardcoded claim
that a particular wheel or CUDA toolkit is installed. Run the converter smoke
test on the chosen nine-class ViT before investing in full training.

## 6. Phase 1 — recover weights and reproduce the baseline

Search tracked manifests/results, LFS, releases and owner-provided accessible
artifact locations for `field_vits2/best.pt` plus its exact metadata. Verify
architecture, class order, integrity, provenance and baseline predictions. The
TFLite model is an inference artifact, **not a directly fine-tunable PyTorch
checkpoint**. Do not claim supervised fine-tuning of the shipped weights if no
matching trainable checkpoint is recovered.

If absent, autonomously initialize the same architecture from documented DINOv2
pretrained weights, create the nine-class head, and train a replacement using the
new clean training split. Clearly name this `pretrained-backbone retraining`.
The shipped TFLite model remains the deployment comparator. Its logits can be
cached on training images for an optional distillation experiment, but supervised
labels take precedence, particularly on healthy failures. Do not distill its
Healthy mistakes or use its predictions as verified ground truth. No reverse
conversion of TFLite parameters is required.

Recreate the pinned diagnostic and compare baseline counts before interpreting
candidate gains. Run existing regressions early. Save immutable baseline outputs
separately from candidate runs. Port the harness as described above; do not call
old static replay a camera evaluation or treat familiar source data as unseen.

Useful starting commands, from repository root after installing the appropriate
environment (adapt paths/parameterization as implemented):

```bash
python -m pip install -r requirements-test.txt
python -m pip install -r tests/model_evaluation/requirements.txt
python tests/model_evaluation/download_public.py
python tests/model_evaluation/evaluate_public.py --workers 4 --threads 2
python tests/model_evaluation/evaluate_public.py --stress
python tests/model_evaluation/run_controls.py
python tests/model_evaluation/analyze_public.py
```

Run full replay verification/notebook creation after building JVM classes and
handling the documented host-specific and device-test requirements. Existing
commands are baseline starting points, not a claim that a complete retraining CLI
already exists. Implement missing commands and document exact final invocations.

## 7. Phase 2 — source acquisition, labels, leakage controls and splits

Original dataset families: Tomato-Village, PlantDoc, Taiwan Mendeley field images,
and PlantVillage. FieldPlant also appears in historical release/licence metadata;
verify whether it is actually included before claiming it trained the classifier.

| Diagnostic source | Pinned repository revision | Historical use / preparation |
| --- | --- | --- |
| `mamta-joshi-gehlot/Tomato-Village` | `dfc246014558293987b381c1edb49398d9f570ac` | Variant-A eight classes; train/val/test; full images and detector crops. |
| `pratikkayal/PlantDoc-Object-Detection-Dataset` | `4730a233a555b30ee98e0879c63ad25d82407455` | VOC XML leaf annotations; mapped tomato Healthy/Early/Late; remaining labels Other. |
| `spMohanty/PlantVillage-Dataset` | `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a` | `raw/color`; deterministic diagnostic seed 20261015, 50 each EB/Healthy/LB and 15 each seven unsupported tomato categories. |
| Taiwan Mendeley `ngdgg79rzb` | Resolve exact version/files/checksums during acquisition | Historical script maps folder `health` to Healthy, Late blight to Late_blight and remaining categories Other. Check actual label semantics and provenance first. |

Obtain complete accessible training portions, not just the 952 diagnostic images.
Use source APIs/direct archives with checksums, bounded retries and resumability.
Retain source URL, revision/version, licence text, attribution and redistribution
conditions. Check dataset access/usage terms before incorporation; accessible
downloads do not imply permission to redistribute images or trained artifacts.
If a family is unavailable or cannot be used, exclude it with an explicit reason
and proceed with usable sources. Add other field sources only with documented
provenance, usable labels, and the same leakage controls. No paid acquisition.

Create a versioned per-original-image manifest with stable ID, URL/local path,
SHA-256, decoded-pixel hash, source/revision, raw/canonical label, species, box
annotations, plant/leaf/session/group IDs when known, duplicate-cluster ID,
split, licence, review status, and any exclusion reason. Retain full image-to-crop
lineage. Deduplicate within and across sources using byte/pixel hashes, pHash
candidates and visual checks. Link original/augmented/segmented versions and
repeated leaf photos before splitting. Split original groups first, generate all
crops/augmentations afterward, and assert no group overlap programmatically.
Cross-split background/session duplicates also matter.

Preserve source TEST splits for diagnostics. Do not turn the evaluated failures
into training examples and continue calling the same benchmark held out. If any
diagnostic image is explicitly used for training, mark it contaminated and retire
its entire related group from generalization claims. Development comparisons may
still report it separately. A newly grouped split of old sources can be clean for
the **new** training run while still being of unknown familiarity to the old model.

Use train, model-selection validation, calibration validation, and a locked final
test split with source/class stratification where group counts permit. Default
allocation for otherwise unsplit groups: 70/10/10/10; honor existing tests first
and adjust for sparse classes with a logged justification. Keep a field-source
holdout when data permit. Truly new source/site data are preferred for independent
generalization claims. Do not repeatedly inspect the final test during decisions.
If data are too small for four splits, use grouped development folds and a separate
locked test, explaining uncertainty and calibration reuse. Never augment sample
counts to claim more independent plants.

Review all high-loss Healthy samples and Healthy/Leaf Miner confusions, including
box coverage and species. Visually screen source labels, record judgments and
ambiguity, and quarantine uncertain cases; agent inspection is not agronomist
confirmation. Do not silently relabel all inconvenient Healthy examples as Other.
Keep Other examples from unsupported tomato disease, non-tomato plants, and
appropriate nuisance inputs as separately measurable groups. Mixed-label scenes
need object-level evaluation and a stated plant-level reference policy; no forced
single healthy label for a scene containing diseased leaves.

## 8. Phase 3 — experiments and autonomous selection

Start with the existing architecture, resolution, label order, detector and app
policy. Establish a controlled supervised reference before changing several
things together. The historical field trainer already uses logit-adjusted CE:
`CE(logits + log_prior, y, label_smoothing=0.05)` with priors derived from weighted
domain sampling. Do not describe adding that same loss as a novel reward method.
Recompute priors on the training split and account for the effective sampling
distribution; combining strong resampling, weights and adjustment can overcorrect.

| Option | What changes in training/model | What changes in Android | When to run |
| --- | --- | --- | --- |
| Frozen baseline and calibration-only | No weights; sweep temperature and Other bias on calibration validation, evaluate full app policy | Manifest calibration and regenerated expectations only if selected | Required comparator, including original settings. |
| Healthy inference bonus | Add a class-specific Healthy bias after combining member probabilities | New parsed manifest bias field/vector and matching `EnsembleDecision` behavior/tests; currently unsupported | Optional diagnostic comparator, validated against false Healthy; never unconditional reward for predicting Healthy. |
| Linear probe/head warm-up | Freeze pretrained/backbone weights, train nine-class head on clean groups | Replacement classifier asset, hashes, calibration and goldens | Required when initializing a fresh head; useful conservative comparator with recovered weights. |
| Targeted supervised fine-tuning | Unfreeze last ViT blocks, then more blocks only if validation supports it; field-balanced sampling | Same model contract; replace asset/calibration/goldens | Preferred main experiment. |
| Class-weighted CE | Weight verified classes based on training prevalence/validation behavior; clipped weights | No new runtime logic | Required limited comparison against ordinary CE and current logit-adjusted CE. |
| Logit-adjusted supervised CE | Train with prior-dependent adjustment and declared coefficient | Usually no runtime change beyond new weights/calibration; document whether adjustment is training-only or also applied at inference | Required limited comparison; not RL. |
| Focal loss / hard-example sampling | Focus on verified difficult samples, capped to avoid noise amplification | Same model contract | Conditional if ordinary/weighted CE fails; mine training errors only. |
| Distillation | Label CE plus temperature-scaled teacher KL on appropriate training subsets | Only student asset if deployed; teacher not shipped | Conditional if catastrophic forgetting or smaller student is justified. Teacher agreement is not an accuracy metric. |
| New field data / augmentation | Increase representative healthy examples and disease lookalikes, camera-relevant crop/blur/light/compression variation | Same contract unless preprocessing explicitly changes | Data quality and coverage work is required; extra sources conditional on access. |
| Detector fine-tune | Fine-tune leaf detector using grouped bounding-box data; reassess precision/recall and crop quality | Replace detector asset/hash; update decode metadata only for an intentional contract change; re-evaluate all classifier crops | Conditional on measured detector/crop failures remaining material. |
| Alternate backbone or ensemble | MobileNet/MobileViT/other exportable student, or multiple classifier members | Manifest architecture/input/members; possible latency/memory/runtime changes | Conditional on deployment performance needs after a single ViT baseline. |
| Quantization | FP16 weights or representative-data INT8/PTQ/QAT after choosing float candidate | Dtype/scale/zero-point handling if I/O changes; interpreter/delegate tests; updated assets and manifest | Conditional; preserve float32 I/O by default. No blind INT8 replacement. |
| Open-set/energy or hierarchical head | New rejection method, binary species/health head, or changed taxonomy | Explicit new output contract, parser/runtime/aggregation/condition mappings and comprehensive tests | Last resort requiring evidence; out of the default nine-class scope. |

There is no need for reinforcement learning just to improve this nine-class
classifier. A label-dependent logit reward can usually be expressed as supervised
loss. An unconditional Healthy reward optimizes output frequency and is excluded.
For theoretical context see
[logit adjustment](https://arxiv.org/abs/2007.07314) and
[confidence calibration](https://proceedings.mlr.press/v70/guo17a.html).

Initial training ranges are starting points, not promised optimal settings:

- Head warm-up: 3–5 epochs, AdamW, head LR about 1e-3, frozen backbone.
- Fine-tune: 15–40 epochs, backbone LR 1e-5..5e-5 and head LR 1e-4..5e-4,
  weight decay 0.01..0.05; cosine decay with 1–3 warm-up epochs; grad norm cap 1.
- Choose measured batch size, initially 32, then increase within VRAM headroom;
  use BF16 after smoke tests and gradient accumulation when useful. Keep training
  numerics/checkpoint selection reproducible and evaluation deterministic.
- Compare smoothing 0 and 0.05, moderate source balancing, ordinary CE, capped
  class weights, and logit-adjustment coefficient 0/0.5/1 through a small staged
  search. Do not run the entire Cartesian product. Keep all other variables fixed
  for each loss comparison and record effective sampled class/source frequencies.
- Use realistic crops and mild color/blur/compression; avoid crops or heavy erasing
  that remove all disease evidence while retaining the diseased label. Avoid
  transforming a labeled diseased crop into an apparently healthy training target.
- Maintain all eight supported classes and Other; Healthy-only fine-tuning risks
  forgetting. Cap oversampling of duplicate healthy leaves.

Run an ingestion/one-batch/overfit-small-clean-subset/export smoke sequence first.
Pilot plausible candidates for 3–5 epochs, promote a few to full runs, then rerun
the top two configurations with three fixed seeds if data/compute permit. Save
last/best checkpoints, optimizer/scheduler/EMA/RNG states, complete config,
data-manifest hash, environment lock, source revision and per-epoch metrics.
Resume interrupted runs safely. Default resource envelope: one provided GPU,
24 GPU-hours for the first complete search including seed checks; up to another
24 only if validation shows improvement and an unresolved justified experiment.
This is a planning cap for this handoff, not a measured duration. Keep a decision
log, avoid unattended infinite retries, and document reduced searches if needed.
Never silently switch to paid cloud compute.

## 9. Phase 4 — calibration, full-pipeline validation and acceptance

Freeze candidate weights before calibration. Fit positive temperature using a
constrained/log-parameterized optimizer on the calibration split. Select any Other
bias, optional Healthy bias or policy change on that split with explicit guardrail
constraints and the exact exported model/ensemble representation. Training loss,
temperature, bias, detector threshold and plant policy are separate choices.
Keep old policy as a comparator. A new temperature can alter plant verdicts even
when per-leaf top-1 is unchanged.

Required reporting at crop, photo and complete app-replay levels where applicable:

- Healthy recall: actual Healthy verdict / all eligible healthy reference plants
  or photos, including Unsure, Other and no-leaf failures in the denominator.
- False Healthy: Healthy verdict / all diseased references, with supported disease,
  unsupported tomato conditions and non-tomato inputs reported separately.
  Also report Healthy precision and conditional false-Healthy among decided scans.
- Supported-class recall/macro-F1, per-class confusion, wrong disease labels,
  false Other, no leaves, Unsure/insufficient evidence, decision coverage and
  evidence frames to verdict. Lower rejection achieved solely by more Unsure is
  not recognition improvement.
- Other rejection and false-known rate by unsupported subgroup; do not hide poor
  field behavior behind many easy lab Other photos.
- Calibration NLL, multiclass Brier and stated-bin ECE/reliability curves; label
  whether these refer to crops, photos or aggregated verdict probabilities.
- Detector recall/precision/IoU and selected-crop coverage, oracle-box versus
  detected-crop classification, lighting/blur/framing/source strata and paired
  error galleries. Preserve nonplant controls and mixed healthy/diseased checks.
- Paired baseline/candidate differences with 95% intervals, clustered by the
  highest available plant/leaf/session/duplicate group, not by augmented crop.
  State unknown grouping and small subgroup limits. Report per-source and an
  explicitly defined equal-source/class summary alongside any pooled score.

Predeclare acceptance on development data before opening the locked final test.
The following are **proposed engineering targets**, not owner-specified guarantees
or proof that available labels can support them. Record any justified revisions
before final test evaluation, never after seeing a disappointing test:

| Gate | Default candidate requirement |
| --- | --- |
| Field Healthy benefit | At least +10 percentage points absolute Healthy recall over baseline on a predeclared field-weighted cohort; report paired interval and counts. |
| Disease concealment | Point estimate false Healthy must not increase in each sufficiently sampled diseased/unsupported cohort; paired one-sided 95% upper bound on increase <=2 percentage points. |
| Other rejection | Unsupported false-known rate increase <=2 points within each sufficiently sampled subgroup. |
| Other supported classes | Macro recall drop <=2 points; no sufficiently sampled individual disease recall drop >5 points. |
| Deployment footprint | No >10% regression in model size, peak RAM or warmed median/p95 end-to-end latency on the same target device without a documented tradeoff; compare under identical settings. |
| Contract and regression | Hashes, I/O, preprocessing, exported logits/probabilities/decisions, golden fixtures and applicable software checks pass. |

Use paired baseline/candidate evaluation on identical examples. The current eight
PlantDoc Healthy photos cannot establish a reliable deployment gain or a tight
false-Healthy safety bound. If confidence bounds or sample coverage are inadequate,
mark statistical acceptance **inconclusive**, retain the production baseline, and
deliver the candidate artifact plus evidence in a draft PR. Do not relax a failed
gate silently. Run locked final evaluation once after selection; if it fails, keep
the result, do not tune against it, and require a new holdout for a later cycle.

## 10. Phase 5 — export and precisely scoped app changes

Use the existing LiteRT Torch NHWC wrapper as a starting point in a compatible
Linux conversion environment. Disable fused timm attention for export. Export the
selected checkpoint as one float32-input, nine-logit classifier at 224 initially.
Keep the detector unchanged unless its conditional training branch is selected.
ONNX is an optional intermediate check, not proof of correct Android conversion.
Historical code warns that onnx2tf misconverted ViT attention: use the verified
LiteRT Torch route and actual numerical/decision checks rather than assuming
conversion success means correct execution.

Test random normalized tensors and real canonical crops, including borderline
Healthy/Other cases. Compare PyTorch and TFLite full logits, calibrated probabilities
and verdicts. Initial FP32 numerical target: max absolute logit error <=1e-3 on the
parity set, with no unexplained label/decision changes; preserve the app golden
tolerances and explain tighter/looser representation-specific tolerances. Use more
than one random input. Quantized candidates need separate accuracy and parity
criteria. Re-evaluate the final TFLite bytes, not just the PyTorch checkpoint.

### Mandatory changes when a qualifying same-contract classifier is selected

1. Replace
   `agribot_android_app/android/app/src/main/assets/models/disease_classifier_0.tflite`
   through existing Git LFS; preserve label order, layout, I/O dtype and preprocessing.
2. Update runtime `model_manifest.json`: unique bundle ID, classifier hash,
   architecture/input metadata if changed, fitted calibration, genuine validation
   metadata and provenance/licence status. Do not copy old accuracy into new model
   claims. Mirror the chosen manifest in `model_training/v2/bundle/model_manifest.json`
   and update the calibration record in `model_training/v2/results/final/` with
   unambiguous selected-run provenance. Avoid overwriting historical evidence
   without versioned retention in the new report/run configuration.
3. Regenerate outputs for the same 13 golden PNGs against the chosen **exported**
   model and canonical preprocessing. Synchronize expectations in
   `model_training/v2/bundle/golden/golden_expected.json`,
   `agribot_android_app/android/ml/src/test/resources/golden/golden_expected.json`,
   and `agribot_android_app/android/app/src/androidTest/assets/golden/golden_expected.json`.
   Keep source-label truth separate from expected model output; do not change
   ground truth to make tests pass. Retain old baseline expectations in history.
   Additional fixtures should cover new healthy, disease, Other and crop edge cases.
4. Keep runtime, UI verdict text, labels, room data and evidence policy unchanged
   for the first same-contract model replacement. No Kotlin inference change is
   necessary for a compatible new weight file/calibration alone. Explain explicitly
   in the PR which files changed and why. Do not add an Internet permission or
   cloud inference dependency; the app is offline.

### Conditional changes: implement only if that experiment is selected

| Selected change | Exact app work required |
| --- | --- |
| Healthy or per-class bias | Add an optional backwards-compatible manifest field/vector; parse/validate length, finite values and default zeros in `VisionManifest.kt`; apply in `EnsembleDecision.kt` at the documented combine stage; pass via `AgribotLeafVision.kt`; update Python calibration/replay, tests and goldens. Current parser ignores arbitrary new JSON fields: changing JSON alone will not implement a Healthy bonus. |
| Verdict thresholds/evidence | Change `VerdictPolicy` in `PlantEvidence.kt`, or explicitly introduce versioned config with old defaults and wire it through **all** callers. Update Check/walk behavior and replay, partial-disease/stop/track tests. Manifest edits alone do not change these source defaults. |
| Resolution/backbone/member count | Update manifest I/O/member metadata, asset loading and actual interpreter tensor checks; verify `AgribotLeafVision` supports shapes; benchmark per-leaf and top-six workloads, memory, startup and both delegates/fallback. Preserve nine logits unless a separate contract migration is justified. |
| Detector weights or output | Replace `leaf_detector.tflite`, hash/manifest shape/coordinates/decoder only as necessary, rerun detection and classifier-crop evaluation and regenerate all detector goldens. |
| Preprocessing or crop pad | Synchronize `CanonicalImageOps.kt`, Python canon/export/evaluation/training evaluation transforms, manifest description and full-input golden checks. Treat as a distinct experiment, not a hidden export correction. |
| Quantized I/O or unsupported ops | Add scale/zero-point/dtype input/output handling and test interpreter/delegate compatibility; retain a working float candidate if unsupported. Never infer quantization support from file extension. |
| Labels/taxonomy | Separate migration across model head/order, manifest, `Conditions.kt`, all domain/export/history mappings, UI resources and tests. Avoid in this task unless required and fully evidenced; do not rename the legacy potassium key as incidental cleanup. |
| Distribution/version | Increment `app/build.gradle` versionCode/name only for an intended deliverable version following repository convention; a model PR need not invent release signing material. Build debug and release/unsigned release and label their signing state accurately. |

The existing model manifest contains a DRAFT notice requiring named human reviewer
sign-off and licence review before release. Preserve truthful status: autonomous
training and PR creation do not constitute that sign-off or authorize release.

## 11. Phase 6 — software, Android and hardware verification

Install test dependencies, run `python -m unittest discover -s tests -p "test_*.py"`,
and enable the optional field-media check with `AGRIBOT_FIELD_MEDIA_IMAGE` set to a
downloaded valid image. Check the environment with `python -m pip check`.
From `agribot_android_app/android`, run the repository CI-equivalent checks:

```bash
./gradlew :domain:test :camera:testDebugUnitTest :ml:testDebugUnitTest \
  :data:testDebugUnitTest :feature-scan:testDebugUnitTest :app:testDebugUnitTest \
  :app:lintRelease :app:assembleDebug :app:assembleRelease \
  --no-daemon --max-workers=1
```

Use `gradlew.bat` on Windows. Verify APK metadata/offline permissions as in CI.
Replay candidate outputs through the actual compiled Kotlin decision classes,
including all newly supported calibration/policy options; require zero unexplained
Python/Kotlin decision mismatches. Add meaningful tests for new parser/decision
behavior and crop/dtype boundaries. Goldens verify numerical implementation,
not agronomic correctness; keep separate labeled accuracy checks.

If an authorized ARM64 Android device is connected, run:

```bash
./gradlew :app:connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class=com.sakshyam.agribot.AgribotModelGoldenTest \
  --no-daemon --max-workers=1
```

The originating phone was Realme RMX5101, Android 16/API 36, package
`com.sakshyam.agribot`. The GPU host may not have it. CPU host timing is a proxy,
not target-device timing. If hardware is absent, build APKs and record device
checks/latency/thermal validation as NOT RUN; do not pretend an emulator validates
ARM64 delegate performance. Supply a concrete device-check checklist and leave
the rollout gate pending. Do not uninstall the owner's app to bypass signing:
its previous update needed a data-preserving reinstall because keys differed.
No phone installation is required for this retraining PR.

## 12. Phase 7 — artifacts, PR, and handback

Implement a reproducible entrypoint or ordered CLI commands with resume support;
do not leave crucial decisions only in transient shell history. Suggested new
work area: `model_training/v3/` with config, data preparation/split audit, training,
evaluation/calibration, export/bundle, orchestration and README. This is a proposed
layout to implement, not an existing runnable pipeline. Reuse correct v2 code.

Keep datasets, caches, raw checkpoints and lengthy logs outside ordinary git.
Commit code, lockfiles/configs, data/source manifests without private paths or
credentials, split/group audit summaries, compact evaluation tables, readable
report, and any qualifying Android assets via LFS. Use versioned downloadable
artifacts for checkpoints/candidate bundles when authenticated repository storage
supports them; record URLs, hashes, sizes and exact reconstruction commands.
Verify a fresh clone can retrieve every artifact required for the PR. If weights
cannot be published under source terms, document access/reproduction restrictions
instead of committing them. Do not commit downloaded third-party images or the
originating phone backups, signing keys, local `.toolchains`, or unrelated files.

The final report must include:

- Objective, base/branch revisions, old/new bundle and file hashes, checkpoint
  provenance, architecture, labels, preprocessing, training/calibration settings.
- Data source/licence inventory, image and independent-group counts by split and
  class/source, exclusions, label uncertainty, overlap audit and holdout status.
- Every attempted experiment with config/seed/time/peak VRAM, validation results,
  selection rationale, rejected alternatives and skipped conditional phases.
- Paired before/after metrics and intervals from the locked test and the historical
  diagnostic separately; Healthy gains, false Healthy and Other tradeoffs visible.
- Export parity, actual Kotlin replay, Python/JVM/lint/build/device checks with
  exact commands/results, and explicit NOT RUN checks. APK/artifact links/hashes,
  signing state, memory/latency evidence and unresolved acceptance gates.
- File-by-file model/app change explanation, rollback instructions to the old
  bundle, reproducible rerun instructions and the justified rollout decision.

Commit only intended files, push the training branch, and open the PR yourself:

```bash
git push -u origin training/healthy-field-<date>
gh pr create --base main --head training/healthy-field-<date> \
  --title "Improve field Healthy recognition with validated retraining" \
  --body-file <prepared-pr-description.md>
```

Use the actual branch name and an accurate title reflecting the final result.
Use `--draft` when key evidence/acceptance/device gates remain unresolved, even
if training finished. Attach CI-accessible evidence, watch checks with
`gh pr checks --watch`, and repair failures attributable to the change. Do not
merge the PR. If gh is unavailable, install it project-locally or use the
authenticated GitHub API; if credentials truly block creation, retain the pushed
branch if possible and provide the compare URL and exact authentication blocker.
Never claim a PR exists without its returned URL.

Final handback to the owner: PR URL, selected candidate or reason to retain the
baseline, measured Healthy/disease/Other changes, test status, artifact locations,
and any remaining rollout gate. Write enough into the PR that a separate reviewer
can assess the change without this conversation.
