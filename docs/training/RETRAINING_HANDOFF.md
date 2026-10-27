# Agribot training failure investigation, recovery, and Android integration

**Normal app 2.0.4 is built:** [delivery and verification](NORMAL_APP_DELIVERY_20261027.md).
The original release-baseline model is retained; no failed replacement is bundled.

**Latest offline evidence:** [corrected cache joins and bounded supervised/reward
controls](OFFLINE_RECOGNITION_20261027.md). Phase 89/91 are explicitly invalidated;
their corrected successors and later CNN/reward controls still fail recognition.
No failed replacement model may enter the normal app bundle.

**Latest outcome:** [27 October acceptance failure and continuation](FIELD_ACCEPTANCE_20261027.md).
The app passes 133 native unit and 22 emulator checks. The frozen recognition
candidate fails fresh OLID Healthy recovery at 87/223, with no improvement.
Both AD and OLID are now consumed development. PR #6 stays draft; no phone
promotion is authorized by these results. Earlier “OLID sealed” notes are history.

**Earlier development pass:** [historical development evidence](QUALIFIED_DEVELOPMENT_20261027.md).
Phase 69 passes every unchanged development gate. Restored and actual LiteRT
verification both preserve all 6,742 verdicts; shared model bytes are frozen.
The later Android port passed verification; OLID acceptance then failed as
described above. The model remains unqualified for promotion.

**27 October takes precedence:** read the
[app and recognition checkpoint](APP_RECOGNITION_CHECKPOINT_20261027.md).
Copied DINO-tail adaptation meets Healthy minima but fails unsupported retention;
whole-image confirmation restores safety but fails Healthy minima. Both are
stopped. The large-cache hash OOM is repaired and cached prefixes audited.
Native app repairs pass final local builds, lint and 20 emulator checks. AD is consumed, OLID is
sealed, the phone is disconnected, and model export/integration remain deferred.

**Latest checkpoint:** [PR 6 code review and bounded continuation](PR6_CODE_REVIEW_20261026.md). Useful cache, training and native safeguards are retained; image/cache integrity findings are repaired, and both CI jobs passed at `399418f`. The subsequent cached/spatial/routing and symptom-region controls have completed and stopped without qualifying recognition. Read the final section for measured results, unchanged gates and the four-hour PR review schedule. AD is consumed development, OLID stays sealed, and export/Android integration remain deferred. The disconnected phone and installed model are unchanged. Historical statements below about AD being unopened are superseded.

Revision: 2026-10-26. Original handoff: 2026-10-15.
Repository: https://github.com/saksham-loonker/agribot.
Audience: the originating assistant returning later over SSH to a PC with an
NVIDIA RTX 5090, or an implementation agent continuing the same investigation.

## Current continuation - 25 October 2026

**26 October takes precedence:** read
[SIGLIP2_CONTINUATION_20261026.md](SIGLIP2_CONTINUATION_20261026.md).
All MobileCLIP-dependent app candidates are blocked by the full Apple research
license. The human-only phase-twelve data is now used for an independent
Apache-2.0 SigLIP2 encoder; no restricted teacher parameters or logits are used.
The corrected MobileCLIP union app replay reaches only 31/53 Healthy on new
internal families, despite passing the older development cohorts. It remains a
research result. The consumed 246 and all related families remain retired from
benchmarking. The independent AD source is still unopened. The app is not fixed.

**Newest work:** read [the MobileCLIP/data-revision continuation](MOBILECLIP_CONTINUATION_20261025.md).
There are 285 valid completed controls before the next revision; eight invalid
new-experiment fits are quarantined and excluded. A five-expert image prototype
passes every development gate (94/129 field Healthy, 112/119 paper Healthy) and
all 6,468 exact-graph verdict checks, but reaches only 51/95 on the consumed
field cohort. It is not qualified. Phase twelve now retires that cohort and its
787 related photos into grouped training/development; the old 246 must never be
used as a benchmark for phase twelve. Original eight regression cohorts and the
untouched independent AD source remain gates. New-data controls are running.
Both APK variants rebuilt successfully with the unchanged interim model; no
new model was installed. Python: 122 run, 121 passed, one skipped. Native CI
passed `ed72afd`. Older reservation-exclusion statements describe earlier revisions.

**Read the newest [leaf-scope continuation](LEAF_SCOPE_CONTINUATION_20261025.md)
before the chronological notes below.** FieldPlant download and role/family audits
are complete. There are 269 completed controls; the first copied-tail leaf control
fails recall and a separate augmented control is running through
`phase10_leaf_pipeline.py`. Weak TOM fitting
targets are masked; regression labels are retained. The cold Abu Dhabi source
remains unopened diagnostically. Leaf CNN/genus and four original-feature heads
still fail end-to-end recall. An opt-in schema-four native/host contract protects
the frozen reference's aggregate named disease verdict on the same observations.
Its native unit tests, release lint and test-APK build pass; the Python suite ran
121 tests (120 passed, one skipped). Phone remains on interim 2.0.2 and was absent
from ADB at the latest check. Do not confuse a completed experiment or successful
build with a qualified recognition improvement.

**Newer continuation:** 262 trained controls are complete. Stronger supervised
four-block and twelve-block Healthy branches reach 51/95 and 55/95 on the opened
field regression, but false Healthy rises from 3 to 4; neither qualifies. A
full-depth mixup control also fails development recall. The original disease
branch remains frozen. Local Python tests: 119 passed, one skipped; Native CI
passed `8f6fe18`. Phone remains 2.0.2.

The next work is data scope rather than another blind loss sweep. FieldPlant's
1,403-photo/2,786-leaf tomato subset is being acquired by
`phase10_fieldplant.py`; check `phase10_fieldplant.log` and
`phase10_fieldplant/download_complete.json`. Confirm actual whole-leaf box roles
and audit overlaps/families before fitting. Broad blight/spot labels must not
be invented named classes. The separate 502-image Abu Dhabi source is downloaded
and family-audited but remains **unopened for diagnostic inference and fitting**;
its 104 Healthy images span only 20 inferred families. Preserve that source for
a future frozen evaluation. Read the campaign record's latest continuation and
machine evidence before acting; older pending-job/reservation statements below
are chronological history.

**Latest decision:** 246 trained controls completed. The focal image-context
candidate passed development gates, but failed the one-time prospective
reservation: Healthy 44/95 versus interim 8/95, below the frozen 70% target.
False Healthy remained 3; per-source disease/unsupported retention passed. It is
rejected for deployment. The reservation is now opened regression data, including
related capture families; do not reuse it as fresh acceptance after further
tuning. The next experiment is broader supervised image-context backbone
adaptation with the original nine-class branch frozen; later promotion needs
new independent field evidence. Phase-seven controllers are complete. Phone
2.0.2 remains installed. This decision supersedes the older progress notes below.

Read [the current campaign record](RECOGNITION_CAMPAIGN_20261024.md) first.
App 2.0.2 is installed and its phone parity checks pass, but field generalization
is unresolved. Earlier adapted candidates failed SLIF and fresh TOM2024/paper-leaf
acceptance. They were not installed. Those cohorts are now opened development
data; do not call them independent again. See the campaign record's phase-five
continuation for the current supervised training and untouched reservation.

The attempted generic detector trial was stopped and quarantined. Its SLIF boxes
describe disease/healthy regions, not whole-leaf boundaries, so the apparent
0.61176 versus 0.01238 mAP gap compares different annotation targets and is invalid
evidence of a detector failure. Do not resume those detector checkpoints. Separate
logit-adjustment, expected-reward-plus-CE and PPO controls completed and failed
retention when used alone. The full-ROI confirmation prototype improves their
safety, but its fresh acceptance improvement was insufficient for promotion.
Broader supervised adaptation now uses correctly scoped TOM2024 and paper-leaf
classification data, with 246 previously unopened photos reserved before fitting.

SSH alias: `agribot-5090`. Remote repository: `/home/genesis/agribot`.
Python: `/home/genesis/.venvs/agribot-train/bin/python`. Artifact root:
`/home/genesis/agribot-data/recognition_campaign_20261024`.
Inspect `tmux list-windows -t agribot` and the campaign record's current checkpoint.
The teacher audit, detector control, ROI adaptation, separate species/Healthy
CNNs, five frozen-feature heads and monotone logit fusion have completed.
There were 238 completed trained controls before the two current scene-negative
continuations; policy threshold grids are not additional training runs.
The best relative-safe ensemble reaches TOM2024 Healthy 87/129 (67.44%), below
the 70% development target. A 97/129 alternative adds one unsupported-condition
false Healthy call and is rejected under the unchanged gates. No new candidate
has been installed. Do not infer that a more complex loss fixed recognition.

Current controller: `phase7_negative_controls.py` under the artifact root;
tmux window `scene-negatives`; log `phase7_negative_controls.log`. It trains
16-epoch CE-negative-cost-8 and focal-gamma-2-negative-cost-4 image-level CNN
continuations, then fits calibration-only logit fusion and replays opened app
cohorts. Research outputs are in `phase6_roi/scene_*_selection/` and
`phase6_roi/fusion_selection/`. Check completion files before restarting.
All inference remains development-only. Whole-image image-label evidence is
not a verified disease-free-plant diagnosis. Native schema-three full-ROI
confirmation is implemented, including optional Unknown confirmation; the newer
whole-frame ensemble has no native runtime/export implementation yet.
`phase5/reserved_manifest.json` has not been inferred yet. The work is saved in
draft PR #6, whose initial Android Native CI passed.
`phase3_pipeline.sh` is stopped; its dataset is quarantined and new guards reject
resumption/export. Full app replay and independent field acceptance remain
mandatory before installation. The stable original comparator is under
`original_reference/`; repository Android assets now contain the 2.0.2 interim
model and must not be mistaken for the original classifier.

Local authoring checkout: `.toolchains/recognition-work` on branch
`research/recognition-campaign-20261024`. Cached native build checkout:
`.toolchains/production-work`. The phone has the maintained package
`com.sakshyam.agribot`; the obsolete `.field` package is disabled with data kept.
Never clear either package's user data to install a candidate.

## Execution update - 24 October 2026

The subsequent exhaustive supervised, logit, reward, CNN, tree and isolated DINO
adapter campaign is documented in
[RECOGNITION_CAMPAIGN_20261024.md](RECOGNITION_CAMPAIGN_20261024.md). Read its
candidate decisions and acceptance accounting before reusing any newly opened
cohort. Earlier rejected candidates below remain historical context.

Further supervised Healthy/retention experiments and a shared-backbone adapter
were completed after the initial recovery. Broader grouped regression exposed
three new disease-to-Healthy errors, so the adapter was rejected and the original
classifier retained in app 2.0.1. Read
[APP_VALIDATION_20261024.md](APP_VALIDATION_20261024.md) and its linked results
before any further training. The former locked test has now been opened and
must only be called regression data. Collect a fresh, adjudicated field set for
the next acceptance decision; do not tune another confidence threshold on these
same errors and describe it as independent validation.

Also make colour decoding explicit. Some downloaded JPEGs carry an OPPO/Apple
ICC profile. PIL's raw RGB conversion and Android BitmapFactory's colour-managed
decode can produce different pixels even with identical normalization code.
Phone parity fixtures must preserve the Python reference's RGB byte values in
metadata-free PNGs; retain source identity separately. Test colour-managed photo
import independently if that feature is added. The current camera pipeline does
not decode these source JPEGs.

The PC was inspected and the original trainable `field_vits2` checkpoint was
recovered and verified against all 13 shipped golden fixtures. The v3 trainer
was repaired, genuine head-only fine-tuning completed, and full development app
replay performed. The candidate did not improve app Healthy recall, so production
assets remain unchanged. Read the [recovery report](../../model_training/v3/RECOVERY_20261024.md)
and its compact results before continuing; the unknown checkpoint/remote-state
entries below describe the pre-investigation baseline, not current findings.

## Read first: the October 24 recovery mandate

The owner reports that subsequent retraining achieves only approximately 30%
accuracy, and that restoring Healthy recognition causes recognition of diseased
classes to deteriorate. They suspect a code bottleneck, possibly inconsistent
0–1 versus 0–255 preprocessing, and ask whether logit-based RL is the next step.
These are **reported symptoms, not reproduced measurements**. The remote training
repository, commands, configuration, checkpoints, datasets, environment and logs
have not yet been inspected by this assistant. The meaning and denominator of
the reported 30% are unknown. Do not substitute the October 15 evaluation for
evidence about that failed run.

The next execution is an investigation and repair before another optimization
campaign. First establish that images, labels, pretrained weights, gradients,
metrics, saved checkpoints and exported runtime all mean the same thing. Then
recover a joint Healthy-and-disease supervised baseline. Consider more elaborate
losses only after that baseline works. A larger reward experiment cannot repair
an image-scale mismatch, permuted labels or an incorrectly restored checkpoint.

This revision adds the recovery protocol below and retains the original full
dataset/training/export/app/PR specification later in this file. **The October 24
protocol takes precedence wherever the original search plan would launch long
training or weighted/logit experiments before correctness has been established.**
Original deployment acceptance and offline Android contracts remain in force.

This editing task does not initiate SSH, remote package installation or training.
When the owner later instructs this assistant to connect, use their supplied
host/user/authentication information and the existing authorized task scope.
Do not invent connection details, claim remote access, or treat an absent PC
artifact as proof that it does not exist on the PC.

### A. Evidence status and the first decision

| Statement | Status as of this revision | Consequence |
| --- | --- | --- |
| Retraining is around 30% accurate | Owner report; metric, split and runtime unknown | Recover raw outputs and independently recompute the metric. |
| Healthy recovery coincides with diseased-class collapse | Owner report; intervention and timing unknown | Reconstruct the exact before/after checkpoint, config and per-class transition. |
| A 0–1/0–255 normalization error is responsible | Hypothesis, not established | Trace and compare actual tensors; do not apply an extra division by 255 blindly. |
| The checked-in field trainer uses 255-scaled means/stds | Verified source observation | This expression is valid for raw float pixels in 0–255. |
| The Android classifier uses /255 then ImageNet normalization | Verified source observation | Equivalent normalization algebra; resize/crop and upstream data still need checking. |
| The old field trainer already adjusts logits in supervised CE | Verified source observation | Establish plain-CE control before attributing a benefit to a new logit objective. |
| The failed remote run used exactly this trainer/revision | Unknown | Diff the PC checkout and preserve local modifications before editing. |
| The shipped Android model was globally about 30% accurate | Not established by the old diagnostic | Keep source-specific historical results separate from new runs. |
| A trainable `field_vits2` checkpoint is unavailable everywhere | Not established | It is absent from this checkout; search documented remote artifact locations. |

Decision at entry: **diagnose first; no PPO/GRPO/reward-model run and no blanket
Healthy bonus.** Do not assume Healthy should become the majority prediction.
The desired output is correct diagnosis with honest uncertainty and preserved
recognition of diseased leaves.

**Owner clarification: preserve the existing strong classifier; full retraining
is not the default.** First recover and verify its matching trainable checkpoint
on the PC. Establish frozen-model results, test calibration alone, and then
fine-tune the existing head conservatively, unfreezing the last backbone blocks
only if justified. Retain representative supported diseases and Other examples
alongside reviewed Healthy failures. Preserve the existing trained head when
compatible; do not reset it merely because a head warm-up recipe exists below.
Use the verified original model as an optional retention teacher on reliable
training examples, while allowing correction of its known mistakes.

The preferred order is: verified existing weights -> calibration-only comparator
-> targeted head fine-tuning -> limited backbone fine-tuning. A new pretrained
backbone and broader retraining are fallbacks if matching trainable weights
cannot be recovered or controlled evidence shows targeted adaptation inadequate.
Explain the fallback trigger before spending its compute budget. Supported
conditions incorrectly rejected as Other should improve; genuinely unsupported
conditions must retain rejection. Do not optimize away the Other label itself.

### B. Remote session entry, preservation and durable progress

Before changing the PC, establish a durable `incident_<UTC timestamp>` directory
outside tracked model assets. Use the actual remote OS/shell; commands below are
examples for Linux and must not be pasted into Windows PowerShell unchanged.
Record paths, invocation, timestamps and exit codes without tokens or secrets.

1. Verify the intended SSH host identity using the owner's supplied/trusted host
   information. Use existing authorized SSH credentials. Do not disable host-key
   verification as a shortcut. If access details are missing, request only those
   details when remote execution is actually requested.
2. Record hostname, OS, current user, workspace path, free disk/RAM, GPU/driver,
   GPU processes, virtualenv/conda/container identity and installed torch/timm/
   torchvision/CUDA/export versions. Check other users' jobs before using the GPU.
3. Record `git status --short`, branch, HEAD, remotes, tracked diff and names of
   relevant untracked training files. Save local diffs before any synchronization.
   Inspect new AGENTS.md instructions. A clean separate worktree is preferred;
   never reset the failed training workspace to make it look reproducible.
4. Copy or hash existing configs, manifests, result tables, selected logs, best/
   last/EMA checkpoints and code used by the failed run. Keep originals immutable.
   Do not move/delete multi-GB checkpoints or caches until their provenance and
   usefulness are understood. Record symlink targets and actual file SHA-256s.
5. Inventory active training commands from known job supervisors/process records.
   Do not kill an unrelated GPU job. If replacing a run is necessary, stop only
   the identified task after saving its checkpoint/logs and checking ownership.
6. Use tmux or an equivalent existing job supervisor for long authorized jobs;
   tee logs to the incident/run directory and persist PID, command, run ID and
   resume location. An SSH disconnect must not silently erase the experiment.
7. Maintain `STATUS.md` with current stage, completed checks, observed failure,
   next command, active process, latest checkpoint and remaining gates. Update it
   at each stage so a subsequent assistant can continue without chat history.

Example read-only Linux inventory, executed only after connection is authorized:

```bash
pwd
hostname
uname -a
git status --short
git rev-parse HEAD
git remote -v
nvidia-smi
df -h .
python -V
python -m pip show torch torchvision timm numpy
```

Avoid dumping environment variables, SSH private keys or credential-bearing URLs
into the report. Install fixes in an isolated environment after collecting the
old environment; do not overwrite the only known environment before comparison.

### C. Reconstruct what “30%” and “Healthy resumes” mean

Recover these answers from artifacts before asking the owner to reconstruct
them from memory:

- What exact command, code revision, config, seed, input size, architecture and
  initialization checkpoint produced the run? Was it actual fine-tuning, a fresh
  pretrained backbone, or a randomly initialized model mislabeled as pretrained?
- Is 30% train accuracy, train minibatch accuracy with augmentation, validation,
  test, macro recall, binary health classification, crop accuracy, whole-photo
  accuracy, TFLite accuracy or the Android final verdict? Was class Other counted?
- What class/source/group counts make up its denominator? What are the majority
  class baseline, uniform nine-class baseline (11.1%), empirical class-prior
  prediction baseline, and confusion matrix? A 30% score alone proves neither
  underfitting nor a normalization defect.
- Does “Healthy resumes” mean an epoch, a resumed checkpoint, a new Healthy-only
  training stage, different sampling/weights, a threshold/bias, an ensemble member,
  or restoring the original app model? Establish the observed event; do not choose
  one interpretation and build a fix around it without evidence.
- Did disease recognition fall in the raw PyTorch logits, after calibration,
  after TFLite export, or only after app aggregation? Were examples actually the
  same before/after? Report sample IDs and checkpoint hashes, not screenshots alone.
- Were eight-class and nine-class models compared using the same label order?
  Was Healthy moved to index 0 by ImageFolder/binary labels? Was Other added in
  the middle, inferred from sorted folders or treated as a ninth disease?
- Did the evaluation silently exclude rejected/unsure/no-leaf cases, fail to
  restore eval mode, reuse stale predictions or choose a different best checkpoint?

Deliver `failure_reproduction.json` with the recovered facts, missing evidence,
an independently computed metric and representative failures. Repeat the failed
evaluation on frozen weights before reproducing hours of failed training. Use
development data for diagnosis; do not repeatedly probe a locked final test.

### D. Numerical truth: exact image-range contracts

The existing classifier expects standardized RGB pixels, not plain 0–1 values
and not plain 0–255 values at its neural-network input. For raw pixel x:

```text
mu    = [0.485, 0.456, 0.406]
sigma = [0.229, 0.224, 0.225]

Valid form A: (x / 255 - mu) / sigma
Valid form B: (x - 255 * mu) / (255 * sigma)

The forms are mathematically equivalent when x means the same RGB pixels.
```

A local float32 sweep of all integer channel values 0..255 on 2026-10-24 found
maximum absolute difference `2.384185791015625e-7` between these forms. This is
an algebra check, not a remote-model validation or a full preprocessing parity
test. It does not rule out a wrong remote cache, crop or double normalization.

Correct standardized endpoint values for the three RGB channels are approximately:

| Input channel value | R | G | B |
| --- | ---: | ---: | ---: |
| 0 | -2.117904 | -2.035714 | -1.804444 |
| 255 | 2.248908 | 2.428571 | 2.640000 |

For synthetic pixels set to `255*mu`, standardized channels should be approximately
zero. Real float resize values may be nonintegers. After mean/std normalization,
values below zero and above one are expected; a generic “must be 0–1” assertion
at the network input would be wrong. Bounds above apply to valid unaugmented
0–255 RGB input, not arbitrary augmented tensors or random erasing noise.

The leaf **detector** has a different contract: RGB letterbox pad 114, division
by 255, no ImageNet mean/std. Its pad becomes about 0.44706. Do not share a final
classifier normalization transform with the detector just because both take NHWC.

Common faults to deliberately detect with negative controls:

| Fault | Observable effect / distinguishing check |
| --- | --- |
| Divide by 255 twice | White and black both normalize near negative mean/std; image contrast nearly disappears. |
| Normalize 0–255 using unscaled 0–1 mean/std | Channel magnitudes can exceed 1,000. |
| Normalize already standardized tensors again | Invalid second transform; hook immediately before forward reveals it. |
| Normalize float 0–1 with 255-scaled constants | Nearly constant dark-image equivalent input. |
| Cast float 0–1 cache to uint8 without multiplying first | Most image values become zero; check cache histogram and decode lineage. |
| Let PIL/loader convert through a wrong color/alpha/range convention | RGB patches or input pixel hashes differ before normalization. |
| Pass NHWC to timm or NCHW to the exported wrapper | Wrong axes/strides; square shapes can make some mistakes less obvious. |
| Treat `channels_last` memory format as an NHWC shape conversion | PyTorch shape is still NCHW; only memory layout changed. |
| Normalize in the export wrapper and also outside it | PyTorch/export graph contains an unintended duplicate transform. |

One important local check: `torchvision.transforms.v2.ToDtype(float32, scale=True)`
converted a uint8 tensor of 255 to 1.0, but left an already-float32 tensor of 255
at 255.0. **Do not assume the word `scale=True` repairs an earlier `.float()`**.
Audit actual dtype and transform dispatch in the installed remote version.
The checked-in `train_clf.py` starts from PIL/ToImage, so this observation does
not itself establish a bug in that script; it is a concrete risk for a modified
cache/DataLoader path. Save remote version and reproduction output.

Reproduce the algebra/sentinel check independently on the PC before testing its
actual DataLoader. This snippet does not load a model or change training files:

```python
import numpy as np

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
pixels = np.arange(256, dtype=np.float32)[:, None]
unit_form = (pixels / 255.0 - mean) / std
byte_form = (pixels - 255.0 * mean) / (255.0 * std)
np.testing.assert_allclose(unit_form, byte_form, rtol=0, atol=1e-6)
print("formula_max_abs_error", np.abs(unit_form - byte_form).max())
print("black_rgb", unit_form[0])
print("white_rgb", unit_form[-1])

# Negative controls: these MUST disagree with the valid normalization.
double_scaled = (pixels / 255.0 / 255.0 - mean) / std
not_scaled = (pixels - mean) / std
assert np.abs(double_scaled[-1] - unit_form[-1]).max() > 1
assert np.abs(not_scaled[-1] - unit_form[-1]).max() > 100
```

For production transforms use an explicit named input contract such as
`uint8_rgb_0_255` or `float_rgb_0_1`; do not infer it from each batch's maximum.
A valid dark image and an incorrectly scaled image can have overlapping numeric
ranges. Validate dtype/range/lineage together and reject incompatible inputs.

Primary references for these contracts:
[torchvision Normalize](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Normalize.html),
[torchvision ToDtype](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.ToDtype.html),
[PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html),
[checkpoint saving/loading](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).
For integer class targets, CE consumes logits; do not add an unexplained softmax
before CE. Custom soft-label losses need their own documented contract.

### E. Source observations to investigate, not asserted root causes

These observations refer to the local source at baseline commit `e67f2af` plus
this documentation edit. Reconfirm paths/lines against the PC's actual revision.

| Local code | Verified behavior | Investigation / required correction if relevant |
| --- | --- | --- |
| `train_field.py:35,57,64` | 255-scaled mean/std; training/eval assume raw 0–255 cached pixels | Verify cache dtype/range and every upstream conversion; no automatic /255 patch. |
| `final/canon.py:classifier_input`, Android `CanonicalImageOps.kt:classifierInputInto` | Crop/resize then /255 and unscaled ImageNet mean/std | Compare identical decoded pixels/crops through full paths; normalization algebra alone is insufficient. |
| `build_cache.py:base`, `train_field.py:gpu_eval` | Bicubic 448-base image, then bilinear antialiased resize | Different from direct canonical area resizing; quantify tensor and prediction differences. This difference exists, but its contribution to 30% is unmeasured. |
| `train_field.py:68–74` | Defaults to MobileNet; initializes head to zero, loads backbone with strict=False, copies old head rows by position | Check actual architecture/weights/class-name row mapping; fail on unapproved missing/unexpected keys. |
| `train_field.py:31–33,105` | Domain sampling and effective-prior logit adjustment plus smoothing | Compare ordinary CE with natural sampling first; log sampled priors and coefficient choices. |
| `train_field.py:teacher/KD branch` | Teacher can have eight classes; student first k logits used for KD | Assert name-by-name class alignment, input contract, teacher correctness and loss contribution; disable initially. |
| `train_field.py:gpu_aug` | Aggressive random crop, flips/rotation/color/blur/erasing | Inspect disease symptom retention; disable for the tiny-set and clean reference controls. |
| `train_field.py:96,117` | Pools clean/leaky validation and selects weighted source accuracy | Exclude known leakage; add per-class disease/Healthy gates so a coarse average cannot conceal collapse. |
| `train_field.py:114,119` | Samples loss every 16 iterations and rescales an approximation | Replace with sample-weighted full epoch loss for diagnostics; do not interpret that approximate logged number as exact loss. |
| `train_field.py:118–120` | Saves best EMA state_dict; metadata written only at end; no complete resume state | Verify evaluated state is selected EMA; persist config from start and full resume state separately. |
| Historical train/eval scripts at module top level | Argument parsing, cache reads, CUDA allocation and loops execute on import | Extract testable functions and add main guards in new tooling; do not import these scripts casually from a CPU parity test. |
| `build_cache.py:group` | Some group values such as `va_full`, `tw_<class>`, `pv_<class>` are source/type/class tags | They are not physical-leaf IDs; recover real lineage before using “grouped split” claims. |
| `eval_field.py`, `ens_eval.py`, `final/stage_a.py` | Different historical logit/energy/calibration conventions and pooled validation | Label representation explicitly; evaluate the actual current app combine/policy. |
| `final/convert_litert.py` | NHWC wrapper permutes to NCHW; random input is 0–1 | Add representative standardized inputs and full output parity; random 0–1 alone is not coverage of deployed inputs. |
| `AgribotLeafVision.kt:classifyWhole` comment | Mentions fallback; current analyzed-leaf flow has no whole-image fallback | Trace callers; do not treat a stale comment as app behavior. |

Zero-initializing a new linear head is not proof that the whole pretrained model
is random. It can make first-step backbone gradients zero until head weights
move. A fresh head also has no reason to match old classifier accuracy at step
zero. Distinguish legitimate initialization behavior from accidental head reset
while resuming a trained checkpoint. Correct shapes do not prove correct classes.

### F. Build a small diagnostic packet before long GPU work

Implement tools under a new diagnostic/training directory, preserving historical
v2 behavior for reproducibility. The following filenames are deliverables to
create during remote execution, not files already implemented by this edit:

```text
incident_<timestamp>/
  STATUS.md
  inventory.json
  failure_reproduction.json
  source_revision_and_diff/
  dataset_manifest.jsonl
  diagnostic_sample_manifest.jsonl
  preprocessing_contract.json
  tensor_audit.jsonl
  normalization_probes.json
  label_audit.json
  checkpoint_audit.json
  sampler_audit.json
  gradients_and_updates.jsonl
  probe_results.csv
  per_class_metrics.csv
  prediction_transitions.csv
  failure_gallery.html
  decision_log.md
```

Build a fixed, source-stratified development packet with at least several
verified examples of every available class, Healthy/disease lookalikes, supported
and unsupported tomato leaves, non-tomato controls and crop edge cases. Aim for
roughly 10–20 independent groups per class if accessible; record actual coverage
and absent classes. Use the historical goldens as implementation controls, never
as an independent accuracy benchmark. Keep diagnosis and final-test sets separate.

Each diagnostic record must retain source/original ID, image hash, ground-truth
label and review status, class index, crop box/padding, dimensions, orientation,
decode backend, dtype/range, preprocessing version and split/group lineage.
Capture full tensors only for the small packet; use summary statistics/hashes
for bulk data so the audit does not fill disk or expose private images.

### G. Trace one identical image through every boundary

Capture arrays before and after: decode -> cache write -> cache read -> crop ->
resize -> augmentation -> conversion to float -> normalization -> layout wrapper
-> model input -> raw logits -> calibrated probabilities -> app plant verdict.
For the deterministic parity run disable all random augmentation. Never compare
randomly different crops and interpret their tensor difference as a scaling bug.

For every boundary log shape, dtype, strides/layout, channel order, min/max,
mean/std per channel, selected quantiles, fraction zero/saturated, finite status,
pixel/tensor hash, selected sentinel pixel values and class ID. Install a model
forward pre-hook for the diagnostic run so the final captured input is what the
network actually received, not what a helper function intended to produce.
Inspect model modules/export graph for embedded rescaling or normalization.

Compare the following paths using the same predecoded RGB array and crop:

1. Direct Python canonical preprocessing used for the shipped TFLite diagnostic.
2. Actual training evaluation transform and training transform with randomness off.
3. Cache-based path versus direct-from-original path, including the 448-base stage.
4. PyTorch NCHW input versus exported NHWC wrapper input after the documented permute.
5. Android golden/static input path when hardware/build tooling is available.

For strictly equivalent FP32 normalization on the same already-resized pixels,
start with `atol=1e-6` and inspect exact worst-case values; tolerate explained
rounding, not a 255x discrepancy. Canonical input golden tolerance remains the
declared contract. Interpolation/decode differences need separate comparisons,
not an unjustified expectation of byte identity. Float64 and float32 paths must
be labeled. Save tensor MAE/max error, correlations, logit errors and changed
predictions; quantify whether the actual discrepancy explains the accuracy drop.

Use synthetic black/white/gray, red/green/blue patches, a channel ramp, checkerboard,
portrait/landscape crops and odd-sized boxes to test range, RGB/BGR, axes and
rounding. For color metadata/EXIF, compare decoded pixels separately from the
normalization transform; do not silently apply a different orientation on only
one path. Detect double preprocessing with intentional invalid transforms in a
diagnostic control, not by deploying them as automatic fallback hypotheses.

Gate G passes when every boundary has a documented input contract, identical
operations agree within stated tolerance, and real differences have been isolated
and measured. Merely logging overall tensor min/max is insufficient.

### H. Labels, data order, sampling and metric integrity

Make one explicit class-name-to-index manifest authoritative across all loaders,
checkpoints, teachers, exports and Android. The nine deployed keys/order later
in this document remain the default. Persist each loader's `class_to_idx` and
assert equality rather than relying on folder sort. Remap teacher/checkpoint head
rows by exact canonical class name; require an explicit reviewed alias map for
spelling differences. Never infer semantic alignment from head dimensions alone.

Verify original photo/annotation, crop, sample ID, label tensor and final prediction
remain paired after shuffling, multiprocessing, cache indexing, filtering,
concatenation and sampler use. Use an immutable original index and batch lineage.
Check cache metadata length, `n`, initialized rows, dtype, shape, file hash and
cache producer revision/config; reject partial caches rather than learning from
zero-filled trailing rows. Do not fix the schema by sorting images without
sorting/rebuilding their labels identically.

Audit raw/effective source and class counts, missing classes, mislabeled species,
unsupported-to-Other mappings and mixed scenes. Review the largest errors and
high-loss examples without automatically relabeling them. Keep corrupt or
uncertain data in an exclusion ledger, not silently dropped from reported totals.

For samplers, record actual drawn sample IDs and class/source histograms for at
least one full epoch, unique groups visited, repeat counts and probability mass
by subgroup. Compute the effective prior from actual selection probabilities.
Check all probabilities/weights are finite/nonnegative, sum mass is nonzero,
and all intended classes receive training draws. Compute optimizer steps per
epoch; `per_epoch` is a sampling budget, not necessarily one complete dataset pass.
Protect rare disease classes from disappearing while adding healthy samples.

For metrics, save logits/labels/sample IDs together. Independently recompute
top-1, per-class recall/precision, macro scores and full confusion with the fixed
nine-class axis order. Undefined metrics for absent classes must be explicit,
not replaced with 100% or silently omitted to inflate a macro average. Separate
model argmax from app decided/unsure/Other/no-leaf outcomes. Plot predicted class
frequencies and Healthy false positives by disease/source as well as accuracy.

### I. Checkpoint, resume and optimizer audit

Before training, save architecture and input resolution, pretrained source/hash,
head dimensions/order, parameter count, selected checkpoint hash and load report.
Use strict loading when resuming the same model. For intentional head replacement,
whitelist only that change; abort on unexpected missing backbone keys. Account
explicitly for wrapper prefixes (`module.`, `_orig_mod.`), checkpoint dictionaries
versus bare state_dicts and EMA versus non-EMA weights. Load trusted checkpoints
only; do not disable serialization safeguards merely to bypass an unknown file.

Evaluate the checkpoint immediately after loading and before optimizer creation.
Then save/reload in a new process and compare deterministic logits and the
diagnostic confusion matrix. For a recovered original classifier, agreement with
its known baseline on identical preprocessing is a prerequisite to calling it
“resumed.” For a new pretrained backbone with a random head, report that distinct
starting state without expecting nine-class classification competence yet.

Log `requires_grad`, model train/eval mode, dropout/drop-path and BatchNorm behavior
where applicable, optimizer groups and actual LRs, weight decay, scheduler step,
global step, batch/accumulation size and EMA state. Add trainable parameters to
the optimizer when unfreezing; changing requires_grad alone does not repair an
optimizer built with a filtered frozen parameter list. If a head is replaced,
do not retain stale optimizer tensors for old head rows.

On the first several batches record raw loss components, per-group gradient norm,
clipped norm and parameter update norm. Assert gradients/losses/updates finite;
verify trainable weights actually change and frozen weights do not. Test eager
FP32 first, then BF16 and compilation as separate speed changes. If the loaded
model is correct but the first few updates destroy disease recall, inspect LR,
sampler composition, head mapping, erroneous resume state and loss coefficients
before running another 40 epochs. Compare EMA and online predictions separately.

Full resume state must include online/EMA weights, optimizer, scheduler, scaler
when used, RNG states, global step, config, class map, data/split hashes and
preprocessing identity. An intentional weights-only new experiment must be named
as such. Validate resume by comparing uninterrupted versus interrupted execution
for a short deterministic sequence; define allowed numeric tolerance and sources
of nondeterminism rather than claiming universal bitwise CUDA reproducibility.

### J. Minimal supervised recovery ladder and go/no-go gates

Do not start with every historical regularizer active. Each stage requires a
written result and decision in `decision_log.md` before expanding scope.

| Stage | Controlled experiment | Required evidence / response |
| --- | --- | --- |
| J0 | Reproduce frozen failed evaluation and shipped baseline on the same development packet | Know which runtime/representation first diverges; preserve both results. |
| J1 | Numerical/data/checkpoint audits above; no optimizer steps | Fix proven defects and show a before/after regression case. No guessed normalization patch. |
| J2 | Overfit 2–4 clean training examples per class, plain CE, no augmentation/KD/prior/bias, online weights | Near-perfect tiny-training-set fit is a debugging target, not a generalization claim. If ~95%+ cannot be reached after a reasonable finite diagnostic run, inspect labels/load/gradients/capacity before scale-up; allow explicit ambiguous-sample exceptions. |
| J3 | Frozen-backbone feature extraction and linear-head baseline on grouped development splits | Assess whether inputs/features separate classes without representation drift; compare Healthy and diseases together. |
| J4 | Joint nine-class training with ordinary CE, natural shuffled training distribution, clean deterministic validation | Establish stable train/validation curves and per-class outcomes before imbalance correction. |
| J5 | Controlled single-change interventions: sampler OR class weights OR prior adjustment | Same initialization, development split, data budget and seed; measure Healthy benefit and disease cost. |
| J6 | Head then last-block unfreezing; cautious backbone LR; add realistic augmentation gradually | Preserve disease recall while improving verified Healthy field cases. |
| J7 | Calibration and exact exported/runtime/app evaluation | A working PyTorch model must survive export and policy unchanged before deployment consideration. |
| J8 | Final selected candidate, seed confirmation and locked-test evaluation | Apply the original acceptance gates; no tuning on the opened final test. |

For J2, freeze choices must still permit memorization. If a frozen feature head
cannot fit ambiguous/identical-feature examples, try a carefully bounded unfrozen
model before declaring the optimizer broken. Log the number of examples, steps,
loss, raw accuracy and exceptions. Do not interpret “memorizes tiny set” as proof
of good labels across the corpus or robust field accuracy.

J4 starting control: recovered compatible classifier if available, otherwise the
documented pretrained DINOv2 backbone plus a new nine-class head; no teacher,
no class bias, no energy gate, no label smoothing, no balanced-softmax/log-prior
term, no focal loss, no heavy random erasing, no mixup and no compile. Use raw
logits for CE, FP32 for initial diagnostics, explicit seed and a conservative LR.
Keep the deployed app's frozen calibration/policy evaluation as a separate
comparator; disabling bias in the training control is not a production change.

Learning rates, epoch counts and batch sizes later in the original handoff are
search suggestions after these gates, not executable defaults for a resumed ViT.
Train all relevant classes jointly; do not sequence Healthy-only optimization
followed by disease-only repair and call the oscillation learning progress.

### K. Explain the Healthy-versus-disease collapse with transitions

Make a paired table for every relevant intervention/checkpoint on identical
development samples. Record raw logits, calibrated probabilities, truth, old/new
argmax and old/new final verdict. Summarize at least:

- Healthy -> Healthy improvements versus disease -> Healthy false reassurances.
- Previously correct disease -> different disease, Other, Unsure or no leaves.
- Per-disease count/recall, disease macro recall, Other rejection and source mix.
- True-class margin, Healthy margin, predicted class distribution, logit scales,
  Healthy bias/head norm, feature norms and losses by class/source over time.
- Online versus EMA, before/after restore, before/after calibration, and
  PyTorch versus TFLite versus Kotlin transitions, separately.

Interpretation must be tied to an intervention:

| Observed signature | Leading checks, not automatic conclusions |
| --- | --- |
| Collapse already present at step zero after load | Class order, checkpoint provenance, missing keys, preprocessing and metric alignment. |
| Tiny-set optimization fails even with trainable weights | Input/label pairing, gradients, loss contract, LR, finite values and optimizer membership. |
| Train improves while source-held-out disease falls | Overfitting/domain shortcuts, split differences, sampling coverage and augmentation validity. |
| Both Healthy/disease margins shift abruptly after a bias change, with fixed logits | Decision/calibration tradeoff; weights did not learn new features. |
| Head-only probe works but full fine-tuning collapses | Representation drift, too-large backbone LR, update schedule, regularization or resume error. |
| Adding Healthy data sharply changes predicted prevalence | Effective prior, class weight/sampler interaction and source background shortcuts. |
| PyTorch good, TFLite poor on exactly the same normalized tensor | Export graph, dtype/layout, selected weights or operator parity. |
| TFLite crop good, app verdict poor | Detector selection, multi-leaf tracking/evidence, partial-disease override, calibration or stop policy. |
| Reported aggregate improves while disease macro recall drops | Metric/checkpoint objective hides the failure; use constrained selection. |

If representation forgetting is evidenced, compare conservative unfreezing,
smaller backbone LR and replay of diverse original diseased training examples.
Distillation from a verified prior classifier or a penalty on departure from
initial weights is conditional; log coefficients and supervised/retention terms.
Do not anchor a student to teacher mistakes on the very Healthy cases being fixed.
Do not maintain a “healthy model” and a “disease model” with ad hoc precedence
rules without evaluating their combined false-Healthy behavior.

### L. Loss, logit adjustment and RL: a precise decision protocol

Terminology matters because the old trainer already uses logits in supervised
learning. Label the tested method with its actual formula and implementation.

| Method | What it does | Role in this recovery |
| --- | --- | --- |
| Ordinary CE | `-log softmax(z)[y]` | Required transparent joint-class control. |
| Weighted CE | Weights labeled examples/classes | Conditional response to demonstrated imbalance/costs, not Healthy-only reward. |
| Training logit adjustment / balanced-softmax variant | `CE(z + tau*log(pi_train), y)` with declared prior and sampling convention | Reintroduce only after plain CE works; sweep one small coefficient at a time. |
| Inference class bias | `softmax((z + b)/T)` or bias at the app's combined log-probability stage | Decision boundary tuning on calibration data; cannot teach a visual distinction. |
| Margin/contrastive/retention term | Changes supervised representation or pairwise separation | Optional if a concrete, measured remaining error motivates it. |
| Label-based differentiable “reward” | Directly optimizes an expression of correct-class probability/logits | Usually testable as supervised loss; write its gradients/objective explicitly. |
| Actual policy-gradient RL | Samples actions and estimates reward-weighted policy gradients | Not the next default step for a fully labeled nine-class classifier. |

Adding a positive constant to Healthy at inference raises its decision preference;
it does not improve the image evidence. A label-conditioned log-probability
reward is simply negative CE when its formula matches `log p(y|x)`. Maximizing
`p(y|x)` is a different differentiable objective; do not call all reward losses
mathematically identical to CE. For a per-image known reward matrix R(y,a), its
expected reward `sum_a p(a|x) R(y,a)` is directly differentiable: sampling actions
with REINFORCE is not necessary just to estimate that sum over nine classes.

Only propose an RL experiment after all correctness gates pass and a defensible
objective remains that the supervised/cost-sensitive baselines do not address.
For example, a future sequential scan policy involving view acquisition costs is
a different task from a single-image label classifier and needs its own data and
environment validation. It is not authorized as an unbounded detour here.

If an actual RL comparison is scientifically justified within the authorized
budget, predeclare state/action/reward, whether decisions are independent or
sequential, ground-truth source, disease false-Healthy penalties, unsupported-input
behavior, exploration/off-policy assumptions, supervised anchor/KL constraint,
seed controls and stop criteria. Compare to ordinary and weighted CE on identical
splits and total compute. Rewards based only on model confidence, avoiding Other
or predicting Healthy are disallowed objectives. Report reward separately from
real accuracy; reward going up while disease recall falls is a failure.

Do not implement PPO/GRPO merely because training accuracy is low. A failure in
normalization, labels, checkpoint restoration or class coverage must be fixed in
all methods, and an RL run over that failure would obscure the diagnosis.

### M. Concrete implementation boundaries after diagnosis

Make each demonstrated defect reviewable with a minimal failing fixture, the
repair, and a regression assertion on actual tensors/labels/predictions. Proposed
new modules and schemas below are to implement as needed, not existing tooling:

| Area | Required implementation if implicated | Android consequences |
| --- | --- | --- |
| Preprocessing | Shared explicit transform specification with input dtype/range/color/layout/crop/resize and a version/hash; deterministic CPU/GPU evaluation parity | Fix Android only if its contract is demonstrably wrong for the selected deployed model; otherwise correct training/evaluation to match it. |
| Cache/loader | Validate producer config/hash, initialized length, dtype/range, per-image label lineage and split membership; reject stale incompatible cache | No app change for a training-only cache repair. |
| Label mapping/head restore | Immutable canonical class list, named head-row mapping and strict permitted-key load report | Preserve nine labels/order; a true taxonomy change requires the explicit migration later in this handoff. |
| Trainer | Flags for independent sampler/loss/regularizer/precision modes; full loss/gradient/update logging; full resume checkpoints | No runtime change until a model qualifies. |
| Checkpoint selector | Reject candidates violating disease/Other guardrails; report Healthy and all disease classes plus source cohorts | No Healthy UI shortcut; preserve disease priority/partial-leaf logic. |
| Evaluation | Bundle/config/split/preprocess hashes in resume keys; raw and calibrated metrics separately; real Kotlin parity | Policy changes, if justified, must be synchronized with replay and all consumers. |
| Export | Verified selected state, NHWC wrapper, real standardized input parity and final byte hashes | Update assets/manifests/goldens together only after qualification. |

Never let changing an experiment's preprocessor update a shared cached result
under the old hash/name. Cache invalidation is based on content/config identity,
not only modification time. A run with corrected inputs is a new experiment;
preserve the failed run for comparison.

### N. Resource use, stop conditions and deliverables for the SSH run

Use the first session to recover facts and pass the deterministic audits. An
initial few hours of GPU diagnostics may be appropriate, but do not spend the
full original 24–48 GPU-hour search envelope before identifying the failing
boundary. Treat resource numbers as caps, not targets; keep useful work moving
without repeated approval requests for routine, authorized project-local fixes.

Stop escalation to larger training when inputs/labels are invalid, checkpoint
identity is unresolved, tiny-set fitting fails without explanation, loss/updates
are nonfinite, repeated predicted-class collapse occurs, export parity fails,
or validation repeatedly worsens disease concealment. Preserve evidence, address
the cause, and resume at the relevant gate. Do not “fix” a gate by suppressing
its warning or removing failing classes from metrics.

At a minimum, return a root-cause report that distinguishes confirmed defects,
contributing factors, hypotheses tested and remaining unknowns. A confirmed cause
needs a reproducible example and a controlled correction showing the predicted
effect. Multiple causes are possible; no claim that a plausible code smell alone
explains the reported 30%.

Then deliver the corrected diagnostic/training tools, versioned config and data
split manifest, per-class before/after evidence, stable joint Healthy/disease
baseline if attainable, final export/runtime/app results and the PR/artifact
handback described below. If the training fix restores performance without any
app code change, say so. If no acceptable candidate exists, preserve the shipped
assets and return the best measured candidate with explicit failed/pending gates.

### O. Recovery completion checklist

- [ ] Remote failed run and reported metric identified or specific missing artifact recorded.
- [ ] Existing code/config/checkpoints/logs preserved; active jobs respected.
- [ ] Exact input-range/color/layout/crop/resize contracts traced with sentinel tensors.
- [ ] Normalization hypothesis tested numerically without blindly adding /255.
- [ ] Cache and image/label alignment verified; named class/head mapping asserted.
- [ ] Frozen checkpoint load and save/reload predictions verified.
- [ ] Actual gradients, parameter updates, sampler coverage and resume behavior checked.
- [ ] Tiny-set control and joint-class ordinary-CE baseline completed or failure explained.
- [ ] Healthy gains and per-disease losses explained with paired transitions.
- [ ] Each loss/sampling/retention change isolated against a controlled baseline.
- [ ] Calibration and exported-byte parity separated from representation learning.
- [ ] Final test, statistical guardrails and target-device checks honestly labeled.
- [ ] Status, evidence, reproducible commands and PR handback durable across SSH sessions.

---

## Original complete training and deployment specification, updated for recovery

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
| Ordinary CE recovery baseline | Joint classes, explicit natural-distribution control, no prior/bias/KD and deterministic evaluation | No runtime change during diagnosis | Required before weighted/logit experiments; see recovery stages J0–J4. |
| Class-weighted CE | Weight verified classes based on training prevalence/validation behavior; clipped weights | No new runtime logic | Conditional controlled comparison after ordinary CE works and imbalance/cost evidence motivates it. |
| Logit-adjusted supervised CE | Train with prior-dependent adjustment and declared coefficient | Usually no runtime change beyond new weights/calibration; document whether adjustment is training-only or also applied at inference | Conditional controlled comparison after recovery; not RL. |
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
