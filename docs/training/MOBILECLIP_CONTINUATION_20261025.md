# MobileCLIP, exact context verification, and the next data revision

Revision: 25 October 2026. Read this before older pending statements in the
handoff. Recognition remains unresolved; the phone still uses interim 2.0.2.
The unchanged original nine-class disease model is the reference throughout.

## Completed work and decision

There are **285 valid completed trained controls** before the phase-twelve
revision. Starting from 262, these add two leaf CNN/genus controls, four DINO leaf
heads, two copied-tail leaf controls, eight MobileCLIP/mixed leaf heads, four
MobileCLIP image heads and three fitted image-logit stackers. Threshold grids,
prototype assembly and replay do not count as training runs. Eight invalid
exploratory fits are separately quarantined and excluded from this count.

The degradation-augmented copied DINO leaf control also failed end-to-end
Healthy recall. Full-ROI recovery and replacing individual leaf disease votes
while preserving the reference's aggregate disease verdict did not fix it.

The new representation is the image encoder from
[MobileCLIP2-S0](https://huggingface.co/timm/fastvit_mci0.apple_mclip2_dfndr2b),
the timm port of [Apple's published model](https://github.com/apple-aiml-research/ml-mobileclip).
Repository revision: `577a2e992da2c7ba7cadc2a490bcfadfb2d69bce`.
`model.safetensors` SHA-256:
`70a995b7e7385126d0db94ee3d395d7fd3c251ce6d8dcd4f4c8d8fc7c29e3d04`.
The model card declares `apple-amlr`; these weights are not redistributed in Git.
Preserve that license/attribution in any eventual distributable model.

Its checkpoint expects RGB in [0,1] at 256 pixels. Our explicit contract starts
with the existing canonical ImageNet-normalized NHWC 224 tensor, reverses that
normalization, and uses bilinear resizing to 256 inside the encoder. This differs
from the publisher's PIL bicubic preprocessing. Training and future export must
use this same recorded adaptation; do not substitute normalization or resizing
to imitate a published benchmark score. Features use L2 normalization and RMS
scaling. Mixed heads concatenate these with normalized original DINO features.

The corrected leaf verifiers reach about 0.992 validation ROC AUC. Their complete
graph assembly reproduced 8,909 cached reference/head cases: maximum reference
logit difference 0.000224, binary-head difference 0.00000573. App-level retention
still limits field Healthy recovery. The best retained mixed leaf policy reaches
15/129 TOM Healthy and 113/119 paper Healthy, versus interim 14/129 and 51/119.
Less conservative leaf policies add false Healthy calls and remain rejected.

Four image-level heads were then fitted on the existing 22,472-photo scene
corpus. Their labels remain image/primary-view evidence; they are never broadcast
onto individual leaves. A retained single-model policy reaches 68/129 TOM
Healthy. Less conservative policies reach 121/129 but add false Healthy calls.

Three nonnegative, supervised logit stackers combine five calibrated inputs:
the existing adapted DINO linear head, original/adapted DINO MLP, copied DINO12
image verifier, genus CNN, and new MobileCLIP image verifier. They use the
existing 65% hashed calibration-group partition of development validation, with
the other groups for internal selection. Base checkpoints already used broader
validation, so this is explicitly internal development, not independence.

Candidate 144 passes development retention and the frozen 70% source targets:

| Opened development task | Interim 2.0.2 | Fused prototype |
|---|---:|---:|
| TOM2024 Healthy | 14/129 | 94/129 |
| Paper Healthy | 51/119 | 112/119 |
| Combined false Healthy | 11 | 11 |

Its Healthy threshold is 0.97; no external genus gate, center-view agreement or
individual named-leaf veto is used. Existing aggregate named disease verdicts
remain authoritative because this policy only recovers Healthy from an
undecided/unsupported baseline. The five nonnegative weights are
`[0.053147743, 0.249330331, 0.269718651, 0.138401827, 0.645228332]`, bias
`-1.390159393`; inputs are clipped to +/-10. Negative-cost fit: 2.

`phase11_context_model.py` constructs a self-contained executable graph, not
just a spreadsheet of cached scores. A fresh restore exactly matches its saved
bundle. Full replay covers 5,820 distinct development paths and **6,468 verdicts**
including overlapping cohort membership. Every verdict matches the cached
selection; maximum clipped context-logit discrepancy is 0.000437. All development
retention and recall gates still pass with the exact graph.

The exact graph then replays the consumed phase-eight cohort. It recovers only
**51/95 TOM Healthy**, versus interim 8/95. False Healthy stays 3; disease and
unsupported retention pass. Recall again fails the unchanged 70% requirement.
This is not a qualified production candidate. The Abu Dhabi source was not
opened to diagnose or tune this failure.

## An invalid new experiment was caught and quarantined

The first MobileCLIP feature extractor constructed an encoder without loading
its pretrained state. Cache-versus-runtime parity rejected it before its app
policy replay; its calibrated Healthy logits differed by 0.797. This was a bug
introduced in this continuation, not evidence about the installed app.

The two feature directories and all eight fitted heads are preserved under
`*_INVALID_UNLOADED_ENCODER_20261025`. Their AUC and comparisons are invalid.
The corrected extractor loads weights strictly, verifies a fresh restore on
54 distributed cases, and only then writes a usable feature cache. Corrected
feature error is 0.00000430. Each assembled head also checks the actual cached
calibrated logits across all replay inputs. Both corrected families were retrained
from their deterministic seeds. The installed classifier and earlier DINO
experiments were unaffected.

## New data revision: the consumed cohort is retired as a benchmark

Repeatedly treating the same opened 246 photos as permanently untouchable fitting
data has stopped being useful. Phase twelve admits their **787 related photos
from 235 capture families** into a newly versioned image-level training/development
corpus. This is an explicit benchmark retirement. It does not make later scores
on those families independent or qualify a model by itself.

`phase12_admit.py` reconstructs all 5,214 original source records and the original
perceptual/input-hash family unions. It checks recovered families against the
immutable old audit, preserves all 200 original quarantined IDs, and leaves all
old membership files unchanged. Original development families stay in their
existing partitions. New family splits use hashed seed 503 with 20% for internal
validation. No diagnostic model is used during admission. Cold-source file/RGB/
perceptual-hash checks found no overlap family; AD remains unopened diagnostically.

The new corpus has 23,259 scene rows. SHA-256:
`aa429d09e6800af20e27b0ee1158179984b2bb6795780451ee61e0268055cfb9`.
Additional membership is 210/53 train/validation TOM Healthy, 193/72 TOM negative,
and 204/55 paper negative. Generic virosis and broad conditions remain binary
negatives; no nine-class disease labels are invented. These are publisher image
labels, not independently adjudicated disease-free-plant labels.

**For phase twelve and later, do not use the old 246-photo cohort or its related
families as regression or acceptance gates.** They now belong to training/internal
development. The original eight app cohorts remain regression gates; the separate
AD source remains available for a genuinely frozen independent assessment.
Old phase-eleven numbers above remain immutable pre-admission historical results.

The first new-data comparison fits four frozen MobileCLIP heads, followed by
app replay on the original eight cohorts. These four fits completed, bringing
the count to 289 before encoder SFT; their retained policy still misses recall.
The separate MobileCLIP encoder/head is now being fine-tuned with supervised CE,
focal and label-supervised distillation controls, preserving
the complete installed nine-class model. Photometric/degradation augmentation
must retain the image target. Optional distillation may use the five-expert
teacher on old training images where it agrees with the label; it must not
force the teacher's failed predictions onto newly admitted Healthy examples.
Source-balanced grouped validation and unchanged regression gates select a
candidate. No user approval is needed to continue these already authorized
research steps; no new candidate qualifies merely because its fit completed.

## Native app, APKs and release status

The opt-in schema-four 18-output reference-disease-priority implementation is
tested and pushed in `ed72afd`. It concerns leaf recovery and is disabled in the
existing schema-two assets. The new whole-image context graph is still a research
prototype; its domain evidence/runtime and exported-byte/device parity are not
implemented or qualified. Do not mislabel its two auxiliary logits as disease
class logits, apply nine-class temperature to them, or create fake leaf boxes.
Image evidence must use actual frame counts and identify its image-level scope.

The rebuilt debug and unsigned release APKs are at local
`dist/agribot-2.0.2-reference-priority-20261025/`. They retain classifier SHA
`d70135804cf750341218e89a055370bef46a9b337951897b9f55f2570844ed39`.
Debug APK SHA-256:
`7918784ddfdc3603bbe5436fb122210b0ed16ec3ace08a4ef30f2b8b8a06f739`.
Unsigned release APK SHA-256:
`6c98f82ccfbc8f7870c82c926a773e0586fb2d7cdf04fbd6b279c863583a4a69`.
The debug certificate matches the existing development certificate. No new
installation or device-validation claim is made; ADB was absent at the last
check. Do not uninstall the user's app or clear its data.

Native validation: 105 unit tests, release lint, test-APK assembly, debug and
release assembly passed. Android Native CI passed `ed72afd`, run 40203940342.
The latest Python suite runs 122 tests: 121 passed, one skipped. New model code
also has real checkpoint/cache/prototype parity checks described above.

## Resume and reproduce

SSH alias: `agribot-5090`; Tailscale/tmux are already configured. Repo scripts:
`/home/genesis/agribot/model_training/v3`. Python:
`/home/genesis/.venvs/agribot-train/bin/python`. Artifact base:
`/home/genesis/agribot-data/recognition_campaign_20261024`. Long jobs run in tmux
session `agribot`; inspect completion files before restarting. Never pull over
the dirty training checkout. Git publication uses the separate
`/home/genesis/agribot-research-publish` checkout and draft PR #6.

Pinned source metadata is `phase10_leaf/mobileclip_source.json`. Its HF file
is downloaded with `hf_hub_download` using the revision/SHA above, then timm's
`pretrained_cfg` records the native mean/std/size. `mobileclip_features.py` checks
these identities, strictly loads the persisted encoder and rejects stale caches.
The existing scene cache is reused through absolute, read-only pixel-shard
indexes; no 13 GB source cache duplication is needed.

```bash
PY=/home/genesis/.venvs/agribot-train/bin/python
BASE=/home/genesis/agribot-data/recognition_campaign_20261024
$PY phase11_prepare_encoder.py --root "$BASE/phase10_leaf"
# These completed historical experiments use the old split:
$PY phase10_feature_train.py --root "$BASE/phase10_leaf" --feature-family mobileclip
$PY phase10_feature_train.py --root "$BASE/phase10_leaf" --feature-family mobileclip_dino
$PY phase10_feature_train.py --root "$BASE/phase11_scene" --feature-family mobileclip --task-scope primary_image --primary-source tom2024
$PY phase11_scene_select.py --campaign "$BASE"
$PY phase11_logit_fusion.py --campaign "$BASE"
$PY phase11_scene_select.py --campaign "$BASE" --fusion-run logit_fusion --out-name fusion_selection
$PY phase11_verify.py --campaign "$BASE"
# Current new-data controls; old 246 are now retired as benchmarks:
$PY phase12_admit.py --campaign "$BASE"
$PY phase10_feature_train.py --root "$BASE/phase12_scene" --feature-family mobileclip --task-scope primary_image --primary-source tom2024
$PY phase11_scene_select.py --campaign "$BASE" --model-root "$BASE/phase12_scene"
$PY phase12_pipeline.py --campaign "$BASE"
$PY phase10_report.py --root "$BASE/phase10_leaf" --out "$BASE/phase10_evidence.json"
```

Logs: `phase11_scene/prototype_verify.log`,
`phase12_admit.log`, `phase12_scene/initial_controls.log`,
`phase12_scene/phase10_feature_train.log`, and
`phase12_scene/phase11_scene_select.log`. Exact-graph artifacts live under
`phase11_scene/prototype_verification/`; the retired benchmark role is recorded
under `phase12_scene/RETIRED_BENCHMARK.json`.

Before promotion, freeze the final data/model/calibration/implementation, verify
the exact exported LiteRT bytes, open the independent AD source once with
family-aware uncertainty and unchanged retention rules, integrate honest native
image evidence, and run device parity/navigation checks. A compact supervised
student is preferable if it retains the teacher's quality and improves measured
phone latency. Do not install a model that failed these gates, lower a frozen
target to force a pass, or present an in-sample result as a new benchmark.
