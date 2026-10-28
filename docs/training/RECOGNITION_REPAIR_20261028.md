# Recognition repair: recovery-only training and new evidence

## Objective and immutable qualification

Repair rejected Healthy tomato leaves without trading them for diseased leaves
reported Healthy. The existing Android model remains deployed until a
replacement qualifies. This research is separate from the completed normal-app
2.0.4 build and emulator runtime checks.

The initial frozen prospective OLID evaluation failed: 87/223 Healthy, zero
gain, and 16/507 disease-to-Healthy errors. **AD and OLID are permanently
consumed development.** None of the results below is independent acceptance.
The provisional TCP reservation remains diagnostically sealed, unused for
training, and of unestablished capture independence.

Preserve the original complete-population gates: at least 157/223 consumed
Healthy, at least 28/40 in the fixed validation cohort, at least ten percentage
points paired gain with positive lower family-bootstrap confidence bound,
no source/cohort disease-to-Healthy point increase, applicable upper confidence
bound no greater than two percentage points, and original named-disease/source
retention. Do not shrink denominators to the population the expert routes.
Full restored-model verification, export and replacement-model Android work
follow complete cheap recognition/safety checks, not benchmark AUC.

## What changed

`conditional_recovery_policy.py` protects every existing Healthy and named
disease result by returning it unchanged. A new expert can admit a previously
rejected case only through the existing whole-image, genus, compact-evidence
and suspected-disease guards. The `.95` expert threshold and all existing
evidence thresholds remain fixed. This isolates recovery from the legacy
Healthy-veto behavior present in older experimental routing.

The conditional fitting population has 345 training and 71 validation photos
after canonical-input/family quarantine. There are 13 retained-correct Healthy
validation photos outside the recovery population, 18 rejected Healthy cases
that could pass the fixed guards, and nine blocked cases. The counterfactual
guard ceiling is therefore 31/40, not 18/18. This is a diagnostic bound, not
accuracy. Shared exact input identities are joined by canonical SHA, never by
local row numbers from different cache namespaces.

New controls compare supervised BCE, cross-source supervised contrastive
learning, a gradient-reversed conditional domain adversary, differentiable
expected admission reward, longer warm-up, explicit target-error replay,
decision-margin loss, local patch pooling, whole/ROI feature fusion, LoRA,
clinical anchors, condition subclass supervision and hard-negative ranking.
Each comparison has a predeclared epoch cap and actual-verdict stopping rule.
Expected admission reward uses existing training labels; it is not
environmental RL/RLHF and does not manufacture new clinical information.

The target-replay sampler assigns half its probability mass to consumed OLID
training errors and half to protective training replay, retaining class and
family balance within the two populations. The later full-source control uses
half target errors, one quarter original anchors, and one quarter new-source
training. Validation labels never determine sampling weights or input features.

## Completed controls

The table reports each selected checkpoint's Healthy count on the same
40-photo consumed validation denominator. Unless explicitly stated otherwise,
selected checkpoints have zero newly Healthy diseased validation photos.
Higher-recall unsafe epochs were rejected. These are early screens, not
full-population qualification.

| Phase | Separately tested controls | Selected Healthy / 40 |
| --- | --- | --- |
| 118 | Conditional supervised; supervised contrastive; conditional domain adversary; expected admission reward | 13; 13; 13; 13 |
| 119 | Longer balanced warm-up; target replay; replay plus admission-margin loss | 13; 13; 13 |
| 121 | 224px gated-attention MIL; sparse top-k patch damage | 15; 14 |
| 122 | Additive whole/ROI fusion; bilinear fusion | 14; 15 |
| 124 | Rank-eight Q/V LoRA in all twelve DINO transformer blocks | 19 |
| 127 | Clinical disease anchors; condition subclasses; subclass contrastive loss | 14; 14; 14 |
| 129 | Native-448px DINO gated-attention MIL; top-k patch damage | 19; 15 |
| 132 | Licensed DINOv2-L ROI residual linear head; residual MLP | 13; 15 |
| 134 | LoRA with clinical-class-balanced disease anchors and nine-class image supervision | 13 |
| 135 | LoRA with supervised subclasses of Other and conserved initial logit mass | 13 |
| 136 | Clinical-anchor LoRA plus hard-negative pairwise ranking | 14 |
| 139 | Full 2,063-photo source, supervised head; ranking alternative | 13; 14 |
| 140 | Larger-encoder RBF SVM, C=1 and C=10, training-only family-OOF calibration | 13; 13 |
| 142 | Full-source copied CNN stages, supervised; ranking alternative | 14; 14 |
| 144 | Full CNN with label-symmetric photometric replay | 15 |
| 145 | Warm DINO LoRA with the same photometric replay | 20 |
| 146 | Original nine-class DINO LoRA with primary clinical supervision and photometric replay | 13 |
| 147 | Larger clinical head warm-up then DINOv2-L LoRA teacher | 25, with two new false Healthy |
| 148 | Continue selected large teacher with guard-aware expected admission reward | 14, with one new false Healthy |

All failed the 28/40 cheap gate and stopped before complete adapted-model
inference/export. Phase 130's cached complementarity diagnostic includes
twenty already selected experts. The optimistic OR of its nineteen
zero-new-FP experts reaches only **22/40**. It rules out that simple ensemble
as sufficient; it does not select a new deployment policy. The older CNN
suffix expert introduces two new false Healthy photos under recovery-only
routing and is excluded from that bound.

A subsequent guard-aware cutoff diagnostic used existing scores, including
the later photometric controls. Even an optimistic union with a separate
oracle zero-new-FP cutoff per expert reaches only **24/40**. Its best individual
bound is 22/40. These cutoffs deliberately consume development labels and are
not a selected deployable policy or acceptance evidence. The diagnostic rules
out a simple cutoff/ensemble repair of these selected checkpoints; it does
not rule out every possible model or training dataset.

Clinical replay matters for the hypothesis even though it did not qualify:
conditional-only training omits correctly recognized diseases. The replay
controls expose those disease exemplars instead of trying to repair all
Healthy rejection by shifting a scalar threshold. Clinical labels supervise
the image bag. Broad diagnosed-disease labels exclude Healthy without being
invented as a particular supported diagnosis or Other subtype.

The subclass LoRA control duplicates the old Other head into supported
training subclasses, subtracting `temperature * log(number_of_branches)`
from their biases. Their summed calibrated non-Healthy mass initially equals
the old Other mass. A restore audit and regression test verify preservation
of the original Healthy margin before training. These internal classes never
become unsupported app diagnoses. Hard-negative ranking is a separate
training-only loss against the highest diseased scores in each batch.

## Cache reuse, measured work and safeguards

Original DINO/CNN features, canonical pixels, detector outputs, whole SigLIP
features and compatible prefix caches are reused. New representations are
computed only when weights, preprocessing or input identity changes.

| Cache/work | New computation | Measured duration |
| --- | --- | --- |
| 120 local final tokens | Final two original DINO blocks on 868 existing prefixes; zero prefix reruns | 21.67 s |
| 123 LoRA input prefix | 868 unchanged patch embeddings; 16 final-token audit cases | 38.69 s |
| 128 native-resolution DINO | 868 native-448px ROI inputs; 16 dynamic-224 graph audit cases | 42.65 s |
| 130 complementarity | Cached head scores only; zero encoder inputs | 7.47 s |
| 131 licensed DINOv2-L ROI | 868 genuinely missing canonical ROI features | 44.12 s |
| 133 disease-anchor prefix extension | Reused 723 exact prefixes; 895 missing patch embeddings; 16 original-CLS audit cases | 9.98 s |
| 137 full-source original features | Reused pilot inputs; 1,807 new detector inputs and 14,880 missing encoder inputs | 382.86 s |
| 141 full-source CNN spatial prefix | Reused 1,618 old prefixes; 17,056 newly required spatial representations; zero DINO/detector reruns | 344.90 s |

Native 448px inputs come from the original source images and verified detector
boxes, not enlarged 224px tensors. Dynamic DINO position interpolation is
pinned to timm version and implementation SHA. Matching original 224px CLS
and CNN features preserve the warm classifier; native patch tokens add local
detail. The dynamic-224 audit maximum CLS error was 2.84e-6. The larger
encoder cache checks the existing licensed encoder contract and exact pixels
before reuse: none of these ROI inputs matched its older whole-image cache,
so all 868 needed computation.

Phase 120 initially wrote a NumPy shape list into the NPY header. Its repair
changed the header only; payload hash stayed
`1ab5a376b6767981ae91466b495b12704be0bc890d1e55ba802ed232325ee2c9`.
The archived source/protocol and header-repair record are retained. A guarded
compatibility path verifies unchanged inference AST and all input/model/cache
identities before reuse. `safe_feature_memmap.py` now always writes tuple
shapes and rejects changed existing dimensions/dtype. The first phase-121
initialization failure and a phase-134 import-shadowing initialization failure
are archived separately; neither had trained a checkpoint. No completed
experiment was restarted to repair either initialization issue.

The frozen phase-132 protocol inherited a replay-hypothesis description from
its parent driver. Its actual methods are `large_linear` and `large_mlp`, as
shown by its source, checkpoints and results. This descriptive metadata error
does not change the executed computation. Treat this paragraph as an erratum;
do not alter historical protocols or rerun completed weights to change prose.

## Full-source supervised control

The verified Mendeley development collection contains 1,013 Healthy Leaf and
1,050 broadly Infected Leaf photographs. The pilot used 128 per class; a new
bounded control expands supervision to all retained images. Camera/style
confounding remains recorded. All images receive the same four right-angle
rotations; this reduces orientation shortcuts without proving capture
independence. The entire collection remains development-only.

Before additional neural work, phase 138 compared all **2,063** proposed
training photos against the **2,461** provisional protected images using the
already cached geometry fingerprints. It found zero matches in **23.11 s**,
computed no new fingerprints or encoder outputs and opened no clinical
predictions. This screen protects the reservation; it does not establish its
independence or make it acceptance-ready.

Phase 137 reuses the pilot's exact-input feature and detector cache and computes
only newly required inputs. It exposed all 2,063 images and found actual ROIs
in 2,062. Phase 139 compared ordinary supervised BCE and BCE plus hard-negative
ranking on that expanded cache; both failed. Phase 142 then adapted the copied
CNN stages with those same supervision/replay populations, and also failed.
No failed candidate was exported. Entire-collection usage is recorded in the
PC source's usage marker; the unused-remainder acceptance prohibition remains.

## Appearance-specific controls and numerical validity

A private contact sheet of the consumed Healthy failure cases suggested an
appearance/lighting split: several low-scoring leaves are brighter and appear
underside-like. This is visual triage, not a new clinical or leaf-side label.
No validation image was relabelled. Phases 144–146 therefore apply the same
moderate RGB brightness, contrast, saturation and white-balance variation to
both clinical classes, shared across real views of each training photo.
Validation pixels remain unchanged. They use the audited 256-photo new-source
pilot plus original clinical anchors and target-error replay. The old source
collection is not promoted to prospective data.

All 3,794 restored raw inputs are checked against their exact canonical SHA
and original CNN features before photometric training. The initial phase-144
audit caught cuDNN TF32 precision drift; both TF32 switches are now disabled.
A subsequent startup index failure came from a margin variable shadowing the
pixel bank. The pixel bank now has a separate name and a two-consecutive-batch
runtime check precedes optimizer updates. Failed startups are archived apart
from the completed valid run. The earlier phase-142 negative-control training
used library-default cuDNN precision and is not an FP32 restore/parity result.

Phase 145 reused unchanged original DINO input embeddings for identity views
and computed patch embeddings only for genuinely changed photometric inputs.
Phase 146 also persists the 2,176 missing pilot input embeddings for subsequent
compatible controls. Encoder passes during changing-weight training are not
represented by the earlier cached-head `new_encoder_inputs=0` counters. The
photometric controls instead distinguish cached features from full-encoder or
adapter training in their protocol metadata.

## Larger clinical teacher and reward continuation

Phase 147 uses the already downloaded, licensed official DINOv2-L encoder.
It restores/reuses matching original CLS features, computes only missing ones,
and first fits a nine-class clinical head from that cache. Up to ten head-only
warm-up epochs may exit early on a cheap pass. If necessary, a single rank-eight
Q/V adapter control follows: 512 training-photo draws per epoch, batch two,
eight epochs minimum, six stale actual-verdict epochs, twenty-four maximum.
The same clinical loss, partial-label handling, photometric replay, protected
legacy verdicts and unchanged Healthy/safety gates apply. Mutable-parameter
checkpoints exclude the 1.2GB frozen encoder; restore rejects incomplete or
changed trainable-parameter contracts. This is a **teacher capacity probe**;
mobile student distillation and its own qualification would still be required.

The first large-encoder cache audit caught a second numerical issue. A single
audited input drifted by 0.012824 in normalized features when identical RGB
values used contiguous NCHW rather than the original encoder's channels-last
NCHW input view. A direct comparison reproduced the drift, showed zero added
error from zero-initialized LoRA, and matched the native original encoder/cache
within 9.7e-6. The teacher now explicitly pins patch-input memory layout and
audits reused cases **before** bulk extraction. The first incompatible new
feature shard is archived as `INVALID_NCHW_LAYOUT`; it must never be reused.
Original verified feature caches were preserved. Precision/layout fixes protect
experimental validity; they do not establish the cause of the app's rejection
problem or constitute an accuracy improvement by themselves.

With layout pinned, the preflight passed. Phase 147 reused its matching old
CLS features and computed only missing new ones, completed its cached clinical
head warm-up, then stopped after six stale adapter epochs. The selected
checkpoint reached **25/40 Healthy with two new false Healthy**, so it failed
both the 28/40 Healthy screen and the zero-new-FP selection constraint. It was
not distilled or exported.

Phase 148 continued that selected teacher checkpoint with a training-only
guard-aware expected admission reward. It reused the completed parent warm-up
and learned adapters, as well as compatible frozen features. Only training
photos for which the fixed app guards can admit Healthy receive action reward;
protected legacy verdicts and the new-source auxiliary images retain clinical
supervision. The reward's false-Healthy cost is eight, boundary temperature .5,
weight .2; this is differentiable exact expected reward from known labels, not
environmental RL or new acceptance feedback. After its bounded continuation,
the selected checkpoint was **14/40 with one new false Healthy**. A later
28/40 epoch introduced nine new false Healthy and was rejected. This reward
alternative did not solve recognition or qualify a deployment candidate.

Inspect current `tmux` windows and completed result identities before doing
more work; do not rerun either completed teacher experiment. A missing result
for any subsequent control means incomplete, never a pass.

## Reproduction and next action

### Additional completed controls: 149–158

These controls also failed the unchanged recognition/safety screen. No failed
weights were exported or added to Android. Counts below are actual consumed
validation app verdicts, including the 13 protected Healthy cases; the minimum
is 28/40 with zero new disease-to-Healthy errors before complete development
screening. Neither AD nor OLID is independent acceptance.

| Control | Selected Healthy / 40 | New false Healthy | Outcome |
| --- | ---: | ---: | --- |
| 150 semantic clinical prompts, two regularizations | 13 each | 0 | Stop |
| 151 true native-448 global CLS, original clinical head | 13 | 0 | Stop |
| 151 native-global residual linear / MLP | 13 / 18 | 0 | Stop |
| 152 corrected per-photo clinical weighting, large teacher | 21 | 1 | Stop |
| 153 original CNN Healthy output, full-source supervised tail | 14 | 0 | Stop |
| 154 ConvNeXt-Tiny clinical teacher with channel-MLP LoRA | 17 | 0 | Stop |
| 155 Healthy patch memories, 224 / 448 | 13 each | 0 | Stop |
| 156 cached large clinical head, hard-negative continuation | 14 | 1 | Stop |
| 158 learned large-DINO / ConvNeXt joint clinical residual | 14 | 1 | Stop |

**Clinical weighting correction.** The original photometric clinical controls
146–148 flattened all valid views before averaging. New-source photos have
four rotations whereas old photos have one; that inadvertently gave each
new-source photo four times the clinical loss weight. The binary photo loss
did not have this problem. `balanced_clinical_loss.py` first averages valid
clinical views within each photograph, then averages photographs. Phase 152
reran the changed objective with the verified large teacher and otherwise
matching sampler, seed and input policy. It reached 21/40 with one new false
Healthy and failed. View-replication invariance, coarse disease labels and
missing-view rejection have regression coverage. Earlier completed results
remain historical evidence, not evidence for the corrected objective.

**CNN output semantics.** The retained CNN's two outputs are independent binary
logits: column zero is Healthy; column one is knownness. They are not a
two-class softmax. Experimental margin controls such as 101/110/142/144 reused
their difference, which does not preserve the original Healthy warm function.
The existing app fusion already handles the original semantics correctly.
Phase 153 restores the original Healthy row directly through the existing
single-output tail, audits zero-update parity and applies the full-source
supervised recipe. Its 14/40 safe result does not solve recognition. See the
recorded [axis diagnostic](evidence/recovery_controls_20261028/cnn_axis_semantics_diagnostic.json).
Do not describe this as an app normalization defect or a shipped-app fix.

**Semantic features without re-encoding photos.** Phase 149 obtains the official
SigLIP2 text tower from immutable revision
`3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab`. Every vision tensor exactly matches
the already cached image encoder, proving the shared coordinate basis. Nineteen
fixed clinical prompts use 103 vocabulary rows; exact ID remapping and a
full-versus-pruned small-model forward check give zero error. HTTPS ranges
downloaded 714,918,784 bytes in 240.48 seconds rather than the full model.
The full parent-file hash was not verified; range hashes, pinned identity and
the complete vision-tensor proof are recorded instead. Phase 150 uses family
out-of-fold training calibration, whole-pixel overlap quarantine and two fixed
regularizations. It runs no image encoder. Both controls fail at 13/40.

**Independent backbone probe.** Phase 154 uses official ConvNeXt-Tiny weights,
audits frozen-feature restoration, warms its nine-class head from the cache,
then adapts the channel-mixing linear layers with zero-initialized LoRA and
the corrected per-photo clinical loss. Inputs remain the canonical 224-pixel
app ROIs with ImageNet normalization; this is not the standard 236-resize
ImageNet evaluation transform. Mutable checkpoints omit the frozen backbone.
The selected safe checkpoint is 17/40; the 26/40 epoch introduces five false
Healthy. No mobile student or export was attempted.

**Normal-patch alternative.** Phase 155 uses existing normalized local tokens
at 224 and 448 pixels. Healthy reference banks and normality calibration use
separate inner training folds joined by family and shared ROI identity.
Farthest-point coresets retain 512 actual training patches; distance quantiles
feed a logistic calibrator constrained so greater anomaly cannot increase
Healthy probability. Per-view probabilities retain the conservative crop-pair
minimum and fixed .95 admission. This is inspired by patch-memory anomaly
detection, not an exact PatchCore reproduction. Both resolutions recover no
additional Healthy validation photo. No encoder inference is needed.

**Cheap hard-negative stopping test.** Phase 156 continues phase 152's completed
cached clinical-head warm-up on the same frozen large-encoder feature basis.
It adds training-only hardest-negative ranking to clinical/binary supervision,
with one fixed objective and six-stale-epoch stopping. It finishes in 3.06
seconds, without loading an encoder or decoding photos, and fails at 14/40
with one new false Healthy. Its head cannot be mixed with adapted-encoder
features. No further encoder training is justified by this head result alone.

**Separation audit.** Phase 157 replays 33 selected prediction caches with exact
validation-index checks and file hashes. It uses consumed validation labels
only for an explicitly optimistic diagnostic: allow each expert its own
zero-new-FP threshold, then union its safe recoveries. Even that oracle union
reaches only **23/40**, below 28. There are 18 guard-reachable rejected Healthy
cases and 32 admissible negative cases. These thresholds are never candidates
or deployment settings. This stops simple monotone threshold/OR exploration
for these 33 selected checkpoints; it does not rule out learned interactions,
other checkpoints, architectures or improved supervision. The older phase-130
diagnostic includes a different expert set and is not directly interchangeable.

Phase 158 explicitly tests the learned-interaction alternative rather than
another threshold combination. It joins the existing large-DINO and ConvNeXt
representations by canonical input SHA, initializes a joint nonlinear residual
to zero over the completed clinical head, and uses the same clinical/binary/
hard-negative supervision. Exact warm-logit preservation has regression
coverage. This bounded control finishes in **3.16 seconds**, with zero new
encoder inputs, at 14/40 with one new false Healthy. It also stops; no full
encoder verification, distillation or export follows that failed screen.

The next substantive model work must state what new information or visual
representation should resolve the remaining separation failures before another
run. Preserve the corrected objectives, original named disease verdicts and
all safety gates. Do not start another loss/threshold grid or rerun completed
adapters. Suitable additional field supervision requires documented clinical
labels, source/family separation and overlap audits; the existing publisher
collection's orientation/EXIF confound makes it insufficient independent field
evidence. No recognition completion date is supportable while no candidate
passes the development screen and fresh acceptance is still unqualified.

Campaign: `/home/genesis/agribot-data/recognition_campaign_20261024` on
`agribot-5090`. Python: `/home/genesis/.venvs/agribot-train/bin/python`.
Code overlay: `/home/genesis/agribot/model_training/v3`. Useful jobs run in
`tmux` session `agribot`. Inspect jobs, logs, results and cache identities first;
do not pull over the dirty overlay, disrupt useful runs or rerun completed
controls. Completed failed approaches stop at their declared caps.

If a new candidate passes the cheap screen, freeze that candidate and
its routing protocol, produce predictions for every applicable consumed case,
and run all unchanged complete-population gates. Any previously computed
matching prediction must be reused. Only a candidate passing all of those
checks proceeds to restored-model verification and parity. Truly suitable
unconsumed prospective data must have an independently justified protocol
before opening; AD, OLID and this new collection cannot fill that role.

Local regression validation before these additions: **228 Python tests, one
optional media skip**. The additional clinical weighting, adapter merge,
monotone calibration and range-offset/corruption tests are included in the
latest machine-readable test evidence alongside this document.
The latest full run passes **240 tests with one optional media skip**, in
22.333 seconds. These are software/contract checks, not recognition acceptance.
Tests cover protected verdict identity, GRL direction, contrastive edge cases,
replay mass, zero-residual warm parity, LoRA merge/key preservation, readable
NPY headers, cache-identity rejection, condition logit conservation and
hard-negative gradient direction. Android sources and shipped models did not
change during these experiments; the existing app validation remains separate.

PR #6 was checked ad hoc and had no external head change from `ddeae3f` before
this work was published. No periodic monitor is enabled and no auto-merge is
authorized. See [machine-readable evidence](evidence/recovery_controls_20261028/summary.json)
and the [app/completion checkpoint](COMPLETION_CHECKPOINT_20261028.md).
