# Recognition convergence, capacity and checkpoint handoff — 29 October 2026

## Release decision

No replacement recognition model is qualified. The normal Android 2.0.4 app
remains built with its original classifier. Do not export or integrate these
failed research weights, mark PR #6 ready, or interpret software-test success
as recognition acceptance. The phone remains disconnected.

This continuation follows [phases 159–182](FIELD_AND_NATIVE_REPAIR_20261028.md).
It preserves the original Healthy, false-Healthy, source-retention and exact
verdict gates. AD and OLID are consumed development; TCP is still provisional,
diagnostically sealed and excluded from training. There is no newly independent
acceptance result. PR #6 is checked when useful; recurring monitoring stays off.

## What the new evidence rules out

The suspected 0–1 versus 0–255 CNN normalization mismatch was checked directly
against the loaded model buffers: canonical and CNN means are
`[.485,.456,.406]`, standard deviations `[.229,.224,.225]`. The copied full
mobile expert passes original-feature parity before adaptation and preserves
all 71 warm validation verdicts. This does not certify every historical script,
but it rejects that specific explanation for the current verified pipeline.
The earlier CNN Healthy-axis training error remains historical and corrected.

More training alone did not fix the native adapter: 40 extra epochs reduced
sampled training loss from .362331 to .0241305, while the best safe incumbent
remained 14/40 Healthy. Its peak 27/40 introduced six new false-Healthy outputs.
Every one of those 40 checkpoints and prediction files is retained and its
file hash was checked. Restoring omitted field cases, balancing disease
conditions, adapting all local patch tokens and unfreezing complete suffix
blocks also failed the unchanged screen.

The frozen mobile-head family has a stronger stopping result. Phase 207
recomputed **unclipped** calibrated logits for 166 completed checkpoints using
the exact original cached features. Saved probability and .95-admission parity
were checked. Even the label-oracle zero-new-false-Healthy cutoff recovers at
most 16/40 Healthy; the union of all such per-head oracle outputs is also
16/40. Those oracle cutoffs are diagnostic and cannot be deployed. They show
that monotonic recalibration and OR mixtures of individually zero-error
thresholds cannot meet 28/40 in this declared family. This is not an
impossibility proof for a different representation, arbitrary conditional
classifier, or different data.

## Completed controls

Healthy counts below use the **full 40-image development denominator**, with
13 fixed retained Healthy cases. `+FP` means new disease-to-Healthy outputs.
An incumbent means the safe starting model remained preferable; it does not
claim the trained model itself was safe. None of these rows qualifies a model.

| Phase | Change and result |
|---|---|
| 183 | Exact native LoRA model/optimizer/RNG continuation, 40 extra epochs. Safe incumbent 14/0; peak 27/+6. |
| 184 | Cross-label geometric check: 24 consumed nitrogen-deficiency ROIs against 194 Healthy ROIs, 4,656 pairs, no geometric matches. No relabeling. This does not establish clinical correctness or independence. |
| 185 | Binary/source/clinical-condition/family-balanced native head: 14/0. |
| 186 | Explicit 75-dimensional color/texture evidence appended to frozen native features; two training-only calibrated kernel settings: both 14/0. |
| 187 | Optimistic convex probability and clipped-logit committees over 98 experts: both upper bounds 23/40 at zero new false Healthy. Scope predates later full-patch and full-mobile runs. |
| 188 | Fixed final-ten native parameter average, without validation-selected averaging weights: 24/+3. |
| 189 | Restored 71 omitted field-training cases: 142 new native and 142 new ConvNeXt inputs; 2,106 earlier joint features reused exactly. |
| 190 | Full-field native frozen heads, binary and clinical-balanced sampling, 60 epochs each: both selected 14/0; peak 23/+3. Every epoch weight and prediction retained. |
| 191 | Two-block LoRA plus attention over all 1,369 adapted local tokens, 40 epochs: safe incumbent 14/0. All epoch weights and predictions retained. |
| 192, 194, 197 | Fixed negative veto diagnostics on existing Healthy decisions; exhaustive replay corrects some consumed false Healthy but changes protected earlier verdicts. Truth-label audit distinguishes likely corrections from losses. Current exact gate still rejects every variant. |
| 193 | Every quarantined original training case, target field cases and additional field supervision; clinical-balanced BCE and BCE-plus-ranking, 40 epochs each: 14/0 and 15/0. All epoch weights retained. |
| 195–196 | Exact 90/180/270-degree training rotations: 34,983 new encoder inputs, 11,661 original inputs reused. Same two exhaustive replay objectives, 40 epochs each: both 15/0. Validation images remain unaugmented; whole-family quarantine preserved. |
| 198 | Existing-Healthy veto must agree in both crops and all four orientations. SFT changes 18 protected earlier publisher-labeled non-Healthy cases, ranking 37; neither changes a protected publisher-labeled Healthy case. Both remove 2/16 consumed OLID false Healthy, retaining 87/223 correct Healthy. Still rejected by exact retention and Healthy recall. |
| 199 | Full final two native transformer blocks plus adapted local patches, 40 epochs: safe incumbent 14/0. Reused the persistent frozen prefix; computed only 142 missing prefixes. |
| 200–202 | Cached actual-admission policies: single native branch plus fixed mobile OR cannot cover 28 Healthy. Of 3,486 two-native pairs, 18 can cover 28; .05 new-admission veto variants still have at least +4 FP at 28 Healthy. Existing outputs preserved. No new encoding in the final veto screen. |
| 203 | Family-distinct, weighted five-neighbor classifiers, unrotated versus four-orientation training banks; grouped training-only calibration. Both 13/0. |
| 204–205 | Standard-MAP (.5) new-admission veto. Phase 204 failed after valid encoding because a loop overwrote validation IDs; its prediction NPZ is invalid. Phase 205 reused the 96 valid features and canonical IDs, with zero new encoder calls: best still 28/+4 (OLID three, TLID one). |
| 206 | Full original DINO and CNN feature encoders unfrozen, original clinical heads fixed, CNN running statistics fixed, exhaustive replay and shared right-angle augmentation. Forty epochs: safe incumbent 15/0. Both real encoder gradients verified. |
| 207 | Unclipped-score audit of 166 frozen mobile heads: optimistic zero-FP single/union ceiling 16/40; stop monotonic calibration of this family. No encoding. |
| 208–209 | Restore repair, training-only loss audit, then 60 additional full-mobile epochs (100 total). Training binary EMA .140827 to .063965, stopped after 12 stale epochs; safe incumbent still 15/0. No promotion. |
| 210 | Matched full-mobile clinical-loss ablation, 40 epochs: safe incumbent 15/0; peak 28/+16. No promotion. |
| 211 | Training-only class-conditional worst-group reweighting over the exhaustive frozen-feature replay, 40 epochs: selected 14/0. Zero new encoder inputs. No promotion. |
| 212 | Combined full source-107, exhaustive original replay and field-93 supervision, one supervised head, 40 epochs in 4.226 seconds: selected 15/0. Every epoch checkpoint backed up; no new encoding. |
| 213–214 | Two targeted nitrogen-deficiency/Other versus Healthy heads, 40 epochs each, then 108 fixed new-admission veto policies: best 18/40 with zero new false Healthy, below 28/40. No new encoding and no candidate selected. |

The phase 198 veto has an [inactive, separately tested retention proposal](CLINICAL_RETENTION_PROPOSAL_20261028.md).
No active gate imports it and no authorization is assumed. Even that proposed
limited false-Healthy-to-Unsure correction would not solve Healthy recall:
87/223 remains below 157/223. Correct named disease decisions, correct Healthy
decisions and all numerical/fresh-acceptance gates must remain protected.

## Full-mobile convergence control: phases 208–209

Phase 206's aggregate loss did not distinguish binary underfit from clinical
preservation cost. Phase 209 first restores its exact final epoch-39 model,
optimizer and RNG, then audits binary, clinical and ranking components on 256
fixed, training-only weighted draws. It does not select a loss or threshold
from validation labels. If the unaugmented binary BCE is already at most .03,
it stops without more training.

If binary underfit remains, the declared control adds at most 80 epochs with
the same objective, supervision, crop aggregation and clinical-balanced
sampling. Encoder/head learning rates use a cosine multiplier .25 to .05 of
the original rates. It logs the three loss components separately, stops after
at least 24 extra epochs and 12 stale binary-loss EMA epochs, or immediately
when 28/40 Healthy and zero new false Healthy are reached. The latter is only
permission for complete restored-model verification, not export or promotion.
All epoch predictions and model-state digests are retained; full model
snapshots were planned at absolute epochs 79 and 119, plus best and rolling
full optimizer/RNG. The run stopped at epoch 99 after 60 additional epochs,
12 stale binary-EMA epochs, 292.644 seconds total. Binary training EMA fell
from .140827 to .063965. The selected safe incumbent is still 15/40, epoch -1;
the adapted models do not meet the screen. Only epoch 79's planned additional
full snapshot was reached. Longer training of this exact objective is closed.

## Matched clinical-loss ablation: phase 210

The remaining bounded hypothesis is that forcing the expert to preserve its
original fine-label geometry conflicts with learning the Healthy/diseased
boundary. Phase 210 restarts from the **same original encoder and phase-196
warm head as phase 206**, with its same data, seed, sampler, augmentations,
learning rates and 40-epoch budget. Only the clinical-loss multiplier changes
from .3 to zero; binary BCE and ranking .2 remain. This is a matched objective
ablation, not a continuation from a later favorable validation checkpoint.
It uses the repaired strict restore and batched SHA-verified pixel loading.
All original actual-verdict gates remain. Its conditionally routed expert
cannot overwrite original named-disease outputs. Save every epoch's prediction
and model digest, best and rolling optimizer/RNG, and a final full model
snapshot if reached. It completed 40 epochs in 222.637 seconds: safe incumbent
15/40, peak 28/40 with 16 new false Healthy, final sampled training loss
.069416. Removing this auxiliary term did not solve the safety/generalization
problem. The declared objective ablation is closed unsuccessful.

## Cached worst-group control: phase 211

Mean condition-balanced loss can still neglect hard training groups. A single
predeclared control applies an exponentiated loss adversary (step .01) to
training source-by-clinical-condition groups. Adversary weights normalize
separately within Healthy and diseased classes, retaining equal class mass;
within a minibatch, only observed groups contribute and their weights are
renormalized. This is an explicitly specified class-conditional variant of
[the original group-DRO formulation](https://github.com/kohpangwei/group_DRO),
not a claim of its guarantees for this dataset. It retains the phase-193
supervised control's full replay, cached features, warm head, seed, sampler,
optimizer and 40-epoch budget. No validation label updates the adversary.
No source/group labels are required at inference. Three tests cover hard-group
weighting, equal binary mass, evaluation/state restoration and unknown-group
rejection. Adversary state is included in checkpoint/restoration state.

All 40 epoch weights and predictions were saved. The selected result is only
14/40 Healthy at zero new false Healthy; no encoder inputs were recomputed.
The frozen-feature worst-group control is closed unsuccessful. Do not launch
a full-model version or coefficient grid on the strength of this failure.

## Combined-supervision coverage control: phase 212

One final cheap coverage gap remained: phase 139 used full source 107 with
bounded old anchors before field 93 existed; phase 193 used exhaustive old
replay and field 93 but excluded continued source-107 training. Phase 212
tests the combination, retaining target mass .5, exhaustive original mass .25,
and splitting the extra-source quarter equally between source 107 and field
93. It uses the same supervised head, warm ancestry, clinical-balanced
sampler, seed, optimizer and 40-epoch budget as phase 193. All source-107
images remain a single development-only publisher collection; no validation
or prospective data are reclassified as training. Existing real ROI bags,
rotations and input-identity caches are reused without new encoder inference.

The control completed in 4.226 seconds, selected 15/40 Healthy at zero new
false Healthy, and failed the unchanged screen. All forty epoch models, best
and final optimizer/RNG are independently backed up and hash-verified on the
laptop under `.toolchains/model-checkpoints/phase212/`; its committed receipt
lists 42 files. This combined-supervision objective is now closed unsuccessful.
It does not justify another full-encoder run or full restored-model evaluation.

## Targeted negative specialists: phases 213–214

The remaining parent-policy errors specifically involved nitrogen deficiency
and publisher-labeled Other. Instead of another all-disease average loss,
phase 213 fits those two condition-specific negative-versus-Healthy heads.
Each uses all applicable quarantined Healthy training cases and only its
selected negative condition; other diseases are excluded rather than relabeled
Healthy. Binary classes retain equal mass, with source/condition/family
balancing inside each. Raw BCE logits replace the inherited general-class
calibrator; they are not claimed as calibrated clinical certainty. All
features come from existing exact-identity caches.

Phase 214 tests only the declared single/joint new-admission vetoes at fixed
.5 and .95 thresholds on the 18 previously declared coverage-capable parents:
108 policies, no fitted validation cutoff. Main .95 admission and every
original guard/retention requirement remain. Existing Healthy and named
disease outputs cannot be overwritten. The best policy uses the Other head
at .95 and reaches 18/40 Healthy with zero new false Healthy, still below
28/40. No candidate was selected, exported or sent for full inference.
All 80 epoch models and both selected/final optimizer checkpoints are backed
up and SHA-verified on the laptop; the phase-213 receipt lists 84 files.
These specialized vetoes are closed unsuccessful under the current screen.

Phase 208 failed its restore preflight before taking any training step. The
shared partial-load assertion expected BatchNorm bookkeeping keys that
PyTorch's backward-compatibility loader suppresses. The new
`strict_trainable_restore.py` checks the exact trainable key set, shape and
dtype, merges those values into the current complete state, then performs a
strict full load. Two tests verify frozen parameters and BatchNorm running
buffers/bookkeeping are preserved and invalid states cannot partially mutate
the model. Historical executed-source bytes are preserved; the old helper is
not silently rewritten. Phase 209 applies that restore repair and improves
pixel I/O with six read-only CPU threads and batched GPU copies, keeping every
canonical input SHA check. It repeats pixel loading, not feature inference or
completed training. Phase 208's redundant untrained warm checkpoint was removed
only after proving all tensors equal the independently backed-up phase-206
warm best; its failed protocol and explicit execution-error note remain.
The current Git entrypoints for failed phases 204 and 208 now exit with an
explicit pointer to their completed repairs, preventing accidental repeated
encoding or restoration against the relocated parent. Their original exact
PC execution bytes remain in `executed_sources.zip`; those two guarded Git
files intentionally differ from the archived failed execution sources.

The initial phase-208 GPU pixel-loading process was briefly SIGSTOP-paused while the
parent checkpoint transferred to the laptop, then resumed after independent
SHA verification. No training restarted and already loaded pixels were kept.
The script deliberately rejects an implicit second execution; if interrupted,
restore its exact rolling checkpoint under an explicit continuation protocol.

## Checkpoint and cache preservation

Remote campaign root:
`/home/genesis/agribot-data/recognition_campaign_20261024`.
Python: `/home/genesis/.venvs/agribot-train/bin/python`.
GPU PC `agribot-5090` uses a dirty SCP overlay: **never git pull it**.

Phase 206 selected and final optimizer checkpoints have verified independent
laptop backups under `.toolchains/model-checkpoints/phase206/`:

- `best.pt`, 93,403,181 bytes, SHA-256
  `f83d840d6dbe8cdf62523809d53cada21de2a7d9e1de23abce47219440b93a76`.
- `last.pt`, 280,292,733 bytes, SHA-256
  `701e2749a461989e08d0d517cacd88e9cd893c05c33d9b956192447cd695b1b7`.

The original PC phase-206 final full model is also persistent as `epoch_039.pt`.
Phase 209 relocates the backed-up rolling RAM checkpoint into its own directory
before advancing optimizer state; it does not duplicate it on the nearly full
disk. RAM checkpoints are volatile. Preserve best/last and snapshots to laptop
storage before reboot or cleanup. The completed phase-209 and phase-210 best,
additional full snapshot, and final optimizer/RNG checkpoint files now all have
independent, SHA-verified laptop backups in their corresponding
`.toolchains/model-checkpoints/phase209/` and `phase210/` directories. Their
committed `checkpoint_backup.json` receipts identify every file and hash.
The private backup worker resumes partial transfers and verifies complete files
before marking them backed up; it does not change training or remove PC files.
The phase-181 independent laptop backups
remain. Its two RAM weight copies were removed only after rechecking both
local and remote SHAs, reclaiming 407 MB for phase 210. Small metadata and
predictions were copied to its persistent preflight directory first. Restore
those weights from the laptop before any future phase-181 model load; do not
assume the old RAM path still contains them.

Phase 191 persisted the exact frozen prefix: FP32
`2106 × 1370 × 1024`, 11,817,861,120 raw tensor bytes. Its feature-file hash,
prefix-state identity and input order are in the cache completion/protocol
files. Phase 199 reuses it and computes only the 142 additional inputs. The
phase-191 protocol inherited two stale prose fields about a 6×6 local grid and
prefix persistence; `execution_notes.json` explicitly corrects those fields.
Actual code uses all 1,369 adapted local tokens and persists the base prefix.

The original phase-168 and exhaustive phase-193 row namespaces differ.
`exhaustive_replay_data.py` preserves all 2,062 source-107 row positions before
field rows, even though continued source-107 training mass is zero. Do not join
field row indices between those namespaces. Match input SHA, crop metadata,
encoder weights and preprocessing before reusing any feature or prediction.

PC disk headroom is about 1.1 GiB. A normal pip download-cache purge recovered
12.58 GB; no installed package or training cache was deleted. A task-owned
Android build was archived with CRC verification before `gradlew clean`:
`/home/genesis/agribot-data/android_build_before_clean_20261028.zip`,
223,578,018 bytes, SHA-256
`5bd318586ad8ad13d23c90b5ce4101555d4dbe06d039da771b4874e2457ca56d`.
Laptop delivery APKs are unchanged. UV cache pruning was cancelled when it
reported an in-use cache; no lock was forced and unrelated jobs were preserved.

## Software and acceptance evidence

The expanded local Python suite passes **300 tests**, with
one optional downloaded-media skip. Native/app source and bundled models did not change in
this continuation. Published `72fc345` CI passed 300 training contracts and
the Android job; the final additional cached controls require their own
publication CI result.
Earlier 133 native units, 22 selected emulator checks and actual minified ARM64
release smoke remain software/runtime evidence, not independent recognition.

The evidence directory contains hash-bound protocols/results, the explicit
phase-204 failure, checkpoint-backup receipts, normalization and convergence
audits, and an archive of exact PC source bytes (including the inactive tested
proposal). Arrays, images, weights and APKs are intentionally not committed.
The 32 new protocols are stored losslessly in
`extended_protocols_183_214.zip`, indexed by path, SHA-256 and byte size in the
adjacent JSON manifest. This preserves 15,693,235 exact raw protocol bytes
without adding hundreds of thousands of generated protocol diff lines.
Results retain their original raw protocol hashes. Three additional software
tests verify the source archive, compressed protocol identities and result
bindings. Do not use phase-204's invalid prediction file or present historical
datasets as fresh acceptance.

## Next decisions and stopping criteria

An eight-case clinical-review preparation packet is available privately under
`.toolchains/recognition-data-preview/clinical-review-20261029/`. The reviewer
ZIP preserves original image bytes and supplies a blank review form; publisher
labels and provenance remain in a separate mapping. These are deliberately
selected, previously inspected development cases, not an accuracy sample or
fresh acceptance. No reviewer was contacted, no clinical adjudication occurred,
and no labels, denominators or gates changed. Record uncertainty and required
plant context rather than forcing nutrient diagnoses from ambiguous RGB leaves.

1. Phases 209–214 are closed unsuccessful. Do not restart them or add
   epochs merely to accumulate checkpoints. Preserve the evidence and exact
   caches for an independently justified next hypothesis.
2. Only a candidate passing the cheaper actual-verdict screen may receive full
   restored-model evaluation under all original checks, including
   `original_disease_and_healthy_exact`. Never remove a failed check.
3. Fresh acceptance must have a documented protocol and frozen candidate before
   opening labels. Public dataset metadata discovery is not acceptance. The
   DigiGreen farmer-photo card was inspected only as a possible future source;
   no images or annotation tables were downloaded, and independence is unknown.
   Mendeley `jttrv2w27r`, `xc94xbg239` and `rnbsw72zb5` are already used sources;
   do not redownload them as independent data. Coffee/pepper nutrient photos and
   tomato spectroscopy are not valid replacements for tomato RGB leaf labels.
4. If representation controls remain below the gate, prioritize additional
   clinically verified tomato nutrient/Healthy supervision and prospective
   acceptance over another validation-tuned threshold or reward sweep.
   Existing source labels and family fingerprints cannot establish clinical
   truth by themselves.
5. Export, app model replacement, physical-phone parity/latency and signed
   owner release delivery remain downstream of qualification. Do not call the
   model repair or PR complete while those requirements are unresolved.
