# Recognition repair: auxiliary supervision and missing training coverage

## Release status

No replacement model is qualified. The best fixed consumed-development policy
remains **19/40 Healthy, zero new false Healthy**; the unchanged screening gate
requires **28/40**. Keep PR #6 draft and retain the normal Android 2.0.4 model.
AD and OLID remain consumed development. The provisional TCP reservation remains
diagnostically sealed; only geometric identities were used for training-overlap
screening. No new independent acceptance claim is made.

This follows [phases 220–225](FISHER_BILINEAR_SPATIAL_REPAIR_20261029.md).
The relevant evidence is in
[attribution controls](evidence/attribution_controls_20261029/) and
[auxiliary controls](evidence/auxiliary_controls_20261029/), plus
[whole-photo coverage controls](evidence/coverage_controls_20261029/).

## What is actually failing

The verified current preprocessing does not reproduce the suspected 0–1 versus
0–255 normalization error. New auxiliary feature parity reached a maximum
absolute error of 2.15e-6; original/cached whole-SigLIP prefix restoration reached
1.91e-5. These are specific checks of these paths, not a blanket certification
of every historical training script.

Nine of the forty validation Healthy photos are blocked by the unchanged
whole-Healthy support guard below .85. Thirteen are retained Healthy already.
That leaves eighteen eligible rejected Healthy photos, of which fifteen must
be recovered to pass 28/40, with no new false Healthy on any screened source.
The permissive full-cohort baseline remains 87/223 Healthy and 16/507 false
Healthy. It is not replaced by the 19/40 development-policy number.

The new training-coverage audit found another concrete issue: the ROI learners
cannot train on photos without detected-leaf feature bags. Within the nominal
OLID training partition, 20/183 Healthy, 15/37 Nitrogen Deficiency, 8/28
Pottassium Deficiency, 125/137 Leaf Miner, 18/37 Diseased_unspecified and 48/170
Other photos were omitted for that reason. The audit found no unexplained
omissions in these classes. They were not excluded by validation quarantine.
However, absent cached ROIs are not evidence that a validation image was
misclassified, and adding this supervision has not yet qualified a model.

This coverage issue also affected earlier whole/ROI fusion: phase182 still
required ROI bags. Earlier phase85 whole-encoder adaptation used an inference-
eligible population; the 20 Healthy and 15 Nitrogen Deficiency omissions were
absent from its exact prefix cache. Whole-photo supervision can legitimately
include these training photos without assigning photo labels to individual
leaves or weakening inference guards.

Persistent Healthy/stress confusion is supported by the actual verdicts below.
Its deeper clinical cause is not established: publisher labels have not been
independently adjudicated, and model or VLM disagreement does not establish
label error. Do not relabel the data from these predictions.

## Completed bounded controls

| Phase | Change | Actual consumed-validation result |
|---|---|---|
| 226 | Attribution and guard/crop analysis | Diagnostic only; nine guard-blocked Healthy confirmed |
| 227 | Native paired-view linear/RBF features | Both 13/40; zero new false Healthy |
| 228 | Matched ERM, same-label cross-source interpolation, spectral penalty | All 13/40; zero new false Healthy |
| 229 | Domain-conditional stacks of five existing OOF experts | All 13/40; one setting adds a TLID false Healthy |
| 230–231 | Entire expert-labelled cucumber collection as auxiliary training only | 247 unique new feature inputs; zero detected geometric overlaps |
| 232–233 | General and nutrition-only negative heads; original-domain recalibration | All 13/40; zero new false Healthy |
| 234 | 288 fixed auxiliary-veto policies | No pass; best safe 19/40 |
| 235 | Audit of missing fallback scores | Initial interpretation invalidated; see correction below |
| 236 | Actual original classifier on whole-image canonical inputs | 13/40; zero new false Healthy; no new model weights |
| 237–238 | Exact prefix cache; frozen, rank8 Q/V LoRA, copied full suffix | All 13/40; zero new false Healthy |
| 239 | 216 fixed encoder-veto policies | No pass; best safe 19/40 |
| 240–241 | Fixed local Gemma4 visual negative audit; 72 cached policies | No pass; best safe 19/40; no positive VLM branch |
| 242 | Explain missing nominal training supervision | Every omitted class member above lacked a cached detected ROI |
| 243 | All-photo whole-SigLIP linear/RBF supervision, including no-ROI photos | Linear 14/40 with zero new false Healthy; RBF 17/40 with one |
| 244 | 12 fixed combinations with the prior safe policy | No pass; best safe 19/40 |
| 245 | All-photo whole-feature MLP, binary versus clinical auxiliary loss | Both 13/40; zero new false Healthy |
| 246–247 | Restore missing whole prefixes; matched frozen/copied whole-SigLIP suffix | Both 13/40; zero new false Healthy |
| 248 | One training-only target-scope calibration per six fixed whole experts; 36 policies | No pass; best safe 19/40 |

All calibrations described as OOF use cold training fits with grouped held
families/identities. They are development controls, not new acceptance tests.
The auxiliary collection has 249 raw files, 248 unique-file photos and 247
unique canonical input pairs. Its classes are 33 unique-file Healthy photos and
215 nutrient-stress photos after exact-file deduplication. It is never treated
as tomato disease-class supervision or independent validation.

The local VLM audit used the already-installed Gemma4 model, a frozen prompt,
deterministic options and exact image/prompt/model cache keys. It made 71 local
requests in 98.23 seconds: 28 Healthy and 43 VisibleStress predictions. These
predictions are machine opinions, not clinical gold labels. No cloud requests,
model downloads or new positive Healthy branch were introduced.

## Interpretation corrections and format repair

Phase235 initially treated absent `packet.fallback` as failed whole-classifier
recognition. That was wrong: fallback is computed only when no detector boxes
exist. All 730 consumed OLID packets had ROIs. Its corrected result explicitly
marks whole-classifier accuracy **not evaluable from fallback availability**.
The invalid source/protocol/result and invalidation record are preserved in
`invalid_initial_attribution/`. No training or promotion depended on that error.
Phase236 then performed the real original-model whole-photo probe instead of
inventing scores for missing fields.

Phase233 had a preflight serialization comparison fail because Python-scalar
float32 multiplication and restored float64 calibration were compared with
overstrict equality. The corrected path explicitly computes float64 margins;
the failed preflight source/protocol is preserved. Only cheap calibration was
repeated, not training.

The new phase246 prefix writer initially passed a list-shaped `shape` into
NumPy `open_memmap`. NumPy wrote that header but its reader rejected it.
Before any phase247 training, the header delimiters were repaired to a tuple;
the entire payload SHA was verified unchanged. The original header/source,
completion metadata and phase247 preflight traceback are retained, with a
repair receipt. The current writer passes a tuple. No encoding was repeated.

## Current adaptation and stopping decision

Phase246 prepared a bounded whole-SigLIP cohort containing **all quarantined
OLID training photos**, including the no-ROI omissions, plus old replay whose
exact whole-prefix identities already existed. This is explicitly bounded old
replay, not full original replay. It reused 1,487 prefixes and computed only
234 missing prefixes in 28.03 seconds. All restored features agree with their
original cached features to at most 1.91e-5.

Phase247 compares a matched frozen head with copied last-two-block, norm and
pool adaptation. The original prefix and deployed classifier remain immutable.
Each method has four cold grouped OOF fits and a full fit, fixed 40 epochs,
1,024 draws/epoch, batch16, head LR .001 and suffix LR .00001. Training-only
OOF Platt calibration uses C1. Every tenth full-fit snapshot and each fit's
final optimizer/RNG state are retained. Validation does not select epochs.

Phase247 **completed in 239.73 seconds**. Both matched controls scored13/40
with zero new false Healthy. All ten fits completed; eight full-fit intermediate
snapshots and ten final optimizer/RNG states were preserved. No full restored
model qualification or export followed this failure.

Phase248 then fitted exactly one target-scope Platt calibration per fixed
whole-photo expert using only grouped-OOF margins from OLID **training** rows.
Its36 fixed combinations also failed. As a diagnostic only, even a threshold
placed just beyond each expert's largest consumed-validation negative margin
would reach at most22/40 when OR-ed with the incumbent19/40. This is an
optimistic, label-consuming ceiling for that monotone score family, not a
deployable threshold, candidate metric or a proof about all possible models.
It explains why rescaling these particular logits cannot close the gate.

This whole-photo family is stopped. Do not restart completed fits, vary rates
or lower the threshold around these same consumed cases. Only a passing
candidate may proceed to the full cached source/family gates and then restored
full-model verification.

No amount of extra iterations guarantees a pass. If this direct coverage repair
also fails, the next data intervention needs a specific, audited rationale:
expert adjudication of the existing blinded review packet and training-only
field examples that distinguish the observed pale Healthy and nutrient-stress
phenotypes. Do not unseal fresh acceptance to manufacture that rationale.

## Preservation, validation and operational details

All phase226–229 payloads were independently backed up on the laptop and each
member SHA verified: 170 artifacts, 187,668,480-byte TAR, SHA256
`cc76b21b4e9e0d2654d675b8e89054b8ca58dde1fb62e1f3902f62a05918e5b3`.
Phase231–243 similarly has 228 verified artifacts in a 493,168,640-byte TAR,
SHA256 `3fdf9dab2462156c9b53db5359d2ce1effeb73dc628c6837ce8107dde105343f`.
These private archives live under `.toolchains/model-checkpoints/`; no model
weights, source photos or input packets belong in Git. Later artifacts and
volatile token caches need their own receipts; the 231–243 receipt does not
cover phases244 onward or external token storage.

The separate phase244–248 archive transfer is underway; until its member-SHA
receipt exists, retain every corresponding PC checkpoint. Do not describe
that transfer as a verified independent backup. Source/result manifests and
exact executed protocols are already recorded separately.

After the phase231–243 backup was verified,129 phase232/238 PT files
(461,392,283 bytes) were removed from the PC to preserve the active phase247
fit's disk reserve. Every removed file was rehashed against the verified
archive manifest first; the explicit offload receipt lists all paths. Restore
these files from the laptop archive before future loading. No phase247 state
was offloaded.

Two historical phase13 optimizer files were offloaded only after complete
independent SHA verification, freeing approximately 2.23 GB without removing
best models, image data or prediction caches. Preserve those laptop backups.
The PC filesystem still has little reserve. Use `TMPDIR=/dev/shm` for Python
imports, retain useful tmux work and never pull into its dirty training overlay.
New bounded prefix caches are under `/run/user/1000`, which is volatile across
reboots. A prefix cache can be recomputed from its pinned inputs if necessary;
its final pooled features cannot substitute for patch tokens during adaptation.

The software suite passed **347 tests with one optional-media skip**, including
the new coverage-isolation and evidence-integrity checks. Native Android CI at published commit
7fe54e7 already passed unit/lint/build and all22 emulator checks. No Android
source or model artifact was changed by this continuation. Physical-phone and
owner-signing work remain pending; the phone is disconnected.

Measured new training times include 33.17 s for phase228, 184.21 s for phase238,
7.48 s for phase243 and 10.87 s for phase245. Data acquisition/preparation and
independent backup transfers dominate longer wall-clock intervals. Do not turn
these timings into a promised completion date: recognition still has no passing
candidate. Estimate phase247 from its actual completed folds/checkpoint ages.
