# Offline recognition continuation: corrected cache joins and stopped controls

**23:00 app update:** [the normal 2.0.4 app is built and verified](NORMAL_APP_DELIVERY_20261027.md)
with the original release-baseline model. Its source is now publishable with
that baseline; rejected replacement assets stay outside the normal app bundle.

27 October 2026, 20:30 IST. **Recognition is still not qualified. PR #6 remains
draft.** No new model was exported, integrated or installed during this
continuation. The phone is disconnected. Preserve unrelated app edits and all
completed evidence. This extends [the failed prospective acceptance handoff](FIELD_ACCEPTANCE_20261027.md).

## Outcome and unchanged requirements

The original frozen prospective result remains 87/223 Healthy and 16/507
disease-to-Healthy errors. The candidate failed Healthy recall and gain; that
result is immutable. AD and OLID are permanently consumed development. Neither
is independent acceptance, including their grouped validation partitions.

The repaired local Android app still has the previously recorded 133 passing
native unit checks, 22 passing API-35 emulator checks, successful builds and
zero-error release lint. Those checks were not redundantly repeated for these
research-only Python changes. Research APKs remain rejected for release.

The original recognition requirements remain: source/class integer retention,
Healthy recall at least 70%, gain at least ten percentage points with a positive
paired-family gain interval, original named disease verdicts protected, and no
increase in disease-to-Healthy point risk with one-sided family-bootstrap upper
bound at most two percentage points. Actual app verdicts decide qualification;
an encoder AUC or a promising training loss cannot substitute.

## Important cache correction

The phase-87 frozen DINO/CNN feature cache correctly contains 43,115 original
inputs followed by 854 new ROI inputs. The older DINO token cache numbers an
extended input list that also contains scene inputs. Indices 43,115 and above
therefore refer to different inputs in the two caches.

The first token preparer incorrectly joined those namespaces by integer index.
Its sampled initialization audit covered only old rows. **Phase 89 and phase 91
training/evaluation results are invalid**, not evidence that their methods
failed. Their original files and results are retained, with explicit
`INVALIDATED_INPUT_IDENTITY.json` records on the PC. Do not resume, select or
promote either checkpoint. The original prospective failure, phase-87 frozen
features and phase-88 heads do not use that faulty join and are unaffected.
This new-experiment bug does not explain the original deployed Healthy failure.

`input_identity_join.py` now joins canonical input SHA-256 values across
independently numbered caches. `diagnostic_bag_cache_v2.py` reuses the valid
original tokens and the genuinely computed missing-token rows from phase 89;
it computes only the 854 absent inputs. Before fitting, **every one of the
2,626 selected tokens** must reconstruct its corresponding original embedding,
including every new input. Maximum error was `0.00000477`, below `0.002`.
Positional reuse is forbidden. Protocols pin descriptor, input, source, model
and cache identities. The CNN control uses the same hash join and exhaustive
reconstruction requirement.

Regression checks exercise the actual namespace collision, reordered/sparse
inputs and malformed identities. The routing-bound check is compared against
an exhaustive search on tied/zero scores and protected cases. A CNN check
verifies copied inference parity, immutable original weights and frozen
BatchNorm statistics after optimization.

## Completed bounded controls

All numbers below are **consumed development**, never fresh acceptance. The
fixed family validation partition has 40 Healthy images; it needs at least
28 Healthy verdicts. Full consumed OLID needs at least 157/223. Safety includes
paired-family uncertainty, so an unchanged aggregate false-Healthy count alone
does not establish a pass.

| Control | Validation Healthy | Full Healthy | Full diseased called Healthy | Decision |
| --- | ---: | ---: | ---: | --- |
| Phase 88, frozen original DINO/CNN linear and MLP heads | 13/40 each | 87/223 each | 16/507 each | Stop; Healthy gate fails |
| Phase 94, corrected copied DINO suffix plus binary head | 13/40 | 87/223 | 16/507 | Stop; Healthy gate fails |
| Phase 95, continue previously qualified Healthy expert | 17/40 | 104/223 | 16/507 | Capped while improving; resumed once as phase 97 |
| Phase 97, same checkpoint/optimizer/RNG continuation | 21/40 | 115/223 | 17/507 | Stop; Healthy and full safety fail |
| Phase 99, cached-kernel logit-verdict reward | 16/40 | 110/223 | 18/507 | Stop; Healthy, full safety and retention fail |
| Phase 101, copied original spatial CNN suffix | 17/40 | 118/223 | 15/507 | Stop; Healthy and validation family-safety fail |

Phase 94 selected epoch 1. Phase 95 selected its final allowed epoch 11, making
one continuation justified. Phase 97 selected epoch 16, then stopped at epoch
20 after four stale routed-validation evaluations. It retained optimizer and
random state; it did not restart training. Phase 101 selected epoch 6 and
stopped at epoch 10 after four stale evaluations. Its original deployed CNN,
disease path and BatchNorm running statistics stay frozen.

Phase 95/97/101 selection uses actual routed validation verdicts: first prevent
source false-Healthy point increases, then maximize OLID Healthy verdicts, then
old Healthy verdicts. Full source/class retention and paired-family safety are
checked afterward; their failures still reject the candidate. No selection rule
may silently replace those final gates.

The phase-99 reward is a differentiable **supervised** objective over existing
labels, not clinical RLHF or a claim of new human feedback. It models new Healthy
admission at the frozen `.95` gate and abstention from original Healthy at
`.001`, with projected training recall/risk penalties. The training risk is for
the source-balanced conditional routing population, not the full cohort's
disease prevalence. Actual full-cohort app risk is still decisive. It stopped
after 250 steps. This unsuccessful reward must not be expanded into an
unbounded reward/temperature search.

Phase 92 projects the original CNN's Healthy-versus-disease direction on cached
unit-normalized embeddings. Those scores are explicitly not the original
CNN posterior because feature norms were discarded. Phase 93 fits one RBF
kernel classifier using equal unit-normalized DINO, CNN and original whole-image
SigLIP features. Its bandwidth comes only from training distances, with one
fixed regularization. Both use zero new encoder inputs.

## Cached stopping evidence

Phase 96 computes an exact optimistic two-threshold bound on eight completed,
valid score families. It excludes the invalid phase-89/91 runs. It uses consumed
labels **only to rule out these score families**, never to select a release
policy. It ignores the confidence floor, old Healthy retention and bootstrap
constraints, making its task easier than real acceptance. Global and applicable
per-condition false-Healthy point counts cannot exceed the baseline.

None reaches 157 Healthy images. Best optimistic Healthy counts are 107 for
the frozen MLP, 122 for corrected DINO SFT, 130 for the corrected continued
expert, 129 for its resumed checkpoint, and 125 for both kernel and logit-reward
scores. The standalone CNN reaches 139. These optimistic operating points are
not qualified policies and must not be exported.

Phase 102 checks exactly two prespecified equal-log-odds blends. CNN plus DINO
has optimistic maximum 144/223; CNN plus kernel has maximum 142/223. Both fail
before the harder recognition gates. No blend coefficients were fitted and no
encoder inference was repeated. Stop these two blends; do not continue a
threshold or coefficient hunt on the consumed labels.

## Prospective data and the next justified work

### Additional native-resolution control: stopped

An audit found that the 194 newly cached Healthy center ROIs have native short
sides of at least 501 pixels, with median 830. This justified one 448-pixel
spatial CNN control before attributing the failure solely to insufficient image
detail or labels. Phase 104 creates canonical 448-pixel inputs directly from
original files, validates their corresponding 224-pixel parent geometries, and
reuses only cached detector outputs. It never upsamples the cached 224-pixel
tensors or incorrectly reuses their encoder features.

All 2,626 new input identities are frozen. One original CNN forward per new
input captures both its prefix and original logits; every stored prefix must
reconstruct those logits within 0.002. A separate shape-checked hash helper
supports 448-pixel tensors, preserving the legacy 224-pixel helper. An initial
preflight caught that old helper's intentional shape guard before any training;
there was only one actual training run.

Training retains the original family fold and prior duplicate-group quarantine,
uses primary-image bag labels, protects the original classifier, and freezes
BatchNorm running statistics. Validation selection is stricter: it first
prevents every newly Healthy diseased validation image, even when other old
errors are removed. The final original recognition and family-safety gates are
unchanged.

The best checkpoint is epoch 1. It gives 13/40 validation Healthy, 89/223 full
consumed Healthy, and 15/507 false Healthy. Old retention and safety pass, but
Healthy recovery fails. It stopped at epoch 5 after four stale evaluations.
Total recorded setup, cache generation, training and evaluation time was 114.79
seconds. Its exact optimistic cached routing bound is 114/223, below the 157
required even before harder confidence/retention/uncertainty constraints. No
larger-input model is exported or integrated. Keep this control stopped.

[Evidence](evidence/field_candidate_20261027/high_resolution_control_20261027.json)
pins the native-resolution audit, all-input reconstruction, frozen protocols,
checkpoint, final verdict metrics, history and routing bound. The normal app
continues using its original 224-pixel classifier.

The TCP v4 publisher archive remains verified by SHA-256. Only the three
explicit tomato **leaf** folders were extracted; fruit categories were excluded.
Source-label, native-resolution, exact/near-copy, contradictory-label,
SIFT/homography family and prior-overlap screens are complete. Prior comparison
includes consumed AD and OLID, every original ROI source-file identity, and
canonical ROI inputs. It uses 33,135 old file identities and 36,658 old pHashes.

Of 3,668 leaf images, 2,527 remain provisionally sealed: 811 Healthy from 588
inferred families and 1,716 unsupported-disease images from 1,500 inferred
families. They have received **no diagnostic predictions and no training**.
The publisher mixes field, internet and other datasets; overlap screening alone
cannot establish capture independence. `acceptance_ready=false` and
`independence_established=false` are explicit. Do not open its predictions until
the candidate and protocol are frozen and source suitability is established.
Neither filename families nor SIFT components are verified plant IDs.

The separately downloaded `bpfd9cns5g` source is not SLIF. Its previous audit
inspected twelve fixed annotation overlays, recorded under `phase6_leaf_audit`.
It remains forbidden for detector training because its boxes mix spots and
whole leaves and the class-ID mapping is missing. Its 731 raw photos must not
be confused with 1,621 annotated instances. Do not call the entire source
untouched. A future remainder would require a verified publisher class mapping,
label-scope audit, and quarantine of inspected/overlapping capture families.

The next substantive input is trustworthy fine-grained Healthy-versus-damaged
leaf supervision, especially subtle mite/miner/deficiency examples, and a
genuinely unconsumed suitable acceptance reservation. New validation protocol
selection should incorporate family uncertainty directly, not just source point
counts. Do not weaken an original gate, fabricate per-leaf clinical labels from
image labels, restart these stopped controls, or use positive-logit inflation
as a substitute for damage evidence. Full retraining is not established as
necessary by these results; additional epochs on the stopped checkpoints are
not justified either.

Only a candidate passing cheap complete development checks should receive full
restored-model verification, mobile conversion and Android integration. Freeze
the candidate, app source and protocol before opening suitable fresh data once.
Physical-phone installation and latency/memory checks follow qualification.

## Reproduction, evidence and measured estimates

Use `ssh agribot-5090`; Python is
`/home/genesis/.venvs/agribot-train/bin/python`. Research code is under
`/home/genesis/agribot/model_training/v3`; artifacts are under
`/home/genesis/agribot-data/recognition_campaign_20261024` (`P`).
The remote source and native checkouts are dirty overlays: never blindly pull
over them. Persistent tmux jobs completed; no useful worker was interrupted.

[The compact evidence index](evidence/field_candidate_20261027/offline_controls_20261027.json)
pins PC artifact paths, SHA-256 values, original invalidations, per-gate results,
prefix audits, stopping bounds and prospective-source status. The earlier
native build/emulator artifacts remain alongside it. Preserve full raw PC
protocols, checkpoints, optimizer/RNG state and score caches.

Measured complete control times, including their recorded setup/evaluation,
were approximately 22 seconds for phase 95, 20 seconds for its phase-97
continuation, 10 seconds for the logit-reward control and 49 seconds for CNN
adaptation. The RBF control took 40 seconds with zero encoder inference.
The prior full prospective replay took approximately three minutes; the full
native build took 67 seconds and emulator checks 57 seconds. These are
individual PC measurements, not a model-delivery estimate. A reliable completion
ETA is not available while recognition and fresh-source suitability still fail.
There is no useful training job to leave running merely to occupy the GPU.

Publish research source/evidence and the verified normal app source with its
original model. Keep rejected replacement assets preserved separately for
research. PR #6 may leave draft only after all remaining gates pass;
never auto-merge it. Continue the existing four-hour PR monitoring task until
the PR closes or the user cancels it.
