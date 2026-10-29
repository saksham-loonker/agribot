# Historical development pass: subsequent acceptance failed

The frozen candidate subsequently failed OLID acceptance at 87/223 Healthy,
with zero gain. It is not promoted. AD and OLID are now consumed development.
Read [the current continuation](FIELD_ACCEPTANCE_20261027.md) first.

The phase-69 route passes every unchanged development recognition gate. This
is a development qualification, not independent acceptance or release approval.
At this development checkpoint AD was consumed and OLID was still sealed;
the subsequent failed opening supersedes that status.

The [condensed evidence](evidence/qualified_development_evidence_20261027.json)
pins completed controls, protocols, checkpoints, caches and the frozen shared
models. Full artifacts remain under
`/home/genesis/agribot-data/recognition_campaign_20261024` on `agribot-5090`.
Historical [app/recognition checkpoint](APP_RECOGNITION_CHECKPOINT_20261027.md)
and [app verification](evidence/app_verification_20261027/summary.json) remain valid
for the earlier app 2.0.3 candidate retaining the interim model.

## What passed

The selected auxiliary DINO checkpoint is the validation-selected epoch 11 from
the bounded continuation. It retains the original frozen first ten blocks and
the original deployed classifier separately. The earlier seven-epoch checkpoint
and all rejected alternatives remain preserved.

The new routing rule retains the original verdict when the adapted recognizer
makes a negative binary Healthy decision on the whole primary image and on the
original center view of at least two distinct detected leaves. The binary
boundary is 0.5 in each scope. Every preceding guard and safety requirement is
unchanged. This is one frozen corroboration policy, not a threshold sweep or an
exception keyed to a photo, source, condition label or filename.

| Healthy development cohort | Correct / total |
| --- | --- |
| Original TOM | 97 / 129 |
| Original paper | 95 / 119 |
| Internal TOM | 39 / 53 |
| Consumed AD development | 22 / 29 |

All original retention, coarse-disease, internal retention and per-source Healthy
checks pass. Every original named disease verdict is preserved. No old or
internal retention failures remain. Whole-image corroboration was required for
only two proposed Healthy images: one canonical score was reused and one new
input computed in 0.83 seconds. Fresh acceptance data was not read.

## Bounded alternatives completed first

- Phase 54: strong whole/leaf negative abstention finds no eligible original
  Healthy cases. Zero encoder inference; the unsupported regression remains.
- Phases 55–56: continue the saved DINO model, EMA, optimizer and RNG rather
  than restart fitting. Stop after seven additional epochs and three stale
  validation epochs; select epoch 11. Training/evaluation takes 77.55 seconds.
  Healthy improves, but unsupported acceptance still fails by one photo.
- Phases 57–58: one supervised family-weighted upper-tail negative-risk control.
  Stop by the frozen validation rule; the unsupported gate still fails.
- Phases 59–60: one separate explicit-Other condition-supervision control.
  Exactly 365 properly scoped named unsupported targets gain categorical Other
  supervision. Generic, coarse, missing and conflicting conditions stay masked.
  Four epochs, 32.50 seconds; the same unsupported gate fails.
- Phase 61: positive binary leaf-majority agreement reduces Healthy coverage
  and fails to remove the unsupported regression. Stop.
- Phases 62–63 and 65–66: linear and nonlinear classifiers for explicit
  unsupported tomato conditions versus Healthy, at primary-image scope. Known
  diseases, non-tomato, mixed taxonomy, generic disease and every ROI are
  excluded. Both use existing frozen SigLIP features with zero new encoders.
  Both fail the unchanged recognition gate and stop.
- Phases 67–68: one last-two-block SigLIP adaptation for that same primary task.
  Original FP32 prefix tokens are cached and checked against original embeddings.
  Four epochs, 18.62 seconds after cache preparation, exact restored-checkpoint
  probability parity. Reuse 179/184 proposal prefixes and compute five new
  prefixes. The unsupported gate remains failed; this model is stopped.

Sparse prediction now evaluates only the 643 label-blind auxiliary inputs used
by the fixed route. Per-control new evaluation-prefix computation falls from
13,841 inputs to 641, retaining all original cohort cases. Unscored probabilities
remain NaN and are forbidden from routing; indices and arrays are hash checked.
No cached score crosses a changed model identity. The remaining large shared
caches stay intact.

The failure audit confirms a labelled unsupported curled-leaf image is called
Healthy by several experts. Its original verdict is retained by the passing
corroboration policy. The audit does not relabel the image, train on its outcome,
change any safety limit or create an image-specific runtime exception.

## Frozen shared models and verification

`phase70_qualified/construction.json` pins the phase-69 policy and every included
checkpoint. Only the passing route's components are packaged. Rejected
explicit-Other/risk/primary-condition model weights are excluded.

- `leaf.pt`, 108,502,389 bytes, SHA-256
  `d32323eecdcabc5533348256487909ed5911052d0b2bb3b94be8c99718d21c04`:
  original nine classifier logits plus compact positive/negative Healthy and
  adapted-DINO Healthy. Both original/adapted DINO branches share the frozen
  prefix; original CNN features are reused.
- `whole.pt`, 457,675,115 bytes, SHA-256
  `ebf6116a0f27b1da711fa4e6c2af5c19e896a2940a425aabbdadb548477e3f2b`:
  positive Healthy, negative-guard Healthy, tomato genus and capture context.
  Original and adapted SigLIP branches share their unchanged prefix.

`phase71_restored_verify.py` performs the next gate. It compares actual restored
shared-model outputs with the identity-matched cached component outputs on all
leaf views used by the development cohorts, every opened whole image and every
required whole-DINO input. It also checks baseline classifier replay and exact
candidate verdict kinds, labels, counts and scopes, then re-runs every unchanged
recognition gate. Its batch caches are resumable only under the same frozen
weights, inputs, preprocessing, source and runtime identities. Inspect
`phase71_runtime.log` and its progress files before resuming; never restart a
useful job blindly. Restored verification passed in 51.70 seconds across 14,079
leaf inputs, 6,094 whole images and two corroboration inputs, with all 6,742
verdicts matching. Maximum original-classifier logit error is `2.24e-6`.

Both FP32 LiteRT models export with built-in operators only and pass boundary
cases. Full compiled-model replay also passes: all 6,742 verdicts and every gate
match, in 427.45 seconds. Maximum original-classifier logit error is `0.0002322`;
maximum whole-image probability error is below `0.0000373`. Eight-thread PC
inference measured 10.76 ms per leaf model and 38.31 ms per whole-image model;
sampled results are bit-identical to two-thread inference. These are PC timings.
The no-leaf original-classifier fallback also matches on all 34 potentially
recoverable no-leaf inputs, with zero probability error.

Android schema 5 and 6,742 scalar-policy fixtures are frozen in
`phase74_native_contract`. The Kotlin port and actual-frame accumulation are
under verification. Seven distinct admitted timestamps are required for primary
recovery; duplicate callbacks cannot create observations. Original disease
verdicts take priority, and primary-view scope is retained on saved results.
Walk mode excludes off-plant or untracked primary views. The new model assets
have not yet been integrated in the published app.

Port the pure scalar policy into the app and prove native replay parity.
Research static replay counts are not live camera
observations: the app must accumulate actual admitted frames and retain actual
leaf counts, including zero for primary-image-only evidence. Primary-image
supervision does not establish that every leaf or the whole plant is healthy.

Freeze the final candidate bytes and fresh-acceptance protocol before opening
OLID. Prefer completing export/runtime parity first so a teacher prototype does
not consume the final deployment candidate's independent evidence. Independent
acceptance and eventual physical-phone requirements still gate promotion.

## App, CI and operating state

App persistence/lifecycle fixes and the earlier audited adaptation work are
published at `b3074155ba50a601b4c2ff35c518ef43eb1c019c` on PR #6. Both CI jobs
pass, including the new emulator suite. Local verification: debug and unsigned
release builds, zero lint errors, 20 selected emulator tests without skips,
124 collected passing native unit reports. The expanded research suite now runs
179 tests: 178 pass and one optional downloaded-media test skips.

Local APK copies remain under `.toolchains/app-candidates/2.0.3`; their model is
the unchanged interim classifier. No new model is integrated yet. The phone is
disconnected; the API-35 emulator is available. PR #6 remains draft.

Cache preparation left only 2.4 GB usable PC disk space. The ext4 system reserve
was reversibly reduced from 5% (46.52 GiB) to 2% (18.61 GiB), producing 31 GB
usable space while preserving all data and caches. The original reserved-block
count is 12,195,737; the current count is 4,878,296 on `/dev/nvme0n1p2`. The private
audit `.toolchains/pc_storage_reserve_20261027.json` records the change and exact
rollback value. No password is stored in that record or committed helper code.

The four-hour PR-only monitor remains enabled. The broader active user request
is to finish app and recognition together; scheduled messages do not replace
that objective. Keep useful jobs running and compare GitHub changes before
adopting anything. Never merge the PR automatically or pull over dirty work.
