# Cache-first training and recognition decision handoff

**Resumed work and code review:** [PR 6 review and supervised continuation](PR6_CODE_REVIEW_20261026.md)
supersedes the earlier checkpoint below. Compact, source-balanced supervised,
spatial, scoped routing and symptom-region controls have completed and stopped
after failing their gates. New shared caches avoid repeated encoder inference;
the final fixed route uses zero new encoder inputs. No GPU job remains active
at this checkpoint. Recognition, export and independent acceptance gates remain
unchanged. PR 6 is reviewed every four hours through the existing conversation
while the laptop is on, logged in and connected; adoption requires useful,
validated changes, never an automatic merge.

26 October 2026. This is the current checkpoint and supersedes the pending
acceptance/training statements in the earlier continuations. **Recognition is
not fixed and no replacement model is qualified.** The phone is disconnected;
its interim 2.0.2 model is unchanged. No export or Android integration is queued.

## What changed at the user's optimization request

The active job was the supervised SigLIP2 tail adaptation. Its worker was left
running. Only its parent controller was held, preventing the queued unconditional
full app replay. After the worker completed training and checkpoint restore
parity, the obsolete controller was removed. The action is recorded in remote
`phase26_sft_source/optimization_hold.json`. No completed experiment was restarted.

The optimization now has two stages:

1. `cache_first_screen.py` reuses exact-checkpoint supervised validation logits,
   completed frozen-head predictions and shared frozen encoder features. It
   computes only missing supervised full-frame inputs and evaluates actual app
   verdicts against the original retention rules, field Healthy minima, raw
   condition retention and each internal source's Healthy minimum. Benchmark
   AUC does not rank or qualify candidates.
2. Full restored-model app verification is allowed only after those cheaper
   checks pass. Export, fresh acceptance and native integration remain later
   gates. No candidate passed this screen, so none advanced.

The current scope is one completed supervised tail adaptation and one justified
cached alternative: the hybrid previously recovered 42/53 TOM internal Healthy
images, the strongest standalone result in that source. Existing CVaR, ranking
and label-reward heads are reused as negative guards; they are not retrained.
There is no new encoder/head/reward architecture sweep.

## Verified reuse and measured timings

| Operation | Measured result |
| --- | --- |
| Completed 16-epoch supervised adaptation | 354.51 seconds (5m 55s), 21,364,481 trainable parameters |
| Existing logits reused on old development | 3,716 of 5,820 unique images |
| Existing logits reused on internal development | All 274 images |
| Missing supervised full-frame inputs | 2,104; computed once in 7.06 seconds |
| Initial cache-backed screening pass | 19.08 seconds total |
| Final fixed 400-policy screen, all predictions cached | 14.14 seconds total, zero new model inference |
| Frozen feature reuse audit, three representations | 23,259 old rows reusable; only 502 added rows need extraction |
| Full restored fusion replays completed before optimization | Approximately 34–35 seconds each, derived from protocol/result artifact times |

The reuse audit compared every reused prefix row against the already-completed
new feature extraction. Maximum feature error was 6.68e-6 for SigLIP2 and the
hybrid and 0.0 for DINOv2 Large. No encoder was executed and no training run
was restarted for this comparison. Reuse would avoid 97.9% of encoder rows
when extending this 23,259-row frozen corpus by 502 photos. This is a verified
future/cache-recovery saving; it does not erase earlier duplicate extraction.

`feature_cache_reuse.py` creates an explicit plan that checks encoder SHA,
feature-array and row identities, normalized preprocessing/encoder source AST,
runtime package versions, exact input-row prefix, and unchanged shared canonical
pixel shards. Changed weights, preprocessing, runtime, rows or pixel files fail
closed. `permissive_encoder.py` and `dino_large_encoder.py` now extract only
missing IDs; hybrid features concatenate the matching component caches. Existing
complete caches return immediately. The original 54-image restored-encoder
checks remain in place. New sidecars do not rewrite old experimental identities.

Completed caches also validate their own full contract, including the entire
feature-array SHA and all canonical pixel shards. Legacy caches without that
contract require explicit registration after a source/restore audit; the reader
does not silently create provenance. A completed hybrid cache now returns before
loading encoder states or repeating its GPU probes. Real cache-hit checks over
all 23,761 rows took 0.81 seconds for SigLIP2, 2.57 seconds for DINO-L and 2.60
seconds for the hybrid, with CUDA never initialized. See
[completed cache-hit evidence](evidence/completed_cache_hits_20261026.json).

**Do not reuse frozen SigLIP2 embeddings for the adapted supervised encoder.**
Its weights changed. Only the selected supervised checkpoint's own validation
logits were reused; its 2,104 missing inputs were actually inferred. Frozen
negative heads use their matching original features, with all 274 internal
cached input tensors compared byte-for-byte against canonical source images
before reuse. Cached genus scores retain their original identity.

## Recognition outcome and stopping decision

None of the 400 fixed cached policies passed every gate. The stopping rule was
written in the screening artifact: stop this checkpoint and the cached hybrid
if the original, raw-condition and per-source internal app gates cannot all
pass. Do not launch additional reward/head sweeps, full graph replay, export,
Android work or fresh acceptance to rescue an unsuccessful screen.

The failure concerns the requested errors, not an AUC target. For example, a
supervised policy recovers **44/53 TOM Healthy and 21/29 AD internal Healthy**,
with internal retention passing. However, on `phase2_selection` it produces
**four supported-disease-to-Healthy errors where the unchanged gate permits
at most two**, and it also fails other original cohort checks. The hybrid can
recover 48/53 and 21/29 but introduces an additional internal AD false Healthy
and fails original safety checks. At safe cutoffs their Healthy recall drops
below target. These examples are rejected development policies, not candidates
approved for deployment.

Gate failures overlap: 368 policies failed original retention, 180 raw-condition
retention, 214 internal retention, 260 per-source internal Healthy minima and
120 old field Healthy minima. The exact counts and paired limits are in
[cache_first_screen_20261026.json](evidence/cache_first_screen_20261026.json).
The fixed grid includes the existing .995/.999 confidence tails. It must not
expand indefinitely after failure.

Useful next investigation is a small error-level audit of these concrete
supported-disease/unsupported-condition conflicts and the original nine-class
reference evidence, followed by a justified supervised/data change. A new
training proposal must explain which failed cases it addresses, its finite
budget and app-verdict stopping criteria before starting. Another identical
head/reward sweep is not justified. Fresh OLID images cannot provide diagnoses,
threshold selection, error inspection or teacher labels for this investigation.

## Consumed AD is permanently development data

The earlier independently frozen phase22 opening failed: Healthy improved from
42/104 to 62/104 (59.6%), while diseased false Healthy increased from 39 to 45.
There were eleven newly introduced false Healthy results and five removed,
giving the net increase of six. Original named disease verdicts were unchanged,
but that does not make the new Healthy calls safe. Do not install this model.

All 502 AD images and their inferred families are now retired from independent
acceptance. Their human labels, never candidate scores, were admitted using a
fixed family-hash split: 408 training photos (75 Healthy, 333 diseased) and 94
internal photos (29 Healthy, 65 diseased). They extend the human-only scene
corpus to 23,761 rows, SHA
`928f6bd167d0370f9e598dc4193c05942f58a81a4e8707a2bcf47ddc2a4652f2`.
Train/validation groups remain disjoint. The 274-photo internal replay combines
the older 180 consumed photos and these 94; it is not independent evidence.

Before optimization, this revision completed twelve frozen supervised heads,
three human-label fusions and four risk/reward controls. The one supervised
tail adaptation brings the historical valid completed-control count to 338.
Eight invalid unloaded-encoder fits remain excluded; deterministic reruns and
policy grids are not counted as additional independent training controls.
Apple-dependent controls remain research only and never provide product
weights, logits or distilled targets.

Evidence: [failed frozen opening](evidence/context_guard_external_20261026.json),
[human-only admission](evidence/consumed_ad_admission_20261026.json),
[complete licensed inventory](evidence/permissive_scope_20261026.json).

## Fresh source remains sealed

The [OLID I publisher source](https://zenodo.org/records/8105154), CC BY 4.0,
provides 798 tomato images in two publisher-MD5-verified archives. No diagnostic
model has predicted these images and no fitting uses them. After source-label,
duplicate, prior-overlap and SIFT capture-family audits, 730 photos remain:
223 Healthy, 192 Leaf Miner, 183 Mite, 41 Nitrogen deficiency, 28 Potassium
deficiency, 37 combined N/K deficiency and 26 Jassid/Mite images. There are
85 inferred conservative families, including 31 Healthy families, not verified
plant IDs. Audits compared 31,813 prior file hashes and 35,928 perceptual hashes.

Mite and Jassid/Mite are explicitly unsupported Other. Pure N and K retain the
supported class names; combined N/K stays coarse diseased and is never forced
into nine-class Other. No outcome-based filtering or relabelling was performed.
The audit is not exhaustive cross-source SIFT matching or clinical adjudication.

Sealed manifest SHA:
`b6462bb09b960b4bc02591ed7299bc932d09866550b60f8e5b3fc6e2507d5db6`.
Archive SHA values: part 1
`c3af8e6ae4901dab643cd219e9d00fac98f1f4affc47aa610eb52c54e9fd788f`;
part 2 `2ae17f6b56fd09d8795bbbb2cfabd7a614bb95257e99ba7ae91762deafc43591`.
Keep this source sealed until an exact candidate passes all cheaper development
checks and its independent protocol is frozen. Do not reuse AD as independent.

Other investigations do not add another benchmark: `h7bvnhz28x.1` downloaded
successfully but exposes PlantVillage filenames and is unsuitable as a fresh
field cohort. Digital Green metadata contains only twelve Healthy tomato photos;
no pixels were inferred or fitted. TCP has mixed internet/prior-dataset
provenance; PlantSeg has noncommercial terms and TomaMMU is gated. None supplied
new product training or an independent acceptance result.

## Reproduction, estimates and delivery state

SSH remains `ssh agribot-5090` over existing Tailscale. Python is
`/home/genesis/.venvs/agribot-train/bin/python`; artifacts are
`/home/genesis/agribot-data/recognition_campaign_20261024` (`P`). All useful
workers completed. tmux session `agribot` currently has only the shell window;
there is no abandoned training or duplicate replay running.

```bash
PY=/home/genesis/.venvs/agribot-train/bin/python
P=/home/genesis/agribot-data/recognition_campaign_20261024
# This is cached-only and preserves the versioned executable source.
$PY "$P/phase27_screen_diagnostics_source/model_training/v3/cache_first_screen.py" \
  --campaign "$P" --out-name phase27_screen_diagnostics \
  --scores-from "$P/phase27_cache_screen"
```

Revised planning estimates, based on measured progress: cached checks are about
15–20 seconds, a comparable 16-epoch partial adaptation is about six minutes,
and previous full restored fusion replay is about 35 seconds (budget 1–3 minutes
for a future qualified graph including restore/I/O). These estimates cover
specific operations, not a promise of a successful recognition repair. There
is no reliable release ETA while no candidate passes. Stop rejected work
instead of spending the saved time on more unconstrained experiments.

Local verification: **141 tests, 140 passed, one optional skip**. Six new cache
tests cover matching-prefix reuse and rejection of changed encoder, input,
preprocessing, runtime and pixel data. Latest completed native CI before this
checkpoint: `40272535806`, success for `07146ba8e194e02ffbaa927384373f0b7ece130f`.
No Android source or asset changes were made here. Existing draft PR #6 stays
unmerged. Physical phone validation remains deferred until recognition gates pass.

An additional 17,741,734,400 bytes of unused regenerable scene token arrays were
reclaimed after checking active consumers and open descriptors. Progress was
reset for regeneration. Together with the earlier leaf-token reclamation,
38,730,303,744 derived bytes were removed. Source photos, weights, pixel shards,
feature caches and experimental evidence are retained.
