# App and recognition checkpoint — 27 October 2026

No replacement recognizer qualifies for deployment. The copied DINO tail meets
every Healthy recovery minimum and preserves original disease verdicts, but
accepts one extra unsupported-condition photo. A fixed whole-image confirmation
alternative restores safety while losing too much Healthy coverage. Both routes
are stopped; neither is an accepted model.

This supplements the [PR review](PR6_CODE_REVIEW_20261026.md) and
[retraining handoff](RETRAINING_HANDOFF.md). The
[condensed recognition evidence](evidence/app_recognition_closure_20261027.json)
records 13 completed artifacts, their SHA-256 identities, frozen protocols and
app-verdict checks. Remote artifacts are under
`/home/genesis/agribot-data/recognition_campaign_20261024` on `agribot-5090`.

## Completed bounded controls

| Control | Measured work | Stopping decision |
| --- | --- | --- |
| Phase 46: paper-primary spatial coverage | Add 2,338 unique canonical maps in 7.60 s; reuse 45,664 existing maps. Fit 60 epochs in 4.13 s; six checkpoints and two fixed policies. | No passing policy. The diagnostic selection uses app-verdict ranking, not best validation BCE. |
| Phase 47: scoped paper recovery | One fixed cached route; zero encoder inference. | Healthy and unsupported-condition retention fail. |
| Phase 48: optimistic recovery bounds | Count feasibility ceilings under existing guards. | Diagnostic labels never select per-photo routes or create training targets. |
| Phases 49–50: copied CNN tail | Cache 48,002 original-prefix inputs in 53.18 s with zero reconstruction error. Train 11 epochs in 11.05 s; validation selection at epoch 6. Screen one fixed cached route. | Internal TOM Healthy 39/53 and AD 22/29, but old paper recovery and unsupported retention fail. |
| Phases 51–52: copied DINO tail | Freeze the first 10 original blocks; train the last two, norm and classifier copy for eight epochs in 83.17 s. Select epoch 7 by weighted validation BCE. Reuse 34,161 fitting prefixes; compute 13,841 missing evaluation prefixes once. | All Healthy minima pass; unsupported acceptance fails by one photo. |
| Phase 53: whole-primary confirmation | One frozen binary Healthy confirmation at 0.5 on new Healthy proposals. Reuse 169/186 canonical inputs; infer 17 new inputs in 1.07 s. No fitting. | Old retention passes, but old TOM and internal Healthy minima fail. |

The copied DINO uses an auxiliary Healthy margin plus a small supervised
multiclass term where the original condition target is explicit. Coarse disease
labels never become named conditions. Original deployed weights are unchanged.
Optimizer, EMA and RNG state are retained. No Apple-restricted parameters or
teacher predictions enter these controls.

Phase 52 obtains 98/129 old TOM Healthy, 87/119 old paper Healthy, 39/53 internal
TOM Healthy and 22/29 AD Healthy. It preserves every original named disease
verdict and passes internal safety and coarse-disease checks. Public and
prior-test unsupported false-known counts are each 8 against a maximum of 7.
These represent the same extra photo in overlapping cohorts. The gate remains
unchanged despite this near miss.

Phase 53 retains 87/119 old paper Healthy but drops old TOM to 87/129, internal
TOM to 29/53 and AD to 20/29. Its restored safety does not compensate for failed
Healthy recognition. Preserve both results and caches; do not expand their
thresholds or repeat either completed route.

## Large-cache failure and verified recovery

The PC's OOM handler killed initial DINO preparation before its first training
epoch. The shared encoder digest helper used `Path.read_bytes()` and allocated
the entire approximately 13 GB token cache while hashing it. It now delegates
to the existing streaming SHA-256 helper. Digest values, weights, canonical
inputs and preprocessing remain identical, so valid cache identities survive.

The original failed snapshot remains intact. A new recovery snapshot audits
old fitting prefixes against original cached embeddings and re-encodes added
fitting inputs to verify all remaining rows before completing the cache
contract. Restored-checkpoint probability residual is at most
`3.5762786865234375e-6`. No completed fit was restarted. A regression test rejects
whole-file reads in this hash path.

Approximately 13 GB of useful fitting tokens remain in shared memory, linked
from `phase51_dino_adaptation/prefix/features.npy`. They are volatile across
reboots. Inspect RAM, shared-memory and disk capacity before allocating another
large representation; preserve useful caches and existing jobs.

## Completed local app verification

The pending app 2.0.3 / code 5 candidate retains the installed interim model.
Quick checks save their completed run and decision atomically. Walk movement
saves current and skipped plants together before publishing a new position;
failed saves retain position and evidence for retry. Finish retries preserve
committed evidence identities, and GPS snapshots use deterministic event IDs.
Restored unfinished walks start paused; completed runs remain completed.

Backgrounding, pausing and plant changes invalidate stale inference. Walk step
and GPS collectors stop when inactive, paused, finished or cleared. Model
startup failure surfaces an error rather than waiting indefinitely. Ten new
Android tests exercise actual Room rollback, retry, stale frames, restoration
and failed startup. The selected emulator suite totals 20 checks across five
classes, also covering launch, export, verdict rendering and model goldens.

Earlier local runs passed 20 selected checks and then ten strengthened retry
checks. Lint 8.8.2 reported zero errors and seven warnings, with Navigation
custom checks executing. Its standalone version setting follows the
[official Compose lint guidance](https://developer.android.com/develop/ui/compose/tooling/lint).
CI now requires the selected API 35 emulator suites to execute without skips;
debug supports x86_64 and arm64-v8a, release arm64-v8a.

The final debug/release/lint/20-check run passed in 43 minutes 44 seconds, with
20 executed emulator tests, zero failures and zero skips. The Python suite ran
169 tests in 149.08 seconds: 168 passed and only the optional downloaded-media
test skipped. The collected native unit reports contain 124 passing checks.
Lint reports zero errors and seven warnings, with no obsolete custom checks.
[Verification summary and archived XML](evidence/app_verification_20261027/summary.json)
pin APK hashes, embedded model identities, test classes and remaining limits.
APK inspection confirms version 2.0.3 / code 5, minimum API 26, target API 35,
the expected debug ABIs, a valid debug signature and no internet permission.
Both APKs contain the unchanged classifier, detector and model manifest.
The release APK is unsigned. Emulator parity does not establish physical-phone
latency or replace eventual device installation checks.

## Scheduled PR review and remaining gates

The GitHub connection reports PR #6 open and draft at
`187c87d19f075f8c184696cdfaf172d5e6d986b1`, with 16 commits, no reviews,
no issue comments and no inline comments. Both published CI checks pass. Nothing
remote changed, so there is no new code to incorporate or experiment to repeat.
The scheduled 10:13 IST recheck finds the same remote state. The dirty recognition
worktree was preserved; no pull, reset or merge was performed. The final Android
build has completed. No GPU training job remains active. The checkpoint is
updated and the next scheduled review is 12:01 IST.

AD is permanently consumed development. The 730 OLID acceptance images remain
diagnostically sealed. Original recognition and safety gates are unchanged.
No new model was exported or integrated. PR #6 stays draft until recognition,
restored-model verification, deployable runtime/export, frozen-protocol fresh
acceptance and eventual physical-device requirements are met.

Preserve the useful DINO checkpoint and exact-identity caches. A subsequent
control needs a distinct justified hypothesis, a bounded protocol frozen before
its output, and every unchanged app-verdict gate. Full restored-model inference
follows a cheap-screen pass; model export and app integration follow recognition
qualification. Do not use OLID to diagnose these failures, add photo-specific
exceptions or promise a release date from a near-passing control.
