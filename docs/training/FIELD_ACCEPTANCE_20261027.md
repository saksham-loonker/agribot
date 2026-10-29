# Field acceptance failure and continuing recognition repair

**Normal app build completed:** [2.0.4 baseline app delivery](NORMAL_APP_DELIVERY_20261027.md)
publishes the app repairs with the original model, 133 native unit checks and
22 emulator checks. Rejected replacement assets are preserved outside the
normal bundle; recognition remains unqualified.

**Latest offline continuation:** [corrected cache joins, supervised/reward
controls and stopping evidence](OFFLINE_RECOGNITION_20261027.md). No replacement
passes recognition. The new prospective source remains provisional and sealed.

This supersedes earlier statements that OLID is sealed or that development
qualification is sufficient for promotion. **The phase-69 candidate is rejected
for release.** App verification succeeded; recognition acceptance did not.

The active request is to finish the Android app and recognition together before
PR #6 leaves draft. Do not merge the PR, install this rejected model on the phone,
or describe the current APK as a production recognition release. The phone is
disconnected. Keep unrelated edits and every useful cache or running job.

## Completed and immutable evidence

The original development gates pass. Restored PyTorch and actual FP32 LiteRT
verification both reproduce all 6,742 frozen development verdicts. The actual
Kotlin accumulation and quick-check stopping replay also passes every original
recognition gate. Its three differences are conservative Other-to-Unsure moves;
Healthy and named disease verdicts match. These are development results.

Android 2.0.4, version code 6, builds successfully in the isolated PC checkout
`/home/genesis/agribot-native-field-20261027`. All 133 native unit checks pass,
release lint reports zero errors and seven warnings, and all 22 API-35 x86
emulator checks pass without failures or skips. These include native model
scores, disease preservation, persistence, lifecycle, export and English/Hindi
UI. Emulator results do not establish physical-phone latency or memory limits.

The app repair prevents a confident Other verdict from stopping quick check
before its seven-real-frame primary recovery window. Duplicate callbacks cannot
invent observations. Image-level recovery says “Looks healthy in this view,”
retains actual leaf counts, and does not display invented leaf agreement.
Saved verdicts retain that scope. Row walking excludes off-plant primary views.

The shared leaf model has SHA-256
`3b38256d9a6a0b443423ade08d8c6da4460ced51e27f9295eb0d38380d47ad7a`;
the whole model has SHA-256
`e11c8503398160ddc60d2dfa65cb28493a4458f5a079063a1c8f703102659d01`.
The original nine disease logits remain present; original named disease verdicts
are protected by routing. Neither the detector nor its thresholds changed.

## Prospective acceptance: failed, not reusable as independent

The candidate, native implementation and acceptance protocol were frozen before
opening OLID. Protocol SHA-256:
`22a7842047a1c9b5a8ca178a39832bac916e62012ae39039eb7ec944a87dcdcf`.
The exact source snapshot is on the PC at
`/home/genesis/agribot-data/recognition_campaign_20261024/phase76_frozen_source`.
The protocol and result are committed as evidence alongside this document.

| Measure | Original model | Frozen candidate |
| --- | ---: | ---: |
| Healthy correctly reported Healthy | 87/223 (39.0%) | 87/223 (39.0%) |
| Disease images reported Healthy | 16/507 | 16/507 |
| Healthy gain | — | 0 percentage points |
| Original named disease verdicts changed | — | 0 |

Safety retention passes. Healthy recall fails the unchanged 70% minimum, the
10-point gain requirement, and the positive paired-family gain interval.
**AD and OLID are both permanently consumed development now.** Subsequent OLID
grouped validation is useful development evidence and cannot be called fresh or
independent. Do not retune and reopen the same source as another acceptance test.

`phase76_fresh_acceptance` retains exact-byte input/model prediction caches,
the original baseline comparison, native-domain inputs and predictions, and the
full result. It evaluated 730 images in approximately three minutes. Do not
repeat completed inference or change that frozen result.

## Diagnosed bottleneck and bounded next work

The cached-score audit in `phase79_opened_olid_diagnostic` finds that every OLID
image is classified as isolated rather than a natural scene. Among the 136
rejected Healthy images, 107 pass the positive whole-image threshold, every one
passes genus and compact-positive gates, but none passes isolated confirmation.
Their adapted DINO minimum scores range from 0.049 to 0.609 against 0.95 required
by that branch. Removing confirmation would also admit diseased leaves: it is
not an acceptable repair. No evidence supports changing global normalization.

Phase 80 fits only two supervised confidence experts (linear and MLP32) from
cached model scores. It performs zero encoder inference. Both preserve all old
gates and safety but fail consumed-OLID family-validation Healthy recovery; both
are stopped. Keep their weights, predictions and fixed family split.

Phase 81 extends the original frozen SigLIP feature cache with only the 730 newly
opened images. All 6,094 old whole-image features are reused by exact encoder,
preprocessing and input identity. Its frozen negative/genus/context heads must
reproduce the already cached mobile outputs within 0.0002 on all new images.

Phase 82 fits two frozen-visual experts (linear and MLP128), preserving the same
family split. Both also fail the required Healthy recovery at their conservative
validation operating points while retaining the old gates. They are stopped.
Before spending GPU time on copied-block supervised adaptation, check whether
the operating-point fit included validation negatives that the actual isolated
route could never admit. That bounded cache-only audit must preserve every
original recognition and safety gate and must not manufacture case exceptions.

That audit is complete: all four operating points fail. The scope-correct
warm-up also fails. Its training checkpoint was preserved across an evaluator
name-shadowing fix; `--evaluate-existing` repairs evaluation without restarting
training, and records the original training-source hash separately.

One copied SigLIP suffix/pool control then completed four epochs and selected
epoch zero by validation loss. It preserves the old gates and safety but fails
Healthy recovery. It reuses 504 matching prefix inputs and computes 782 missing
intermediate representations; their maximum original pooled-feature error is
`0.0000191`. Final pooled features cannot replace pre-suffix tokens for training.
This control is stopped. Even ignoring false positives, a 0.60 positive cutoff
could recover only 15/40 validation Healthy images, so further threshold fitting
on these scores cannot meet 70%. No full-model verification/export is justified.

The next distinct representation is the original diagnostic DINO/CNN feature
pair on a single detected primary leaf, with center/full-view agreement. Keep
publisher supervision at primary-image scope, never fabricate clinical leaf
labels, and preserve actual counts. Reuse the existing paired feature cache and
extract only missing image inputs. This is preferable to further SigLIP tuning
or unconstrained positive-logit rewards after its measured ceiling failed.

If actual deployment eligibility cannot produce a passing operating point,
the justified next control is a copied final-block supervised adaptation with
the original classifier and encoder prefix protected. Reuse matching prefix
caches and compute missing intermediate representations only. Use explicit
stopping criteria. No RL reward can substitute for missing healthy-versus-disease
information or relax the disease-to-Healthy limits.

Any promising new candidate needs cheap full development checks first, then
restored/mobile/native verification, and a genuinely unconsumed acceptance
reservation frozen before its predictions. The separate `bpfd9cns5g` source was
previously inspected for annotation geometry and excluded from training; inspect
its exact usage history and quarantine inspected/overlapping families before
considering an unpredicted remainder. Do not assume it is fully untouched.

## Build and transfer acceleration

The PC's existing JDK/Android SDK builds the full app in 67–111 seconds; Windows
builds took several minutes and its emulator instrumentation process crashed
before producing checks. The passing native evidence comes from the separate
Linux KVM emulator. Both temporary emulators are stopped after evidence capture.

`apk_asset_delta.py` transfers APK metadata/code/signing bytes while referencing
verified, identical local uncompressed model payloads. Reconstruction must match
the complete original APK SHA-256 before replacing any output. Three checks cover
byte-exact reconstruction and corruption/mismatched-model rejection. This reduces
the 579,658,783-byte release APK transfer to 5,330,698 bytes. The reconstructed
unsigned release hash is
`c7b2bbb9290ab7abd710d1007d9c10d1f9e94d6dea3c75867ada4d3135827ad9`.
Local artifacts under `.toolchains/app-candidates/2.0.4` are research-only because
recognition acceptance failed. They have not been installed on the phone.

A prospective [TCP v4 source](https://data.mendeley.com/datasets/m4m6j2tjfj/4)
has been downloaded by its publisher SHA-256, with no diagnostic inference or
training. Its methods explicitly include internet/other-dataset images; it is
not presumed independent. Fruit folders are excluded from leaf acceptance.
Any reservation requires label-scope, overlap, duplicate and inferred-family
audits first. Keep this source out of all fitting and policy selection.

Continue through useful offline work. PR #6 remains draft. Only ask for the
phone when a qualified, concrete candidate is ready for its physical checks.
