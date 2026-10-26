# Bounded Healthy recovery: current implementation handoff

26 October 2026. This checkpoint supersedes the pending-training status in
SIGLIP2_CONTINUATION_20261026.md. Recognition is not yet qualified for release.
The user has disconnected the phone; continue offline without requesting USB.
The installed interim 2.0.2 model and existing APKs remain unchanged.

## Completed experiments and actual outcome

There are 318 distinct valid completed research controls: the preceding 292,
four SigLIP2 frozen heads, three original-data fusions, two full SigLIP2
supervised fine-tunes, four risk/reward heads, three admitted-training fusions,
two partial SigLIP2 adaptations, four original-Meta DINOv2 Large heads and four
SigLIP2/DINO-L hybrid heads. Eight unloaded-encoder invalid runs remain excluded.
Four deterministic head reruns, one pooling rerun and four saved-best/final
checkpoint reselections are recorded separately; they do not increase the
independent-control count. Policy grids and graph assembly are not training.
The 22 Apple MobileCLIP controls remain research only and cannot supply product
weights, logits or distilled teachers. All new controls use publisher human
labels independently and Apache-2.0 encoders.

Full supervised adaptation did not solve the field problem: the focal and BCE
controls recovered only 8/53 and 11/53 Healthy internal images in actual app
replay. Reward controls include CVaR hard-negative BCE, partial-AUC ranking,
one-step REINFORCE with asymmetric label rewards and a supervised/KL anchor,
and expected-logit utility. These are label-derived binary rewards, not RLHF
or a sequential environment. None qualified alone. DINO-L recovered 17/53,
and the hybrid recovered 32/53 with the original disease gates retained.

The latest improvement is a separately calibrated whole-image recovery model
plus explicit conflict handling. It does not change the primary nine-class
model or confuse sigmoid image logits with nine-class softmax probabilities.
It retains original named DISEASE verdicts exactly, protects suspected named
disease on UNSURE results, and blocks positive recovery when the primary
posterior has Other <= .1 and total named-disease probability >= .6. Missing
reference evidence blocks recovery. Negative Healthy evidence produces UNSURE
and retains real classifier probabilities, observations and counts.

## Verified candidates and safety tradeoff

`phase21_verified` uses positive Healthy >= .94, CVaR Healthy rejection <= .005,
and no genus cutoff. Fresh restoration matches all 6,468 development verdicts
on 5,820 unique images; maximum probability difference is 8.62e-6 and restored
weights/shared-encoder probes match exactly. It reaches 110/129 TOM Healthy,
109/119 paper Healthy and 39/53 newer internal Healthy. Although all original
retention gates pass and no individually correct original Healthy result is
lost, the broad viral subgroup worsens from 5/368 to 8/368 false Healthy. This
additional safety diagnostic failed. Do not present that policy as accepted.

`phase22_verified` instead selects the highest-ranked original-cohort policy
that also passes per-raw-condition retention. Selection does not use the newer
internal predictions. Its frozen thresholds are:

| Parameter | Value |
| --- | ---: |
| Positive Healthy minimum | .95 |
| Negative-guard Healthy maximum for rejection | .01 |
| Genus minimum | 0 |
| In-domain Other maximum | .1 |
| In-domain named-disease mass minimum | .6 |
| Protect suspected named disease | true |

It reaches **108/129 TOM Healthy, 109/119 paper Healthy and 38/53 internal
Healthy (71.7%, baseline 6/53)**. All original per-cohort/per-disease gates,
additional per-raw-condition checks and exact original named-disease priority
pass. Two individually correct original Healthy cases become uncertain while
every cohort's total correct Healthy count remains nondecreasing. These
individual losses are disclosed, not hidden by claiming exact Healthy
preservation. Internal false Healthy is 2 versus baseline 3. The internal 180
photos and old eight cohorts are development data, not independent acceptance.

Evidence: [context_guard_20261026.json](evidence/context_guard_20261026.json),
[full licensed experiment inventory](evidence/permissive_scope_20261026.json).

## Exact models and preprocessing

`phase21_guard_model.py` produces three raw calibrated binary logits in order:
positive Healthy, tomato genus, negative-guard Healthy. Use separate sigmoids.
The CVaR head shares the same SigLIP2 features with the positive fusion, avoiding
duplicate encoder execution. The original DINO/CNN fusion components are
strictly restored from pre-Apple checkpoints. The primary classifier is a
separate, byte-identical `d70135804cf750341218e89a055370bef46a9b337951897b9f55f2570844ed39`
LiteRT artifact. Neither primary disease weights nor its calibration is trained.

For no detected leaves, use a separate whole-frame primary-classifier
distribution, never a fabricated leaf vote. Historical cached whole-image
predictions were not assumed current. `phase20_whole_reference.py` freshly
evaluated all 248 relevant old/internal photos against that exact installed
LiteRT classifier. Predictions SHA-256:
`98288afd8f4293442a4388583dda825915df5303abb648785b7d2dff218befee`.
Use `context_whole_reference_probabilities`; absent evidence fails closed.

A fresh original-file pixel audit checked 234 photos, including all 180 newly
admitted internal images: maximum canonical cached-input error **0.0**. No
0-255 versus 0-1 defect was established. SigLIP2's inverse-ImageNet,
bilinear-256, native [-1,1] adaptation remains explicit and identical in fitting
and inference. Original DINO NHWC224 ImageNet normalization remains unchanged.
Both CUDA matmul TF32 and cuDNN TF32 are disabled; timm fused attention is off.

Meta DINOv2 Large source uses the original official Apache-2.0 weights:
`https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth`,
SHA `d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428`.
The HF port with conflicting Apache card/noncommercial embedded metadata was
rejected without using its weights. Positional embedding adaptation from
518 to 224 is recorded and verified. DINO-L/hybrid are controls, not components
silently added to the current production candidate.

## Frozen independent opening and remaining work

The separate AD source contains 502 publisher-labelled primary leaf images:
104 Healthy, 11 Early Blight, 96 Bacterial Spot and 291 Mosaic Virus. Explicit
unsupported diseases map to Other. There are 68 inferred conservative visual
families, including 20 Healthy families, not verified plant identities.
It has not supplied training labels, fitted logits or policy selection.

`phase22_accept.py` freezes the exact phase22 bundle, policy, implementation,
unchanged primary/detector manifest, source manifest and acceptance thresholds
before the first predictions. Frozen protocol SHA:
`7b0fc2cf09e341c6bf998d33400d968bce5d8a42e1d8957ee992e9177c1a2450`.
Candidate SHA: `ac93e08a9a5c80beea64afc2ca7a3679913bd0adc19ec28809b5181a758f37f1`.
Bundle SHA: `7eb6e3aa4e56b7250dcddf72e5095ae554dac0cfd092dd85c3cfcf1deb5930e0`.

Requirements: unchanged original retention rules, >=70% Healthy recall,
>=10 percentage point Healthy improvement, positive lower paired family
bootstrap 95% gain bound, no increase in false Healthy for sufficiently sampled
diseased/raw-condition strata, and one-sided bootstrap upper increase <=2 pp.
There is no post-hoc ceiling exception. Always report small-sample limitations
and Wilson Healthy intervals. An empirical bootstrap is not clinical proof.
Failure blocks promotion and consumes the source's independent status; do not
pick another candidate using its outputs and call a repeat independent.

Passing this prototype opening would still require actual LiteRT export,
full exported-byte regression, a separate native image-evidence contract,
real frame/quality bookkeeping and phone probability/decision/timing checks.
The existing schema-four source carries two leaf distributions and does not
implement this image-context path. Do not fake tracks, boxes or seven frames
from one photograph in the Android implementation. Static repeated-frame host
replay is a diagnostic, not a camera-session trial. Explain visible-view scope
without claiming the entire plant is disease free.

## Reproduction and recovery

SSH: `ssh agribot-5090` over the existing Tailscale network. PC repo:
`/home/genesis/agribot`; Python: `/home/genesis/.venvs/agribot-train/bin/python`.
Artifacts: `/home/genesis/agribot-data/recognition_campaign_20261024` (`P` below).
Long jobs use tmux session `agribot`; inspect windows/logs before restarting.
Every new run uses a versioned immutable repository-layout source snapshot.
Do not overwrite old frozen snapshots or alter an opened acceptance script.

```bash
PY=/home/genesis/.venvs/agribot-train/bin/python
P=/home/genesis/agribot-data/recognition_campaign_20261024
$PY "$P/phase22_validation_source/model_training/v3/phase21_validate.py" --campaign "$P" --coarse-priority --out-name phase22_verified
$PY "$P/phase22_acceptance_source/model_training/v3/phase22_accept.py" --campaign "$P" --freeze-only
$PY "$P/phase22_acceptance_source/model_training/v3/phase22_accept.py" --campaign "$P"
```

The RNG resume bug is fixed: new NumPy RNG keys serialize to primitive lists,
and legacy NumPy checkpoints load with a narrowly scoped weights-only allowlist.
No unsafe pickle fallback is added. Checkpoint selection now has a scale
invariant low-FPR option: recall at weighted FPR <=.005, secondary overall
recall and standardized partial AUC. Fixed uncalibrated probability thresholds
previously distorted checkpoint choice; original results are preserved.

20,988,569,344 bytes of regenerable DINO prefix token caches were removed only
after verifying no worker needed them. Source photos, trained models, feature
identities and results remain. Progress metadata was reset for regeneration.
There is about 12 GiB free on the PC; account for export temporary files.

Local Python: 135 tests, 134 passed, one optional skip. Previous Android CI
`40226237366` passed for `4927110961794368085773615b96af493b728fdf` (105 native
tests plus lint and APK assembly). New Python code has no Android model asset
changes. PR #6 remains draft; no merge, new-model installation or release claim.
