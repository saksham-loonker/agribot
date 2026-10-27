# PR 6 code review and continued recognition work

**27 October checkpoint:** [app and recognition continuation](APP_RECOGNITION_CHECKPOINT_20261027.md).
The scheduled GitHub check finds no remote changes at `187c87d`; both published
CI checks pass. Completed copied-tail and whole-image controls remain rejected.
App persistence/lifecycle fixes pass final debug/release builds, lint and all
20 selected emulator checks. Python: 168 passed, one optional-media skip.
AD is consumed,
OLID stays sealed, and no model export or integration is authorized by these results.

Reviewed PR heads `69ac2434f8e4e3f69764f3e2f5f84dd7f012df7b`,
`399418fc9a1dbd35a9d7a77d0e747c42638d40de` and their working
continuation on 26 October 2026. The review focused on the high-risk training,
cache, evaluation and native verdict paths; it is not a claim of line-by-line
review of the entire PR, which includes substantial generated JSON evidence.
**Useful engineering foundations; recognition repair remains unqualified.**

## Findings and repairs

1. **Evaluation resume did not pin image bytes.** `evaluation_identity` hashed
   the manifest, code and models, but a source JPEG could change at the same path
   while an old prediction remained eligible for resume. The schema-two identity
   now hashes each unique source file, checks publisher hashes when available,
   preserves row ordering and rejects duplicate IDs. Missing/changed files fail
   before cached results or inference are admitted. Schema-one resumes require
   a new output directory; historical evidence is not overwritten. The new
   supervised screen also pins all opened source image identities, including
   validation inputs reused by path. Tests cover changed bytes with identical
   file size/mtime, missing sources, duplicate IDs and hash disagreement.
2. **New hybrid assembly checked component metadata without checking each
   completed component contract.** A modified feature array could be combined
   under its old identity. Both component caches now validate their feature
   bytes, encoder, rows, runtime, preprocessing and pixel-shard contract before
   assembly. A regression test corrupts a component and verifies rejection
   before any encoder is loaded. Completed, independently verified hybrid
   caches retain their fast return path.
3. **The green Android CI did not establish that optional ML tests executed.**
   Its lightweight Python environment can skip training tests. A separate CPU
   job now installs the tested ML dependencies and fails on unexpected skips.
   The optional downloaded-media test may remain skipped. This is a source and
   invariant check, not GPU training, model acceptance or device validation.

These changes are used in the continued work. Existing frozen experiment source
snapshots and their identities remain unchanged.

## Parts retained and used

- Original-checkpoint initialization, explicit label ordering, persistent
  optimizer/EMA state, resume identity checks and reliable-teacher retention.
- Canonical preprocessing and actual app-verdict replay, integer-floor safety
  gates, per-condition accounting and explicit consumed-source lineage.
- Cache-first screening and verified feature reuse. Adapted encoder features
  cannot reuse the frozen encoder's feature cache.
- Native frame-quality admission, strict tensor/normalization contracts,
  full-ROI confirmation and aggregate reference-disease priority on the same
  real observations. The optional expanded model schema stays inactive in the
  current assets.

The many experimental heads and reward arms are research history, not production
defaults. Failed model weights are not adopted. `context_abstention.route_context`
uses simulated static-frame evidence for host research; its literal seven-frame
result must not be copied into a live-camera flow. Any future image-context
integration must accumulate actual admitted frames and preserve real leaf counts.

## Resumed supervised investigation

After inspecting completed jobs, the continued work tested a finite actual-ROI
agreement hypothesis, then one compact Healthy head sharing the deployed DINO
and MobileNet features. Both original encoders and all nine-class disease
weights stay frozen. Proper leaf annotations and primary-image labels are used
within their scopes; weak TOM targets and detector-propagated fitting labels are
excluded. AD detected leaves are not assigned AD's primary-image label.

Preparation reused **35,474 original DINO vectors**. Only **7,641 new input
tensors** needed DINO extraction; the combined cache has 43,115 unique tensors
and 29,211 fitting rows. Every stored old evaluation crop's canonical tensor
hash matched its source crop. Reference composition error stayed below 0.002
logits and original verdicts were unchanged. This provides no evidence of a
0–1/0–255 mismatch in that evaluated path.

The first compact fit completed in **24.69 seconds**. Safe policies recovered
too little Healthy. A twelve-policy combination with the stronger field
specialist recovered 41/53 TOM and 22/29 AD Healthy photos but failed safety.
The detector also supplies no leaf on 14/53 internal TOM Healthy photos, limiting
a leaf-only repair; whole-image routing is still only a research prototype.

The measured sampling audit found that AD negatives received **0.1519%** of the
compact fitting draw probability. The historical sampler puts AD inside the
large `original` domain bucket. This is a weighting limitation, not proof that
all earlier experiments were invalid. A separate source-balanced control keeps
the same model, seed, loss and learning rate, assigning each present binary
class/publisher pair 6.25% of total mass, with equal family mass within the pair.
Its fit took **24.38 seconds**. Named-disease false Healthy fell from five to two
in `phase2_selection` for a comparable veto policy, but other gates still failed.

Eight already-trained source-balanced checkpoints were then evaluated with four
fixed policies each, without executing either encoder. None of those 32 checks
passed every original and internal gate. The compact/veto branch is stopped;
its head checkpoints, predictions, features and protocols remain available.

The next isolated control applied this source-balancing change to the existing
supervised SigLIP2 tail adaptation, keeping the same architecture, 16 epochs,
focal loss, seed and learning rates. This is a new fit, not a restart of the
completed experiment. It completed in **352.71 seconds**, with frozen
source at `P/phase33_sft_balance_source` and output at
`P/phase23_siglip/sft_ad_tail2_focal_source_balanced`.
`P=/home/genesis/agribot-data/recognition_campaign_20261024`.

Its cheap screen used eight fixed supervised/CVaR policies: Healthy
thresholds 0.8/0.9/0.95/0.99 and negative cutoffs 0.001/0.01, preserving original
Healthy and named disease verdicts. It reused that exact checkpoint's validation
logits for 3,990 old/internal inputs and inferred only 2,104 missing images, in
6.95 seconds. The eight-policy screen took 0.16 seconds (12.42 seconds including
identities, loading and inference). **None passed.** Its safest tested policy
retained original and internal safety but recovered only 24/53 TOM Healthy and
11/29 AD Healthy. At a lower cutoff it reached 49/53 and 21/29 while failing
safety. This checkpoint is stopped; no further threshold expansion or full
restored-model replay is authorized by these results.

AD is permanently consumed development. All 730 audited OLID images remain
diagnostically sealed, including 223 Healthy. There is no new independent
acceptance claim. The phone remains disconnected and Android assets unchanged.

## Evidence and verification

At published head `399418f`, the full local suite ran **153 tests, 152 passed,
one optional media skip**. Compile and diff checks passed. Both the new CPU
training-contract job and native Android job passed in
[CI run 40295487690](https://github.com/saksham-loonker/agribot/actions/runs/40295487690).
This verifies engineering checks; it does not qualify recognition.

- [Canonical feature preparation](evidence/compact_features_20261026.json)
- [Source balance and completed-head diagnosis](evidence/compact_diagnosis_20261026.json)
- [Rejected actual-ROI agreement](evidence/roi_agreement_20261026.json)
- [Rejected source-balanced veto](evidence/source_balanced_veto_20261026.json)
- [All eight cached checkpoints](evidence/compact_cached_checkpoints_20261026.json)
- [Completed source-balanced supervised fit](evidence/source_balanced_sft_fit_20261026.json)
- [Rejected eight-policy supervised screen](evidence/source_balanced_sft_screen_20261026.json)
- [Final cache-only identity verification](evidence/source_balanced_sft_final_cache_20261026.json): same rejected outcome in 3.76 seconds, zero new encoder inputs

Measured planning: tiny cached-feature fits about 25 seconds, cached policy
screens about 2–3 seconds each, comparable 16-epoch tail adaptation about six
minutes. These estimate operations, not successful recognition or a release
date. Keep PR 6 draft until recognition, exported-byte and native/device gates
pass; do not merge solely because infrastructure checks are green.

## Subsequent bounded continuation and code quality

The useful foundations above are already in use. PR 6 is a large research and
application change, so this review is deliberately **not** blanket approval to
merge it or install every experimental checkpoint. The new reusable pieces are
the opened-image feature cache, shared app-gate screen and explicitly audited
spatial cache. Completed experiments retain immutable remote source snapshots.
The deployed DINO/MobileNet weights and Android assets remain unchanged.

The opened-image cache covers 6,094 unique opened photos. It reused 4,297
matching frozen SigLIP features after verifying the newly decoded canonical
pixels against the original pixel cache. Only 1,794 unique missing tensors
needed encoder inference, measured at 3.21 seconds. Subsequent heads and routes
use this cache. An adapted encoder cannot reuse these frozen final features.
Regression tests verify that completed reuse loads no encoder and rejects
changed source bytes despite identical file metadata, corrupt features, and
different encoder weights.

Spatial preparation extracted the original CNN's 576-by-7-by-7 activations for
43,115 existing inputs in 42.76 seconds, without running DINO. Pooling every map
through the original CNN head reconstructed all stored CNN embeddings with
maximum error 5.25e-6. The contract pins map bytes, preprocessing, runtime,
weights, source snapshot, fitting rows and input descriptors. All maps remain
cached for reuse. A separate primary-scene coverage control reused those maps
and computed only 2,549 new CNN maps in 6.12 seconds.

The following controls are **stopped**, with no export or app integration:

| Control | Measured work and decision |
| --- | --- |
| Strong negative leaf abstention | Two fixed cached policies; neither passes every gate. It can remove four old false Healthy results without removing a correct Healthy result in that control, but other regressions remain. |
| Family prototypes | Two fixed cached policies, 2.94 seconds; insufficient Healthy recovery. No encoder fitting. |
| Spatial Healthy head | 60 epochs, six checkpoints and two fixed policies, 3.52 seconds; no passing policy. Targets supervise the whole annotated leaf/primary image, never individual patches. |
| Primary-scene spatial coverage | Adds 2,674 properly scoped primary rows, 60 epochs in 3.88 seconds; no passing policy. AD's primary labels are never assigned to detected leaves. |
| Capture-context head | Cached frozen features; source metadata supplies weak capture-style supervision, not disease targets. Two policies fail recovery. Its style labels are a limitation. |
| Primary crop verifier | Explicit publisher taxonomy; conflicting/missing PlantDoc genera and detector ROIs are masked. All 94 AD and 704 TOM validation inputs are classified as tomato at the binary decision boundary. This is a source-familiar taxonomy result, not condition recognition or independent acceptance. |
| Scoped context/crop routes | Two policies, then two fixed multi-leaf corroboration policies; none pass all gates. No new encoders or head fitting. |
| Scope-consistent corroboration | One fixed cached policy: internal TOM Healthy 39/53 and AD 22/29, with internal safety retained and all original named disease verdicts preserved. Fails old paper Healthy recovery (51/119) and unsupported acceptance (8 versus maximum 7 in public and prior-test cohorts). No threshold expansion. |

The last failure identifies a structural limitation: a route restricted to
natural scenes cannot repair the original isolated paper Healthy cohort to its
70% minimum. It is stopped rather than subjected to further threshold sweeps.
The unsupported regression is one additional photo represented in both old
cohorts, not two independent failures. Correct Healthy count retention, raw
condition accounting and every original safety gate remain mandatory; allowing
strong contrary evidence to abstain on an original Healthy verdict does not
authorize losing correct Healthy counts.

A separately scoped SLIF control trains a binary **symptom-region** detector,
not a generic leaf detector. Its publisher annotation role is supported by
[the source paper](https://link.springer.com/article/10.1007/s00521-026-12271-0).
All 612 detection-image byte hashes match the corresponding classification
images, annotation identities are retained, and original family splits remain
separate. This does not rehabilitate the quarantined generic-leaf experiment.
The original detector initializes compatible weights; the new binary head is
trained for this task. Training completed in 136.83 seconds, best epoch 14,
region mAP50 0.1933. This is not the deployed leaf detector's metric.

Before any app inference, its frozen 0.75/0.9 cutoffs were checked on all 148
held-out publisher development images in 1.13 seconds. They vetoed only 1/137
and 0/137 diseased photos respectively, with 0/11 Healthy false vetoes. Both
fail the explicitly frozen utility screen. This checkpoint is also stopped;
no full app replay, export, new fit or threshold expansion follows that result.

One additional audit-code defect was repaired: the old phase-34 audit added its
opened-data identity after writing its protocol. The corrected source freezes
the complete protocol first and uses a separately versioned output directory.
Historical source and results remain intact. Primary-crop target tests also
enforce ambiguity masking, unknown-source masking and the primary-image versus
ROI boundary. Spatial-head tests verify localized negative evidence survives
pooling and receives the intended training gradient.

This continuation's full local suite ran **164 tests: 163 passed and one optional
downloaded-media test skipped**. Compile, formatting and diff checks passed.
The previous published revision's two CI jobs passed; the next commit must
obtain its own CI result. Infrastructure success does not override failed
recognition gates.

- [Condensed immutable continuation evidence](evidence/pr6_continuation_checkpoint_20261026.json) includes hashes and paths for every complete remote artifact, original failed checks and selected metrics.
- [Symptom-region validation and stopping decision](evidence/region_validation_20261026.json) pins the checkpoint, predictions and frozen protocol.

AD remains consumed development. The 730 OLID images remain diagnostically
sealed. There is no new independent acceptance result. No useful job was
interrupted or completed experiment restarted. At this checkpoint no GPU
training job remains active; the controls above have finished and stopped.

The user requested periodic PR-6-only review. Local Windows task
`Agribot-PR6-Review-4Hourly` is registered, its first queue test succeeded, and
the next scheduled check is 26 October 2026 at 20:01 IST, repeating every four
hours. The local script is under `.toolchains/pr6-monitor`; it queues this same
conversation, uses no embedded credentials, and disables itself when the PR
closes. The laptop must be powered on, logged in and connected. Reviews must
compare current commits/comments/CI, preserve dirty work and existing training,
and adopt only useful validated changes. This is not automatic PR merging.
