# Cached density and held-source meta-learning continuation

## Decision

The best new fixed development policy recognizes **19/40 Healthy with zero
new false Healthy**, compared with the preceding 18/40 policy. This remains
below **28/40**. No candidate passed the cheap recognition screen, so no full
encoder verification, export or Android model replacement was performed.
The normal app retains its original model; the combined PR remains draft.

These results use consumed development data. AD and OLID are not independent
acceptance. The provisional TCP reservation remains diagnostically sealed and
unused for training. Every original recognition/safety gate remains unchanged,
including exact earlier Healthy/named-disease retention. No clinical labels or
denominators were changed. The inactive retention proposal stays inactive.

This follows [the convergence and capacity record](CONVERGENCE_AND_CAPACITY_20261029.md).
There were no active GPU jobs at the initial inspection. Completed experiments
were not restarted, and useful caches/checkpoints were preserved.

## New models and outcomes

| Phase | Predeclared change | Actual development outcome |
|---|---|---|
| 215, mobile/shared | Weighted train-only PCA-32; one Gaussian per clinical label, shared regularized covariance | 13/40 Healthy, zero new false Healthy |
| 215, mobile/class shrinkage | Same projection; class covariance blended with the shared regularized covariance | 13/40, zero new false Healthy |
| 215, native/shared | Same learner on cached native-518 DINO/ConvNeXt features | 13/40, zero new false Healthy |
| 215, native/class shrinkage | Native features and class-specific covariance | 13/40, zero new false Healthy |
| 217, linear | Logistic meta-head on four cross-fitted density margins per crop | 13/40, zero new false Healthy |
| 217, quadratic | Same evidence plus the ten pairwise/squared products, logistic C=1 | 13/40, zero new false Healthy |
| 218, episodic ERM | Matched supervised control on source-separated support/query episodes | Selected 15/40 at zero new false Healthy |
| 218, MLDG | Differentiate through a support update before computing held-training-source loss | Selected 15/40 at zero new false Healthy |

Thirteen Healthy cases are already retained outside the recovery cohort; a
13/40 result therefore contributes no new target Healthy recovery. Counts use
the full 40-image denominator, including guard-blocked images. No benchmark
AUC substitutes for these app-verdict checks.

Phase 216 tested four density experts as both-crop negative vetoes at the
fixed .05 and .5 thresholds on the 18 existing two-native coverage parents:
**144 policies, zero passing**. Its best selected tradeoff was 26/40 Healthy
with two new false Healthy (one OLID, one TLID), which is a failure.

The unclipped min-crop density-ranking diagnostic has an optimistic zero-FP
single-expert ceiling of 14/40 for mobile/shared and 13/40 for the other three;
their oracle union is 14/40. This is a consumed-label diagnostic only. Its
cutoffs are not candidates. It closes monotonic positive recalibration of
these four density scores as a route to 28/40; it does not prove impossibility
for unrelated representations or arbitrary conditional models.

Phase 219 tested one fixed recovery architecture: retain the mobile/shared
density MAP veto, optionally add nitrogen/Other MAP vetoes, and turn a fixed
.95 recovery OR of the selected episodic ERM, MLDG and earlier Other head
off/on. Eighteen parent pairs × four condition subsets × two recovery settings
produce **144 policies, zero passing**. The best result is the 19/40, zero-new-FP
policy above. Parent admission remains .95. Existing outputs are protected by
the original conditional routing and admissibility masks. This is not a
clinical guarantee or fresh acceptance result.

## Method and leakage controls

The density learner tests [regularized discriminant geometry](https://scikit-learn.org/stable/modules/lda_qda.html),
not another neural-margin or kernel sweep. It gives Healthy and pooled disease
equal Gaussian-mixture prior mass. Weighted PCA and covariance estimates are
fit independently inside each training calibration fold; no validation
features or labels fit those estimates. Shared covariance shrinks halfway
toward trace-scaled identity. The alternate covariance blends class covariance
halfway with that shared base, with a .001 trace-scaled diagonal floor. There
is no dimension, shrinkage or threshold grid.

Four-fold calibration merges inferred families and shared exact input IDs.
The entire field-93 collection stays in one calibration fold. Both crop views
stay together and each receives half a photo's weight. Supervision is on actual
ROI means per crop, not invented clinical labels for individual detections.
Training-only weighted Platt C=1 maps the OOF margins; Healthy admission takes
the minimum of the two crop probabilities. Stored model parameters restore
the recorded density margins exactly in the executed environment.

The meta-heads use the intersection of training photos covered by both feature
families, joined by exact path and checked y/clinical-label/family agreement.
They never join field row indices between the phase-168 and phase-193
namespaces. Only cross-fitted base margins train the meta-head; its centering,
scaling and logistic fit use training evidence. There is no additional
meta-OOF calibration or independent probability-calibration claim.

The [MLDG-style objective](https://arxiv.org/abs/1710.03463) explicitly tests
training-source shift. Each episode samples 32 support and 32 query photos.
Support excludes the held training source and all family/input components
connected to it. Clinical/binary/source/family balancing applies inside the
pools. The objective is half support BCE plus half query BCE after one
differentiable support update, inner step .05. The matched ERM uses the same
episodes without that update. An analytical gradient test checks the Hessian
term; a zero-step test recovers ERM. These are held **training** sources, not
fresh validation or acceptance data.

Both episodic learners run 40 epochs, 32 episodes/epoch, AdamW .001 with .01
weight decay and gradient clipping at one. The protocol permits only
training-EMA stopping after at least 20 epochs, with 12 stale epochs and a
40-epoch cap. All 80 epoch weights and predictions were saved. Neither
objective met the recognition gate, so there is no full-encoder MLDG run or
inner/outer learning-rate sweep.

## Speed, storage and reproducibility

All encoder weights, preprocessing and input identities match the reused
caches. **New encoder inputs: zero.** The four density fits took 4.558 seconds
(mobile) and 2.792 seconds (native), including their training-only folds. The
two meta-heads took 2.614 seconds; both 40-epoch episodic runs took 8.264
seconds together. The two policy evaluations each took about .02 seconds.
Preparation, verification, backups and software tests are additional work;
these times are not phone latency or total user-visible completion time.

PC disk was below the earlier 1-GiB reserve, so density and episodic model
outputs used explicitly marked RAM directories. No useful training, cache or
unrelated user process was stopped to create space. An independent laptop
backup now contains **192 artifacts**, including all **80 epoch checkpoints**,
selected/final heads, optimizer/RNG states, predictions and protocols:

`.toolchains/model-checkpoints/phase215-219/artifacts.zip`

Size: 62,416,545 bytes. SHA-256:
`8821b3c737ca233fcee80e4592f122a37df16eed542fb60265e04b18fbf25771`.

Every archive member hash and ZIP CRC was verified. Both final optimizer/RNG
files were checked against their final epoch model states. The committed
[backup receipt](evidence/density_controls_20261029/checkpoint_backup.json)
and artifact manifest record the identities. RAM output directories remain
volatile; restore from the laptop backup if the PC reboots.

[Batch evidence](evidence/density_controls_20261029/) stores six losslessly
archived protocols, results, exact PC source bytes, runtime versions and the
artifact manifest. Learned arrays and weights are private, not committed.
The current Git utilities additionally harden two edge cases after execution:
zero-mass rows cannot increase PCA rank, and inner meta-updates preserve frozen
parameters. Completed PCA fits had positive weights and the trained heads had
all parameters trainable, so their recorded results are unchanged. Original
executed utility bytes remain archived; `post_execution_hardening.json` pins
both executed and current-tested source identities. Do not silently substitute
current helper bytes when reproducing a frozen historical protocol.

## Stopping decision and continuation

The local full suite passes **312 tests**, with one optional downloaded-media
skip. The new tests check weighted PCA/density behavior, explicit stack basis,
the meta-gradient and its ERM limit, frozen parameters, executed source hashes,
fold isolation and result bindings. Published `2508db6` passed both earlier
CI jobs; this source/evidence update requires its own CI result. Android source
and bundled weights are unchanged.

The declared PCA-32 density settings, two density meta-heads, matched episodic
objectives and fixed policy architectures are closed unsuccessful. No failed
model was promoted or exported. Do not repeat them, add epochs, fit cutoffs to
the consumed validation labels, or expand this into another coefficient grid.

The new safe development result is 19/40, not a qualified replacement. Full
source-retention checks, full restored-encoder verification, family confidence
intervals, fresh acceptance and Android integration remain downstream of a
passing cheap screen. Existing evidence still favors additional clinically
verified Healthy/nutrient field supervision before more of these objectives.
The previously prepared eight-case clinical-review packet is available, but
no clinical adjudication occurred and no disputed labels were silently changed.
