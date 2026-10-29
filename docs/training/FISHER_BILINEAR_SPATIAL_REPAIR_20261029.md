# Full-feature, interaction and spatial-colour continuation

The model remains unqualified. The best fixed development policy still recognizes
**19/40 Healthy with zero new false Healthy**, below **28/40**. This continuation
finished eight additional model controls and 576 fixed routing checks without
changing any recognition or safety requirement. No replacement was exported or
integrated into Android. The original-model app 2.0.4 delivery is unchanged.

AD and OLID remain consumed development. TCP remains provisional, diagnostically
sealed and unused for training. No clinical labels, validation denominators or
original exact-retention rules changed. The separate retention proposal remains
inactive. These findings follow the [density/meta continuation](DENSITY_AND_META_REPAIR_20261029.md).

## Completed hypotheses and stopping decisions

| Phase | Fixed hypothesis | Actual app-verdict screen |
|---|---|---|
| 220 mobile binary / clinical mixture | Full 1408-dimensional supervised Fisher geometry may retain useful directions discarded by PCA-32 | Both 13/40 Healthy, zero new false Healthy |
| 220 native binary / clinical mixture | Same learner on all 1792 cached native dimensions | Both 13/40, zero new false Healthy |
| 221 frozen-base bilinear | Rank-32 DINO-by-CNN products may capture morphology/texture interactions missing from concatenation | All 40 epochs completed; selected warm incumbent remains 15/40, zero new false Healthy |
| 221 jointly adapted bilinear | Adapt the existing head together with the same zero-start interaction residual | All 40 epochs completed; selected warm incumbent remains 15/40, zero new false Healthy |
| 224 structured-only RBF | Relative colour near ridge-like structures and margins may distinguish pale Healthy from deficiency cases | 13/40, zero new false Healthy |
| 224 native plus structured RBF | Combine the fixed spatial evidence with cached larger encoder features | 13/40, zero new false Healthy |

Thirteen Healthy cases are retained outside the recovery cohort, so 13/40 means
no added target recovery. Neither bilinear result is a newly trained improvement:
checkpoint selection retained epoch -1 in both modes. No validation-triggered
early stop was used; all 80 new epoch weights and predictions were saved.

Phase 222 used the four Fisher models and two final epoch-39 bilinear snapshots
as fixed negative checks on the 18 existing native coverage parents. The two
declared negative thresholds and targeted nitrogen/Other veto off/on settings
produce 432 policies. None passed; best remained 19/40 at zero new false Healthy.

Phase 225 applied that same fixed architecture to exactly the two phase-224
kernels: 144 policies, none passing, best again 19/40. There was no further
rank, C, bandwidth, feature, learning-rate or threshold grid. These families
are stopped. Failed models do not justify full encoder verification or export.

The consumed-label, unclipped ranking oracle union is 20/40 separately for the
phase-222 experts and the phase-225 kernels. These are optimistic diagnostics,
not validated thresholds or candidate policies. Individually, neither family
provides enough monotonic positive recovery to take the fixed 19/40 fallback
to 28/40: at most seven target recoveries beyond the fixed thirteen, even before
overlap with that fallback. This is not an impossibility claim for other models.

## Methods and identity controls

Fisher fitting uses all original feature dimensions, class means and a pooled
within-class covariance with trace-scaled numerical regularization. Its
OAS-style shrinkage uses an effective inferred-family count. This weighted,
family-adjusted rule is explicitly a heuristic, not a formal statistical OAS
guarantee. Binary and clinical-mixture controls each give Healthy half the prior
mass. Each calibration fit uses only its training fold.

The bilinear residual starts with a zero readout and preserves the warm head's
outputs and .95 admission decisions before training. Rank is fixed at 32.
Training uses supervised BCE plus the existing .2 hard-negative ranking term,
the same exact quarantined replay and cached right-angle rotation features.
Each photo gets a shared rotation across all ROIs and both views. A dedicated
CUDA generator keeps augmentation independent of frozen/joint dropout behavior.
The frozen-base control also disables base dropout. The training-EMA rule has
a 20-epoch minimum, patience 12 and cap 40; both controls reached the cap.

Phase 223 adds one label-free 69-dimensional descriptor of colour contrasts
around bright/dark ridge-like structures, edge/interior differences and support
statistics. The foreground and ridge masks are heuristics, **not clinical leaf
or vein annotations**. Four synthetic tests check units, finite output,
blank-input behavior, response to internal lines and quarter-turn stability.
No validation labels selected its parameters. The protocol was frozen before
new descriptor computation.

All 2,248 canonical native-518 input hashes were checked against reconstructed
pixels. Their new spatial descriptors are resumably cached by exact input hash
under the frozen descriptor protocol. The previous 2,106 global-colour vectors
were reused **bit for bit**; only newly added parent-224 identities required
those global descriptors. Source-file hashes and canonical-224 parent hashes
bind the additional computations. Existing encoder caches were reused;
**new encoder inputs: zero**.

The two supervised RBF controls have C=1 only. Bandwidth is the inverse median
squared photo-mean distance, fit separately within each training calibration
fold and the final training fit. No held-out feature values fit bandwidths.
Grouped training-only OOF Platt C=1 calibrates the margins. Whole publisher-93,
families, exact shared inputs and both views stay together. Training mass is
.5 target, .25 original replay and .25 field replay. Supervision applies to
actual ROI means per crop; individual ROIs receive no invented clinical labels.

Descriptor-only inputs use equal unit-norm global/spatial blocks. Joint inputs
give equal weight to DINO, ConvNeXt, global colour and spatial colour blocks.
Admission uses the minimum of both crop probabilities at .95, then the original
app guards. Invalid heuristic foreground in any actual ROI prevents new Healthy
admission without removing that photo from a denominator. No invalid foreground
occurred in this cache. Stored RBF arrays reproduced sklearn margins within
7.7e-13 and preserved all .95 decisions. These array checks are not full restored
encoder or phone verification.

## Evidence, backup and execution corrections

[Fisher/bilinear evidence](evidence/fisher_bilinear_controls_20261029/) contains
four frozen protocols, fourteen executed source files, runtime, results,
checkpoint-selection audits and a 188-artifact manifest. The independent laptop
backup `.toolchains/model-checkpoints/phase220-222/artifacts.tar` is 81,315,840
bytes, SHA-256 `bba1bf6b07c37479129bceea8cfe9aff5f5afe1620df6686f5d060ac0dd3482e`.
All 188 manifest payloads were verified by size and SHA, including all 80 epoch
weights and final optimizer/RNG states. Final model states matched epoch 39.

Two small corrections are explicitly separated from historical execution:

- Warm checkpoint BCE originally used an unweighted mean, whereas trained
  checkpoint tie-breaking used source weights. A cached-logit audit recomputed
  every comparison with the same source-weighted loss. Both corrected selections
  remain epoch -1 with unchanged gate results. Current source fixes this tie.
- The phase-222 optional veto used Python `~False`, yielding integer masks.
  The recorded truth values and counts were correct. Current source uses an
  explicit all-false boolean array to remove deprecated coercion.

Exact original PC bytes remain archived. The post-execution manifest pins both
historical and current source identities; do not silently replace historical
sources when reproducing frozen results.

PC disk had about 940 MB free and its RAM disk about 62 MB free. No running jobs
were interrupted. Only after the independent backup and every member SHA passed,
the 84 bilinear PT files were hash-checked again and offloaded from their explicit
RAM directory. Results, protocols and all prediction caches remain on the PC.
Restore the named PT members from the laptop TAR before loading these particular
weights again. No datasets, encoder caches or unrelated files were removed.

[Spatial evidence](evidence/structured_controls_20261029/) contains three frozen
protocols, nine executed source files, cache identities, numerical restore
checks and results. Its private laptop backup
`.toolchains/model-checkpoints/phase223-225/artifacts.tar` contains all 12
artifacts, 21,329,920 bytes, SHA-256
`3b831fa88fb3f5765f5d06ebe3091fb0fac25d129f18f819c22edb9e3ae67cdd`.
Every manifest payload was size/SHA verified. Learned arrays, source photos and
weights remain private and are not committed. PC RAM outputs remain volatile.

The full local Python suite passed **331 tests**, with one optional media skip,
in 26.829 seconds. New checks cover discriminant geometry, zero-start/frozen
adapter behavior, synthetic spatial descriptors, explicit RBF restoration,
training fold isolation and exact source/protocol/result identities. The prior
published head 313192b passed both training-contract and native Android CI.
This batch changes research/evidence only; the verified app artifacts remain
unchanged. Final-head CI is recorded separately in the PR.

Measured new descriptor extraction took 117.36 seconds. Both kernel controls,
including grouped calibration and array restoration, took 4.71 seconds together;
their routing replay took .027 seconds. Full-feature mobile Fisher fitting took
4.46 seconds. Verification, tests and transfer time are additional. These numbers
support quick bounded cached checks, but do not provide a credible completion
date for recognition while every candidate still fails.

## Continuation constraints

Do not restart these finished families. Check running jobs and exact evidence
identities before any continuation. Preserve the full 40-case denominator and
all original full qualification requirements. The deployed full OLID baseline
remains 87/223 Healthy with 16/507 disease-to-Healthy errors; the 19/40 research
policy has not passed full qualification and is not bundled.

Further work needs a justified new source of discriminating evidence rather
than more epochs or thresholds on the failed score families. A blinded eight-case
clinical-review packet exists locally but has not been adjudicated; it cannot
justify label changes. The provisional acceptance reservation must remain
diagnostically sealed until a candidate and protocol are frozen. Physical-phone
verification and owner release signing remain outstanding for app delivery.
