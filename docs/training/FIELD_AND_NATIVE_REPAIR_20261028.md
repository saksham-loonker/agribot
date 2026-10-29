# Field supervision and native-resolution repair continuation

This continuation supersedes the phase-158 stopping point in
[the earlier repair record](RECOGNITION_REPAIR_20261028.md). No replacement
model has qualified, and none of these research weights is bundled in Android.
The original app 2.0.4 delivery and its runtime evidence remain valid for the
original model only. AD and OLID remain permanently consumed development data.
TCP remains provisional and diagnostically sealed; its geometry can only be
used for overlap exclusion. No new independent acceptance result is claimed.

## Fixed decision contract

Selection measures actual app verdicts, including the existing guards and
retained verdicts. The cheap screen requires at least 28/40 consumed validation
Healthy images, counting all 40 rather than only eligible cases, and zero new
disease-to-Healthy errors in the applicable validation sources. The retained
13 Healthy verdicts remain part of that count. Passing this screen is necessary
but insufficient: the complete consumed Healthy cohort needs at least 157/223,
the paired gain and family-bootstrap requirements still apply, and every
original-source, named-disease and safety retention check must pass. In
particular, do not remove `original_disease_and_healthy_exact` from
`OpenedGuardScreen`. No threshold relaxation, failed-model export, or Android
replacement is authorized by these results.

## Additional source and provenance

The separately published [Tomato disease collection, version 1](https://data.mendeley.com/datasets/93h9p62kg4/1)
(DOI 10.17632/93h9p62kg4.1, CC BY 4.0, Ismail Hossain et al.) supplies additional
field supervision. The entire publisher collection was assigned to development
before downloading pixels, including any unused remainder. It is never fresh
acceptance. Its ten labeled folder inventories contain 2,627 records; this does
not establish coverage of any separately described unlabeled Other category.

Phases 161–163 downloaded all 103 available Healthy images and 32 images from
each of nine disease folders: 391 images, 66,271,843 bytes. Every file passed its
publisher SHA-256 and size check. A transient HTTP-200 JSON error body failed
the hash check; resuming reused verified files and fetched the missing images.
It was not accepted as an image. The downloaded set has 391 unique decoded
images, 377 approximate photo families, no exact-label conflicts and no
quarantined files. Healthy examples have iPhone 11 metadata and landscape
orientation; camera/source confounding and specimen identity remain limitations.

An exact/approximate geometric comparison against 40,625 existing gallery
entries, including the protected reservation's geometry, found zero overlaps
for these 391 queries. This is an exclusion audit, not proof of specimen or
capture independence. Protected clinical labels and predictions stayed sealed.
Known matching conditions retain their clinical labels; other disease folders
become `Diseased_unspecified`, never Healthy, Other, or an invented condition.

Phase 164 selected 244 training photos, including one image from each of 100
approximate Healthy families. Actual detector ROIs produced 488 canonical
center/full inputs. Their original mobile DINO/CNN, larger DINO and ConvNeXt
features were computed once with exact weight/runtime/input identities and
preflight comparisons against older matching caches. Subsequent frozen heads
reuse these features. The run took 34.66 seconds.

## Completed bounded controls

Counts below are selected app-verdict Healthy out of the full 40, followed by
new disease-to-Healthy errors. A selected incumbent of 14/40 with zero errors
can mean the adaptation never improved its safe starting point.

| Phase | Change | Healthy / new false Healthy | Decision |
|---|---|---:|---|
| 159 | Image-level disease-witness supervision across real ROIs | 17 / 1 | Stop |
| 165 | Joint semantic/texture head with new field supervision | 14 / 1 | Stop |
| 166 | Same field data, image-bag witness objective | 16 / 1 | Stop |
| 167 | Mobile-compatible head: supervised / ranking | 13 / 0; 14 / 0 | Stop both |
| 169 | Earlier CNN tail: supervised / photo-level MixStyle | 16 / 1; 18 / 1 | Stop both |
| 171 | Remove continued camera-confounded source-107 replay | 14 / 1 | Stop |
| 173 | Native-518 DINO plus existing ConvNeXt texture | 14 / 0 | Stop head |
| 174 | Supervised final-two-block Q/V LoRA | 14 / 0 incumbent | Stop adaptation |
| 175 | Attention over cached original native local tokens | 14 / 0 | Stop |
| 176 | Expected admission-reward continuation from unsafe adapted tail | 14 / 0 incumbent | Stop reward |
| 179 | Restore all 71 omitted labeled field-training examples: supervised / ranking | 13 / 0; 13 / 0 | Stop both |
| 180 | Group-calibrated native semantic/texture RBF: C=1 / C=10 | 13 / 0; 14 / 0 | Stop both |
| 181 | Full final-two-block supervised adaptation | 14 / 0 incumbent | Stop adaptation |
| 182 | Full-field whole-image/ROI RBF: C=1 / C=10 | 13 / 0; 13 / 0 | Stop both |

MixStyle follows the [original method's feature-statistic mixing idea](https://arxiv.org/abs/2104.02008),
with one label-independent donor and coefficient shared across a photo's actual
ROIs. It does not assert that arbitrary transformations preserve plant symptoms.
The matched no-style control also failed. The phase-168 prefix's executed
architecture is MobileNetV3, old cut 4 to new cut 3, shape 40×14×14. Its historical
“stage five/EfficientNet” prose is incorrect; do not reinterpret the weights or
invalidate the measured exact feature parity based on that prose.

The native-518 cache uses original source pixels, actual cached detector boxes,
and the official source's original 37×37 positional embedding grid. It is not a
518px upscale of cached 224px tensors. Its 2,106 new native identities took
228.40 seconds; the original 224px feature preflight differed by at most
0.00001419. CNN texture remains linked by the exact parent 224px input hash.
No row-number joins across cache namespaces are permitted.

The adapted native tail reached 27/40 Healthy with three new false Healthy at
one epoch, so increased recall did not qualify it. That epoch's weights were
not saved; only the safe incumbent and final checkpoint were saved. The final
unsafe checkpoint's 24/40 and three new false Healthy predictions are cached
explicitly in phase 176. Expected reward uses consumed training labels with an
eightfold false-Healthy cost, not an online environment or acceptance feedback.
It did not repair the safety/recall tradeoff. The cached local token grid is from
the original frozen encoder, not from adapted weights.

## Diagnostic findings and rejected explanations

Phase 160's committee feasibility test used an inward numerical margin and
therefore was not a universal upper bound at the exact admission boundary.
Phase 170 corrects that limitation with outward-relaxed inequalities and
optimistic treatment of ties. Both probability and clipped-logit convex mixtures
have an optimal relaxed ceiling of 23/40 on the explicitly scoped sets of 35
and 41 selected prediction caches. Validation-derived mixtures are diagnostics,
never candidates. This result does not cover later checkpoints, arbitrary
nonlinear policies, unselected epochs, or future representations.

Phase 177 tested a suspected mismatch between eligible training examples and
the final retained-verdict guards. All 271 selected training examples were
actually admissible, with zero selection mismatches in training or validation.
That hypothesis was rejected. Do not claim that a selection bug caused these
failures or rerun the identical subset as a corrected experiment.

Phase 178 measured an explicitly label-oracle routing ceiling, without fitting
or promoting a policy. In the full consumed Healthy cohort, 87 are fixed
Healthy, 107 additional images are guard-recoverable and 29 are blocked; the
ceiling is 194/223. In validation, 13 are fixed, 18 additional images recoverable
and nine blocked, for a ceiling of 31/40. All nine blocked validation images fail
the existing whole-image Healthy guard. They remain in the denominator.
The ceiling exceeds the gate, so unchanged guards alone do not explain failure.

The same audit found 358 quarantined available field-training images, of which
bounded replay retained 287. Omitted supervision comprised 66 Healthy, three
potassium-deficiency and two unspecified disease images. Phase 179 restores all
71 using existing mobile features and the original validation exclusions. Its
two failed controls do not support missing replay as a sufficient explanation.
Earlier “full-source” controls referred to source 107 and must not be described
as having covered these omitted field images.

## Execution, stopping and continuation

Frozen heads take roughly three seconds; the native supervised and reward
controls take roughly three to four minutes including the missing GPU prefix
representation. All exact matching image features and predictions remain cached.
The 12GB native prefix is kept on GPU during a run and reused every epoch; it
cannot be reconstructed from CLS vectors, so a new process regenerates only
that unavailable intermediate. Existing disk and shared-memory caches are
preserved. The persistent GPU machine has little free disk; preserve the 1GiB
disk reserve, and do not delete useful runs to start another experiment.

Phase 181 tested the capacity limit of low-rank adaptation by updating every
parameter in the final two transformer blocks, using a tenfold lower encoder
learning rate and the same data/objective/guards. It stopped after eight epochs
in 188.14 seconds. Its 27/40 Healthy epoch added six false Healthy. The safe
14/40 warm incumbent was retained; no epoch improved it safely. Phase 182's
training-only calibrated whole-image/ROI kernels also stopped below the gate.

The phase-181 407MB research checkpoints are backed up under the laptop's
`.toolchains/model-checkpoints/phase181/`, with SHA-256 values recorded in
`phase181_native_full_tail/checkpoint_backup.json`. The GPU host's output and
atomic checkpoint temporary files were redirected to dedicated RAM storage
before training writes, without interrupting the process or evicting old caches.
Its small JSON/NPZ evidence is also copied to the persistent
`phase181_initial_protocol_preflight` directory. RAM copies do not survive a
host reboot: restore them from the verified laptop backup if needed. Do not
silently rerun a completed failed experiment to recreate these files. The safe
checkpoint is an incumbent copy, not a qualified replacement.
Do not continue unsuccessful fixed-objective families without a new, measurable
hypothesis. Do not run restored full-model verification or mobile distillation
for a candidate that fails the cheaper actual-verdict screen.

Source protocols, results, provenance and exact executed source bytes are in
[the evidence directory](evidence/recovery_controls_20261028/summary.json).
The full local suite passes **270 tests**, with one optional downloaded-media
skip, in 21.769 seconds. Tests cover image-bag supervision, committee numerical
bounds, photo-level style mixing, exact native feature joins, adapter/full-tail
gradient flow, expected reward signs and final-guard admission behavior.
All software tests are contract checks, not recognition acceptance. Suitable
unconsumed prospective acceptance and the disconnected phone's final checks
remain outstanding. No release-ready recognition claim follows from this work.
