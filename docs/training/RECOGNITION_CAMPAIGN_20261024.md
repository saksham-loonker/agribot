# Recognition repair campaign: 24 October 2026

**Latest checkpoint:** [cache-first optimization and stopping decision](TRAINING_OPTIMIZATION_20261026.md). The frozen AD opening failed and AD is now consumed development data. Current supervised/hybrid policies fail the unchanged app gates, so full verification, export and Android integration are deferred. OLID remains sealed; the disconnected phone and installed model are unchanged.

**Latest continuation, 26 October:**
[Apache-only SigLIP2 investigation](SIGLIP2_CONTINUATION_20261026.md).
The full Apple model license excludes product development, so the MobileCLIP
branch is a research record and cannot supply app weights or replacement-model
teachers. Human labels are reused independently. No new model is qualified;
independent AD diagnostic predictions remain unopened.

This record extends the recovery and app validation reports. The owner's request
is to improve Healthy recognition and false rejection without sacrificing the
existing disease recognizer, and to test supervised, logit and reward methods
individually. A higher crop accuracy alone does not qualify a model for the app.

The campaign runs on the owner's RTX 5090. Raw data, pixel caches, checkpoints,
per-photo predictions and full logs remain under
`~/agribot-data/recognition_campaign_20261024/`. They are not committed to Git.
The original trainable classifier is retained as the reference and rollback.

## Current checkpoint: detector, ROI and image-context diagnosis

### Newest continuation: alternate features and explicit benchmark retirement

Read [the MobileCLIP continuation](MOBILECLIP_CONTINUATION_20261025.md) first.
The 285 valid controls include a five-expert image prototype that passes
development with 94/129 TOM Healthy and 112/119 paper Healthy, then reproduces
all 6,468 exact-graph verdicts. It still fails the consumed field cohort at
51/95 Healthy, with false Healthy unchanged at 3. The separate AD source remains
unopened diagnostically. Eight invalid new MobileCLIP fits were caught by
cache/runtime parity, quarantined and retrained; they are excluded from the count.

Phase twelve admits the consumed cohort's 787 related photos/235 families to a
new grouped image-scoped training/development revision, preserving all 200 old
quarantines and original development membership. Those families are retired as
benchmarks for this and later revisions. New controls run against the original
eight regression cohorts, with independent AD acceptance still pending. Both
APK builds pass and retain interim weights; no failed candidate is installed.

### Newest continuation: verified leaf supervision and aggregate disease priority

Read [the leaf-scope continuation](LEAF_SCOPE_CONTINUATION_20261025.md) for the
complete protocol and reproduction commands. FieldPlant's 1,403-photo download
and 2,786-leaf annotation audit are complete. Of 141 photos with a Healthy leaf,
128 also contain unhealthy leaves; whole-photo labels cannot be broadcast to
every detected leaf. The new leaf cache masks all 2,858 TOM weak fitting targets
and excludes the later teacher-derived positives, while preserving app regression
images/labels. There are 269 completed controls; the first leaf-supervised copied
tail also fails end-to-end recall. A separate photo degradation control is
running. The six leaf CNN/genus/frozen-feature additions also fail field recall.

An opt-in schema-four, 18-output native/host contract accumulates proposal and
frozen reference evidence on the same real observations. A reference aggregate
named disease takes priority, preventing recovered Healthy votes on other leaves
from outvoting it. Existing installed assets do not enable this contract. Native
unit tests (105), release lint and test-APK assembly pass. The Python suite ran
121 tests: 120 passed, one skipped. The cold Abu Dhabi source remains unopened
diagnostically, and the phone remains on 2.0.2. Older pending statements below
are historical; consult the newest evidence and continuation first.

### Latest continuation: stronger supervised controls and better scoped data

There are now **262 completed trained controls**. This includes four additional
supervised scene experiments and their twelve separately fitted stackers, beyond
the preceding 246. Threshold comparisons are not training runs. The original
nine-class disease branch remains frozen in these experiments.

| Scene adaptation | Opened TOM2024 Healthy | False Healthy on that cohort | Decision |
|---|---:|---:|---|
| Interim 2.0.2 | 8/95 | 3 | Installed comparator |
| Earlier focal CNN prototype | 44/95 | 3 | Failed prospective recall requirement |
| Four copied DINO blocks, focal | 51/95 | 4 | Fails opened-regression retention |
| Twelve copied DINO blocks, focal | 55/95 | 4 | Fails opened-regression retention |

The four-block learner has 7,102,081 trainable parameters; the twelve-block
learner has 21,303,937. Both start from the owner's trained DINO weights, not a
random model. Full-image letterbox and horizontal-flip prefix caches preserve
alignment. Training uses FP16 token storage/BF16 computation; validation remains
FP32. The full-depth experiment adds dropout 0.1 and weight decay 0.05. A further
full-depth mixup control uses image-equivalent mixing at the zero-transformer
prefix; it fails the development field-recall target (72/129, 55.81%). It was not
opened on new acceptance data. The raw leaf-policy probe recognizes only 37/95
Healthy photos before context routing, so relaxing that routing alone is not a
sufficient explanation or fix.

All those runs and replays are complete. `phase9_scene_sft.py` supports exact
optimizer/RNG recovery and independently named runs; `phase9_pipeline.py` fits
development calibration/fusion and replays the consumed reservation without
opening fresh data or promoting a model. The latest local suite passes 119 tests
with one skip, including gradient isolation of the disease reference and the
image/token mixup equivalence. Android Native CI passed commit `8f6fe18`.

Two source investigations follow from the generalization failures:

* The Abu Dhabi greenhouse source, DOI `10.6084/m9.figshare.31145998.v2`, was
  downloaded and verified against publisher MD5. Its ZIP SHA-256 is
  `21a60dc473ae3e072a8542a0e53b1e1b56cd7245481ba5db25ba6ad4ccbe2a62`.
  It contains 502 primary-leaf images: 104 Healthy, 11 Early Blight, 96 Bacterial
  Spot and 291 Mosaic Virus. License: CC BY 4.0; publisher: Muhammad Shafay.
  Standard fingerprints alone miss overlapping crops. SIFT/RANSAC found 68
  geometric overlap links, merging filename families into 68 conservative
  capture families, including 20 Healthy families. These are not verified plant
  IDs. Partial crop overlap does not prove incompatible primary-leaf labels.
  Prior-overlap checking uses 30,781 file hashes and 34,896 perceptual hashes
  from full audits, not only the fingerprints retained in compact training rows.
  Sealed geometric manifest SHA-256:
  `5cf37861dfe835c918643b2753a05c229e9327cd750524435006384bdcfae6b6`.
  **No diagnostic predictions or fitting have used this source.** Preserve it
  for a later frozen, family-aware evaluation; it is not an adjudicated
  whole-plant-health dataset.
* A publisher-linked SFLD original-tomato Healthy folder exposes only 14 files
  through the Drive listing. Its raw-image count is insufficient to serve as a
  substantial new Healthy acceptance cohort. No pixels were downloaded or fitted.
  M-TLD's public repository instead asks readers to contact its authors for data;
  no such message was sent.

The current acquisition is **FieldPlant**, whose original publisher describes
pathologist-supervised annotations of individual leaves. The public mirror's
embedded README identifies
https://universe.roboflow.com/plant-disease-detection/fieldplant and CC BY 4.0;
the export records auto-orientation and no augmentation. Its tomato subset has
1,403 photos and 2,786 annotated leaves, including 279 Healthy instances. The
remaining categories are broad symptom/condition names such as `Tomato blight
leaf` and `Tomato Brown Spots`. Do not infer Early versus Late Blight, or map
every diseased label to nine-class Other. Correctly scoped binary negatives can
be admitted only after annotation-role and overlap audits.

`phase10_fieldplant.py --campaign <artifact-root> --download` retrieves just the
tomato ZIP members with HTTP Range, checking each member's CRC. The pinned
5,912,207,630-byte archive's full SHA-256 is not computed; source export metadata,
catalog identity, CSV SHA-256 and per-member checks record the subset's lineage.
The raw CSV SHA-256 is
`1b069168e3a982d98020fdf0e121305c84dc84880ebce543077a67d2cd83b433`.
Acquisition log: `phase10_fieldplant.log`; root: `phase10_fieldplant/`. Download
completion does not authorize training: first inspect actual boxes, quarantine
prior overlaps, and keep capture families together. Healthy image labels must
not be applied to every proposed background leaf. Explicit Healthy leaf boxes
are stronger supervision than the earlier weak all-proposal positives.

The phone remains on 2.0.2. No failing prototype has been installed, and the new
whole-frame policies still have no native export/runtime implementation.

### Latest decision: prospective acceptance failed

Both scene-negative continuations completed. There are now **246 trained
controls**, including six new fitted stackers; policy grids and the 144-way
union ablation are not additional training. The focal continuation's selected
policy (candidate 421) passed every development retention check and reached
91/129 TOM2024 Healthy and 117/119 paper Healthy, with false Healthy 10 versus
11. The executable PyTorch prototype reproduced 111 frozen development checks;
maximum context-logit discrepancy was 0.00003338. Its policy, model/calibration
identities and acceptance requirements were frozen before reserved inference.

The 246-photo prospective reservation was then opened **once**. It failed the
70% Healthy recall target:

| Reserved source/task | Interim 2.0.2 | Frozen prototype |
|---|---:|---:|
| TOM2024 Healthy | 8/95 (8.42%) | 44/95 (46.32%) |
| TOM2024 false Healthy | 3 | 3 |
| Paper-leaf named disease correct | 12/28 | 12/28 |
| Paper-leaf false Healthy | 0 | 0 |

Per-source relative retention passed. Paired Healthy improvement is 37.89
percentage points; the inferred-capture-group bootstrap 95% interval is
[28.42, 47.39] points. Candidate Healthy recall Wilson 95% interval is
[36.63%, 56.29%]. The improvement is substantial but insufficient. **Reject for
deployment.** The phone remains on 2.0.2; there is no new native model or APK.
This reservation is now opened regression data. Never tune on its outputs and
call a subsequent result independent acceptance. Other photos in the same
reserved capture families do not constitute fresh independent groups.

Post-opening diagnosis attributes 49 of the 51 missed Healthy photos to the
image-context model not agreeing Healthy, and two to species rejection. None is
explained by the protected named-disease routing rule. This points to learned
field generalization rather than a simple app threshold or normalization fix.
The old benchmark disease results and the 12/28 cross-source disease result
also demonstrate the limits of benchmark transfer.

Audit files: `phase6_roi/phase8_acceptance/{frozen_protocol.json,
opened_identity.json,development_prototype_parity.json,predictions.jsonl,
result.json,failure_breakdown.json}`. The next justified model experiment is
supervised image-context adaptation with more trainable backbone capacity while
keeping the original nine-class disease branch frozen. A new independently
reserved, correctly scoped field source is required before any later promotion.
No training is currently running from `phase7_negative_controls.py`.

The material below records the preceding checkpoints and their then-unopened
reservation. The latest decision above takes precedence.

This checkpoint supersedes the historical progress notes below. Field recognition
is **not resolved**. The phone still has interim 2.0.2, classifier SHA-256
`d70135804cf750341218e89a055370bef46a9b337951897b9f55f2570844ed39`.
No phase-five, six or seven candidate has been installed or merged into main.
Draft PR: https://github.com/saksham-loonker/agribot/pull/6.

The campaign has completed 238 trained controls before the two image-context
continuations described below. Threshold grids are policy comparisons, not
additional trained models. The methods include supervised heads, partial and
LoRA adaptation, CNNs, boosted trees, logit adjustment, monotone logit fusion,
label-derived expected reward, REINFORCE, PPO and DPO controls. The latter use
label-derived one-step rewards; they are not RLHF. No result establishes a
0–255 versus 0–1 normalization bug: canonical preprocessing and exported-byte
parity have been checked. Both matmul and cuDNN TF32 must be disabled for strict
FP32 comparisons.

### What the additional experiments established

| Experiment | Opened development result | Decision |
|---|---|---|
| Guarded 12-block LoRA plus Healthy/knownness CNN, including Unknown confirmation | Healthy 65/248 to 149/248; false Healthy 11 to 8; relative regression gates pass, but TOM2024 Healthy remains 41/129 | Insufficient field recall; research export only |
| Original leaf detector adapted with weak Healthy leaf proposals | Original-target detector validation retained; complete app replay Healthy falls from 65 to 62/248 | Reject; better box recall did not improve recognition |
| New detector plus guarded classifier | Healthy 137/248 versus classifier-only 149/248 | Reject detector change |
| Teacher ROI augmentation plus deeper supervised adaptation | Guarded Healthy 174/248; false Healthy 19 versus 11 | Reject as deployment policy |
| Whole-image logit ensemble and strong-negative abstention | Best relative-safe policy: Healthy 203/248, false Healthy 9 versus 11; TOM2024 87/129 (67.44%), paper leaves 116/119 (97.48%) | Below 70% field Healthy development target |
| Less restrictive ensemble, candidate 216 | Healthy 214/248; TOM2024 97/129 (75.19%); paper leaves 117/119; total false Healthy 11 | Fails one unsupported-condition regression: 33 to 34 false known calls in phase-two cohort |
| Require detected-leaf Healthy agreement in both crop variants | Rejects the new unsupported error, but field Healthy recovery collapses | Reject; do not impose this rule on the app |

These are repeatedly inspected **development/regression** comparisons. They are
not independent acceptance accuracy. The 246-photo prospective reservation is
still unopened. Neither selecting a threshold nor completing training qualifies
a candidate for installation.

The unsupported regression is an image labeled bacterial spot whose focused
leaf classifier remains ambiguous while the whole-image ensemble is confidently
Healthy. It stays in evaluation; do not relabel, exclude, whitelist or identify
it by filename in runtime code. A source-label mismatch is possible in public
data but has not been established by expert adjudication. The next controls
address hard negatives generally rather than treating that image as training.

### Data and annotation scope

Grounding DINO tiny (`IDEA-Research/grounding-dino-tiny`, pinned revision
`a2bb814dd30d776dcf7e30523b00659f4f141c71`) was audited on training images only.
The prompt `a leaf.` was approved for limited weak geometry proposals, not ground
truth. It found proposals on 327 of 365 Healthy TOM2024 training photos; 1,200
usable ROIs across 319 photos add 2,400 center/full-ROI training views. Rows now
total 63,265, SHA-256
`687829881aff3317ef996a6265012c5e3a39b687343ff47bf85ce14ecdd196eb`.
The image label supplies Healthy supervision; the teacher only supplies geometry.
Unverified field ROI genus is masked in species training.

A separate source (`10.17632/bpfd9cns5g.1`) remains forbidden for detector
training: its class mapping is missing and inspected boxes mix spot regions and
whole leaves. The older SLIF detector trial also remains quarantined. Never
compare detector mAP across those incompatible annotation targets.

Whole-image training has 22,472 photos, row SHA-256
`e063bafa83acc1c2fbd588a88042a5677c5dd7321b76fb007c07c361325e9dca`.
Healthy versus non-Healthy image labels are weak scene supervision. Any annotated
unhealthy instance makes the scene negative. An image labeled Healthy does not
prove every visible leaf, or the entire plant, is disease-free. Whole-frame
evidence must never be presented as fabricated leaf detections or leaf agreement.

Separate MobileNet Healthy-image and tomato-species models, five frozen-feature
binary heads and three monotone logit stackers completed. Stackers combine two
DINO Healthy heads, a scene Healthy CNN and a species CNN. They fit 2,644 named
calibration groups' photos and select on 1,280 other development photos; base
checkpoint selection used broader validation, so this is not an independent
split. All model identities, calibration hashes and input order are saved.

### Exact continuation and deployment requirements

Current controller on the PC:
`~/agribot-data/recognition_campaign_20261024/phase7_negative_controls.py`.
Its tmux window is `scene-negatives` in session `agribot`; controller log is
`phase7_negative_controls.log`. It sequentially trains `scene_ce_n8` (CE, negative
cost 8) and `scene_focal_n4` (focal gamma 2, negative cost 4), each for 16 epochs
at learning rate 0.00003 from the existing scene model. It then recalibrates,
fits separate monotone stackers and replays the opened app cohorts. Check logs
and completion files before launching anything again. These runs do not open
the reserved photos.

Research root:
`~/agribot-data/recognition_campaign_20261024/phase6_roi/`.
Inspect `fusion_selection/`, `fusion_leaf_selection/`, and the controller's
`scene_*_selection/` directories. Source scripts live in `model_training/v3/`.
`phase7_fusion_select.py --inspect-candidate ID` records changed predictions for
audit without overwriting the policy ranking. Match its `--run-name` and
`--fusion-run` when inspecting a nondefault control.

The native optional schema-three path is implemented and unit-tested: 18 raw
logits preserve the nine-label proposal and frozen reference. The additional
`changed_decision_full_roi` policy confirms changed Unknown decisions too.
Actual-byte replay of the phase-five open-set export passed 7,793 crop checks
(maximum prototype/reference discrepancy about 0.000337). That is numerical
parity, not sufficient recognition quality. The newer whole-frame context
policies are **research Python only**; there is no schema-four export or native
whole-frame evidence implementation. Do not ship their weights or claim the
phone already runs them.

Before promoting a context candidate: freeze its policy and acceptance protocol;
implement explicit image evidence without fake leaf counts; preserve named
disease decisions and abstain on contradiction; export and replay actual bytes;
open the prospective reservation once; report denominators and source scope;
then build and run parity, navigation and latency checks on the connected phone.
Keep the current app data and rollback artifacts. Passing development gates
alone is insufficient.

The latest GPU Python suite passed 115 tests with one skip; the corresponding
laptop rerun is recorded separately. Native ML unit checks pass for the optional
Unknown confirmation changes. Earlier full release lint/build checks passed for
the interim assets. Tests verify implementation, not field generalization.

## Earlier continuation: broader supervised adaptation

The phase-four confirmed prototype was rejected on its first opening of two new
classification sources. TOM2024 Healthy remained 9/100; paper-background Healthy
improved from 46/97 to 55/97. The combined gain was 4.57 percentage points, below
the frozen 5-point minimum, and paper Healthy recall remained below 70%. Safety
counts were retained, but this is insufficient improvement. It was not installed.
These opened photos are now development data, not independent acceptance.

Phase five acquires the complete relevant leaf classes from TOM2024
(DOI `10.17632/3d4yg89rtr.1`) and the paper-leaf source
(`10.17632/swtt5wy2pr.1`). Fruit classes are excluded. Generic virosis is binary
diseased supervision only: it must not become an invented named disease or a
nine-class Other target. The complete resized archive was verified against its
publisher SHA-256 before extraction. All source licenses are CC BY 4.0; the
publisher metadata and attribution remain in the remote source manifests.

Before inference or fitting, capture families were prospectively reserved from
previously unopened photographs. TOM2024 uses shared three-minute filename bursts
across classes; paper photographs use inferred adjacent Object_ID groups.
Perceptual matches merge groups; conflicting inputs and overlap with earlier
data are quarantined. These are inferred capture families, not verified plants.
Of 5,214 source photos, 4,227 enter development, 200 are quarantined, and all 787
photos related to reserved groups are excluded from training/calibration.
The reserved evaluation selects 246 photos across 235 groups, including 95
Healthy TOM2024 photos. It is a same-source prospective reservation, not a new
independent source. Its predictions remain unopened at this checkpoint.

Reservation SHA-256:
`ca91cf80b05eab1c733a54746542e8fe37bee503c7c0fe03a97022bcdd0f5dd1`.
Development manifest SHA-256:
`249a8a4e199d38dcffaa6fc21ffc7da8ee834da7c19b28545af6f76b1262812d`.
The complete 60,865-view cache references immutable earlier pixel shards and
adds 8,400 views. Row SHA-256:
`b95227351e39b3f58a68b29437dd8744e17fee6c8422cfa96d7387dd74e970c1`.

The source-balanced CNN verifier completed 16 epochs. A rank-8 LoRA student
extends the existing four-block adapter to all 12 DINO blocks while preserving
its trained tail exactly at initialization. It completed 24 epochs with 593,289
trainable parameters. The copied nine-class head remains; the loss combines
supervised CE, binary Healthy supervision and reliable-teacher retention. Coarse
diseased images skip nine-class CE. Training uses BF16; validation/export use
FP32 with both matmul and cuDNN TF32 disabled. Neither standalone student passes
old disease retention, so it cannot be promoted alone. Actual app replay of a
guarded candidate is required before the reserved evaluation is opened.

The optional schema-three runtime emits 18 logits: nine proposed and nine frozen
reference logits, preserving the nine-label vocabulary. When a new positive leaf
decision differs from the reference, a second classification preserves the full
padded ROI with letterboxing. Agreement accepts the new decision; disagreement
uses reference logits. Schema-two models keep the existing path. This addresses
observed center-crop sensitivity; it is not evidence that normalization was wrong.
The runtime source is implemented and unit tested, but this candidate has not
been installed. The phone still runs the tested interim 2.0.2 model.

The latest Python suite passes 112 tests (one skipped) on the laptop's project
virtual environment. A Windows-only pixel-cache cleanup failure was corrected
by explicitly closing memory maps; the GPU checkout's earlier 111-test suite
also passed, including the exact published interim assets. Test results establish
implementation behavior, not field recognition quality. Historical sections below
retain their original cohort-opening chronology; the status above takes priority.

### Subsequent phase-five validation and next detector audit

The deeper student alone still fails old disease retention. The guarded policy
recovers up to 165/248 Healthy validation photos from the interim baseline's
65/248, but adds ten false Healthy calls (11 to 21), so it is rejected. Stricter
verifier settings reduce the added errors while giving up Healthy recovery;
none passes the existing gate. Full-ROI-primary and whole-image-context ablations
also fail safety checks. These are development comparisons, not acceptance.

The context audit separates an important pipeline effect: the raw adapted model
recognizes 115/129 TOM2024 Healthy validation photographs from the full image,
while the guarded leaf-crop pipeline recognizes 57/129 in the corresponding
candidate. There are 23 no-leaf results. This supports further investigation of
crop selection, missing detections and aggregation; it does not prove that the
detector alone causes every rejection. Whole-image agreement is not deployed.

A further eight-epoch supervised continuation raises the negative Healthy loss
cost from 2 to 8, raises binary-loss weight from 0.5 to 1, and uses learning rate
0.00003. It completed in 482 seconds. It also fails the guarded validation gate.
This brings the completed, valid trained controls to 225. The wrong-target
detector trials are excluded from that count. RL controls remain separate
historical experiments; no human-feedback reward signal exists in this campaign.

A twelve-image annotation overlay audit of the additional public source
`10.17632/bpfd9cns5g.1` finds mixed geometry: some annotations cover whole leaves,
while others mark small spots/regions. Its class-ID disease mapping is absent.
Do not remap all IDs to generic leaf boxes. This source is not admitted to
classifier or detector training. Publisher attribution: Ahmed Imtiaz, Fahad Bin
Islam Swapnil, Syed Rayhan Masud and Debajoyti Karmaker; CC BY 4.0.

The next controlled audit uses the original research authors' Apache-2.0
`IDEA-Research/grounding-dino-tiny` model, pinned to revision
`a2bb814dd30d776dcf7e30523b00659f4f141c71`, to propose whole-object leaf boxes.
`phase6_teacher.py` first selects 16 development/train images and renders boxes
for visual target verification. No pseudo labels enter fitting before that audit
passes. They remain model predictions, not human ground truth. The script never
reads reserved pixels or promotes a phone model. Source and usage documentation:
https://huggingface.co/IDEA-Research/grounding-dino-tiny .

The work is pushed as draft PR #6:
https://github.com/saksham-loonker/agribot/pull/6 . Its initial Android Native CI
run passed. It is intentionally unmerged while recognition acceptance is unresolved.

## Continuation on 25 October: installed baseline and further adaptation

The interim guarded CNN model was installed as app 2.0.2 (version code 4).
Classifier SHA-256:
`d70135804cf750341218e89a055370bef46a9b337951897b9f55f2570844ed39`.
Its 54 export parity cases passed with maximum absolute logit error
0.00001383. The phone passed 15 instrumentation checks, including canonical
goldens, downloaded photos, persistence/export and navigation. These establish
runtime correctness; they do not establish field accuracy. The obsolete field
pilot launcher was disabled with its data preserved to avoid opening a different
model accidentally.

That interim model improves the older public Healthy result from 64/80 to 67/80,
and the opened binary-source Healthy result from 14/59 to 20/59. It preserves
the older named disease counts. However, on the further 821-photo classification
cohort it recognizes only 27/118 Healthy leaves (original 26/118), and 147/470
supported disease photos. This is inadequate generalization, even though the
strict regression gates passed. The task is not considered resolved.

After opening that cohort, its complete publisher source was explicitly admitted
to development. It is no longer independent evidence. Of 2,995 downloaded images,
2,851 remain eligible after overlap/conflict quarantine. Capture families and
close perceptual matches are grouped before splitting. The resulting development
cache has 52,465 views, including 6,155 additional views; 459 new-source photos
are held in grouped validation. Neither crops nor augmentation siblings may
cross those groups. Immutable old pixel shards are referenced rather than copied.

The `phase2_source_manifest.json` SHA-256 is
`f10fae17bb7f4f42b4929589c13d5a4f0c04f34855dfc413f4a23e083a364bc4`;
the complete view-row SHA-256 is
`f941eb30876a224a25e4155fc83dad41231c43d8a9fc1bbc5f498799f26cc40d`.
Source membership, hashes and the admission history live with the remote artifacts.

Further trained controls preserve the original encoder and head as a separate
reference branch. Two warm-started CNN experts use BCE/focal losses and doubled
negative Healthy cost. Three nine-class students start from the original head:
rank-8 LoRA in the last block (52,617 trainable parameters), a copied full last
block (1,778,697), and rank-8 LoRA in the last four blocks (200,073). The deeper
student reaches 61/78 Healthy argmax matches on new-source validation crop views,
but none of the three standalone students passes original disease retention.
These are crop results, not photo-level app accuracy.

Six further Healthy verifiers compare linear and 128-unit heads with negative
costs 1, 4 and 12 on frozen original/adapted features plus classifier/CNN logits.
They optimize training labels only. The validation AUC selection in this initial
verifier experiment includes PlantDoc, TLID/Pakistan, classification and binary
sources; the string `tomato_village` does not match the cache's `tomatovillage`
source, so the latter did not contribute to its initial AUC selection. It remains
fully represented in the training data and subsequent app regression gates.
Record this limitation rather than claiming balanced validation over every old
source. The original strict gate largely removes the resulting Healthy gain.

A separately recorded validation-only noninferiority analysis therefore evaluates
an explicit tradeoff: disease recall may decrease by at most 1 percentage point
per photo cohort AND per named condition, while additional false Healthy,
unsupported false known and supported false Other rates may increase by at most
0.5 points. Integer allowances round down, so small cohorts/classes receive no
exception. Healthy correct counts may not decrease from installed 2.0.2. The
original zero-change gate is still reported separately. This is a protocol
revision after observing development data, not a retroactive claim that rejected
models passed the original gate.

Class-conditional policy variants restrict Healthy recovery to original Healthy,
Other, Early blight and/or Leaf Miner winners, the observed major confusions.
An actual nine-class proposal and independent CNN confirmation are required.
Unknown proposals are disabled in this revision so neutral binary evidence does
not create a new rejection. The source code enumerates the entire grid; no
image identifiers or ground-truth labels enter inference.

An additional 338-photo SLIF-Tomato v1 cohort was sealed before this analysis and
has not entered training or calibration. Manifest SHA-256:
`0932f20244b5b5ce541bce6d7c5992a9769f4fbff98420b08d4d4a8a85897fdc`.
It contains 44 Healthy, 120 Early blight, 13 Late blight and 161 unsupported
condition photos. These are published augmented views: one fixed view is selected
per inferred capture family, not independently verified plant identities.
The Phase I file listing is complete; the overall Kaggle listing stopped after
16,000 entries and is not a complete inventory of the 6.6 GB dataset. HTTPS size
checks and local SHA-256 hashes are available; Kaggle supplies no publisher file
SHA-256 here. Its predictions must remain unopened until a candidate is frozen.

Phase-two entry points are `phase2_prepare.py`, `phase2_multiclass.py`,
`phase2_healthy.py`, `phase2_policy.py`, `phase2_select.py`, `phase2_export.py`,
`phase2_verify.py` and `phase2_slif.py`. Their artifacts live under the campaign's
`phase2/` directory. No candidate should overwrite the installed bundle until
actual exported bytes, the frozen independent cohort and the phone are checked.

### Frozen phase-two outcome: rejected

The frozen class-conditional candidate is
`56abc8bdda6c903ecba1a84e8096bf7e75c2ed29975071888d6abc9068e16097`
(candidate JSON hash). Its 100,009,240-byte TFLite graph hashes to
`75a222ad5ae7c71219111d82255578a5b538344d7a5c36924ef8cb86ed9a680b`.
It passed 67 export parity cases and 6,738 actual-byte crop regressions under the
revised noninferiority gate, but did not pass the original strict gate.

On grouped classification validation, Healthy improved from 10/39 to 20/39 and
supported disease decisions from 129/375 to 158/375. Older public Healthy rose
from 67/80 to 70/80 with 521/558 disease decisions preserved. Prior test disease
decisions decreased from 558/585 to 557/585. These are development/regression
tradeoffs, not evidence of external generalization.

The first SLIF opening rejected the candidate: Healthy improved from 30/44 to
31/44 and supported disease decisions remained 28/133. However, diseased photos
called Healthy increased from 0 to 1, and unsupported false known calls increased
from 16/161 to 17/161. Both exceed the predeclared 0.5-point allowances after
rounding down. The model was not installed. There must be no tuning on SLIF while
continuing to call it independent acceptance.

### Rejected phase-three detector experiment: incompatible annotation targets

Inspection found a second failure mechanism. In a failed field photo the old
detector selects a broad plant region while publisher annotations identify many
smaller leaves. There are no detections in 30/133 supported-disease SLIF photos.
Whole-image classification is also weak (only 13/120 Early blight argmax matches
with the installed baseline), so detection alone is not established as the
complete cause. The source contains wide field scenes and mixed leaves, unlike
single-leaf benchmark photos. Image-level disease labels are not reliable labels
for every leaf crop in such a scene.

The original trainable detector checkpoint was recovered at
`~/sakshyam-agribot/v2/det_runs/y9t_416_comb/weights/best.pt`. On all 13 goldens
it matches the shipped TFLite detector within 0.0078 input pixels for raw box
coordinates and 0.000020 for scores. This establishes a usable warm start.

SLIF is now explicitly detector development data. One published view is retained
per inferred family; close pHash matches are merged before splitting. The split
has 464 field training and 148 field validation photos, alongside 6,219 older
training and 744 older validation photos. Field training photos are oversampled
four times, yielding 8,075 training entries. Repeats remain within training.
The older 751-image detector test partition is not used in fitting or selection.
Sampled classification images exactly match the corresponding publisher detection
images by file SHA-256. All annotation IDs map to generic leaf for detection only;
no scene-level disease label is assigned to each annotated crop.

The initial comparison returned mAP50-95 0.61176 on old validation and 0.01238
on the SLIF regions, with field mAP50 0.01984. **This comparison is invalid as a
measure of whole-leaf detector generalization.** Overlay review and the source
paper's section 3.2.2 established that SLIF annotations describe diseased regions
and healthy leaf regions, rather than complete leaf extents. Mapping all these
boxes to generic leaf was an implementation mistake. It does not establish that
the source annotations are corrupt. The library's `metrics/precision` field also
duplicates mAP here and must not be described as ordinary precision.

The rejected trial started from that checkpoint at 320 and 640 pixels,
20 epochs each, AdamW learning rate 0.0001, seed 83, BF16 AMP, 0.2 mosaic
probability and four final epochs without augmentation. Checkpoints and optimizer
state persist; restarting uses the saved last checkpoint. Separate old/field
validation follows training. Export requires old mAP50-95 within 0.01 of the
shipped 320 baseline, field mAP50-95 gain of at least 0.02 and field mAP50 at
least 0.15. Passing detector metrics alone does not authorize app promotion:
actual-byte end-to-end replay, independent field evidence and phone tests remain.
The 320 trial completed 20 epochs; the 640 trial was terminated after seven.
Neither is a deployment candidate. `data_complete.json` now records
`annotation_role=disease_or_healthy_region_not_whole_leaf` and `quarantined=true`;
training, validation and export entry points reject this dataset. No further GPU
time should be spent resuming the invalid target mapping. See the
[source paper](https://doi.org/10.1007/s00521-026-12271-0) for annotation semantics.

### Additional adapted-model controls and fresh classification acceptance

The verified nine-class phase-two data is reused for three additional controlled
last-four-block LoRA trials: logit-adjusted CE, CE plus expected label reward,
and CE plus clipped PPO. These share the original initialization, source/group
sampling, reliable-teacher retention weight 3, seed 53, 12 epochs and batch 64.
Logit adjustment uses half the log training-view prior. Expected reward has
weight 0.1; PPO also has weight 0.1, clipping 0.2, an epoch-frozen old policy,
sample-wise reward baselines and an independent action RNG. False Healthy has
reward -8; correct Healthy +2. These are label-derived contextual bandits, not
human-feedback RL. Their gradient tests pass. Source-code options must not be
reported as completed experiments until corresponding result files exist.

A fresh TOM2024 classification cohort (DOI 10.17632/3d4yg89rtr.1, CC BY 4.0)
uses Category A English tomato files, excluding Category B augmentation and
French copies. Healthy and explicitly named Late blight are mapped directly;
generic virosis and mite remain Other. One file per inferred three-minute
filename burst per class is selected before inference, with source/campaign
duplicate checks. These bursts are not verified plant identities. Its labels
are for classification; no region annotations are remapped to leaf boundaries.
The cohort must remain unopened until the next candidate and protocol are frozen.

The installed 2.0.2 baseline averaged 398.6 ms per one-leaf analysis over five
warmed runs on the connected RMX5101 phone. The last measured detector and
classifier times were 65.4 ms and 333.1 ms. These are one fixture's timings, not
an FPS guarantee for scenes containing several leaves. The benchmark is now
included in instrumentation output so it does not depend on logcat availability.

## Findings that affect the repair

1. The original trainable checkpoint reproduces the shipped classifier. Its
   SHA-256 is `659bd21eceb7ac2ba78abf20b449cb3462f3fb8439778e26f32c6d348e88e35f`;
   the original classifier TFLite SHA-256 is
   `12b9b17194ac5ba779126274d1a475016478e1b07ac6c2170fe51c2ef816af39`.
   There is no need to replace the successful encoder with a random head.
2. The verified model input is RGB, padded ROI, centered square, canonical resize,
   division by 255, then ImageNet mean/std. The original checkpoint and TFLite
   agree on these tensors. A second division by 255 or applying normalization
   directly to 0–255 pixels would violate this contract. The reported 30% result
   has not been established as the accuracy of the verified original model.
3. Disabling CUDA matmul TF32 alone was insufficient for exact batched feature
   extraction. cuDNN's convolution TF32 changed patch embeddings: batch-96
   maximum logit error was approximately 0.0054. Disabling
   `torch.backends.cudnn.allow_tf32` reduced that error to approximately 0.00001.
   The runner and exact feature/CNN extraction paths now disable both forms.
   This precision difference is real, but does not explain a 70-point accuracy
   loss by itself. The first partial-backbone control preceded the cuDNN fix.
4. Training from cached 256-pixel integer crops and resizing to 224 is not exactly
   the same as the app's direct floating-point canonical ROI preparation. New
   feature and pixel caches use the exact app path. Detector-selected crops are
   included alongside annotation crops. This repairs an input mismatch without
   changing the Android normalization contract.
5. Two installed Android packages had the same launcher name, Agribot. The old
   `com.sakshyam.agribot.field` 1.0 pilot uses a different inference pipeline from
   the maintained `com.sakshyam.agribot` app. A successful DINO experiment cannot
   validate the pilot's different model. The pilot was disabled for user 0 using
   `adb shell pm disable-user --user 0 com.sakshyam.agribot.field`; its package and
   data remain installed. The maintained app was launched explicitly. Reversal:
   `adb shell pm enable com.sakshyam.agribot.field`. This removes launcher
   ambiguity; it does not prove which icon was used for previous failures.

## Data acquisition and quality controls

Publisher archive/file SHA-256 values are verified before use. Extraction checks
paths and selected file types. The audit checks source-file identity, exact
canonical input identity, dihedral perceptual similarity and contradictory
labels before fitting or acceptance membership is finalized.

| Source | Actual use and constraints |
| --- | --- |
| Existing Tomato-Village and PlantDoc campaign data | 12,948 crop records across the existing grouped partitions; fitting uses train only and selection uses validation only. Previously opened app sets are regressions. |
| [TLID v2](https://data.mendeley.com/datasets/kt64b2kh89/2) | 15,254 image files actually extracted from TLID, despite different advertised totals. PTLID patches are not used. Healthy and Miner map directly; unsupported pure conditions map to Other. Mixed Miner conditions are excluded from single-class fitting. |
| [Pakistan v2](https://data.mendeley.com/datasets/3mbnb82mxd/2) | Only 830 raw images are considered. The 7,200 preaugmented images are excluded. Republished overlaps, contradictory labels and duplicate inputs are removed. |
| [Bangladesh binary v1](https://data.mendeley.com/datasets/rnbsw72zb5/1) | 1,030 image files: two unannotated extras are excluded; 87 filename/annotation disagreements are quarantined. The 941 agreement photos comprise 802 author-training and 139 author-validation photos. Coarse diseased labels never become nine-class Other labels. |
| [Bangladesh classification v1](https://data.mendeley.com/datasets/jttrv2w27r/1) | Further evaluation-only cohort: a deterministic maximum 120 files per original category, 829 downloads, 821 eligible after overlap and inferred-group exclusions. No predictions enter selection of this cohort. |

The final first-source audit contains 25 contradictory pure-label clusters
(50 image records), including photos labelled both Healthy/Miner and Whitefly,
and Pakistan photos assigned incompatible disease labels. These are quarantined,
not relabelled automatically. Another 4,859 mixed-condition records are excluded
from single-class fitting. Mixed Miner plus another disease is not a clean Other
target because a supported disease is present.

After exclusions, the new pure-label development manifest contains 9,180 photos:
8,718 TLID and 462 Pakistan. A separate 1,589-photo source cohort was sealed before
new-source model predictions. Photo families stay together; inferred families
are not verified plant identities. External data may contain republished images,
so absence of detected overlap does not establish independence from every
historical source used to train the original checkpoint.

The binary protocol changed prospectively before any binary-source predictions:
the clean author-training partition became binary fitting/development data,
while all author-validation groups stayed excluded. Capture-family, source-byte
and perceptual overlap checks remove nearby fitting records. The clean 139-photo
author-validation subset was first opened after the second CNN policy freeze.
It becomes regression data for subsequent policy refinements. This revision is
recorded in `cnn_inputs/protocol.json`; the earlier audit's statement that the
entire binary source remained outside fitting is historical, not the final use.

## Experiments actually executed

The following counts describe separately fitted trials. Threshold combinations
are calibration trials and are not counted as additional trained models.

| Stage | Trials | Model change |
| --- | ---: | --- |
| Partial-backbone supervised control | 1 | Original DINO initialization; one head warmup epoch, then the last four transformer blocks; 20 epochs, strict disease-retention selection. |
| Original-data exact-feature heads | 29 | Linear and residual MLP heads over the frozen original DINO CLS features. |
| Expanded, detector-aligned feature heads | 29 | Same loss/architecture controls with audited new data and detector crops. |
| Preliminary factorized experts | 12 | Separate Healthy/Miner and known/Other corrections. |
| Detector-aligned factorized experts | 42 | Linear/MLP architectures, seven isolated loss families, three correction modes. |
| Dense-token factorized experts | 51 | CLS, patch mean, patch maximum and patch std; includes nine domain-balanced repeats. |
| GPU boosted-tree residual experts | 8 | XGBoost, original log odds as base margins; paired Healthy/Miner and knownness tasks, depth/cost controls. |
| Independent CNN experts | 6 | ImageNet-initialized MobileNetV3 Small; center versus whole-ROI letterbox views, each with BCE, focal or exact expected-reward Healthy loss. |

The 29-head controls compare plain CE, Healthy-weighted CE, class-weighted CE,
logit adjustment, focal loss, label smoothing, margin loss, reliable-teacher
retention, expected reward, reward plus CE, REINFORCE, PPO, DPO, mixup, and
supervised contrastive learning. Supervised contrastive learning uses the MLP
projection; it is not duplicated as a meaningless linear-only arm.

Reward costs distinguish correct Healthy (+2), another correct class (+1),
generic error (−1), rejecting a supported condition (−3), accepting an unsupported
condition (−4), and a false Healthy call (−8). These RL arms are one-step contextual
bandits with label-derived rewards. They are not human-feedback RL or an
interactive crop simulator. PPO uses clipping and a reference KL penalty; DPO
uses the labelled class versus the strongest incorrect reference class.

The first supervised control reached high aggregate validation accuracy but
failed Early Blight, Late Blight and Miner retention. Many replacement heads
improved Healthy while increasing disease-to-Healthy errors. Neither a strong
crop score nor a higher reward was accepted as sufficient evidence for the app.

CNNs have 1,519,906 parameters and two independent outputs: Healthy and supported
condition. The original DINO encoder and disease head remain unchanged. CNN
training uses full-ROI-preserving augmentation, source/target/group balanced
sampling, a head warmup, then backbone fine-tuning for 16 epochs. Binary diseased
targets are masked out of the knownness loss because they do not identify whether
the condition belongs to the app's eight supported classes.

Center BCE/focal Healthy AUCs were approximately 0.9786/0.9793, versus 0.9652 for
the reward-only Healthy loss. Whole-ROI letterboxing did not consistently improve
older app regressions. These are validation crop statistics, not field accuracy.

## Selection, failed candidates and acceptance accounting

The selection metric uses full repeated-static-photo plant replay, independently
summarized for original and new validation sources. The detector, frame admission,
quality weight, calibrated probabilities, multi-leaf tracking and verdict rules
matter. Oracle crop top-1 is insufficient. Static repeated frames remain one
photo; they are never counted as independent observations.

Do not interpret the existing evaluator's `all.correct_rate` as conventional
nine-class accuracy: its numerator counts recognized supported Healthy/disease
verdicts, while its denominator also includes unsupported photos. Correct Other
rejections are reported separately. On an unsupported-heavy source this
recognition yield can be near 30% even when it is not the classifier's top-1
accuracy. Compare the named scope metrics and their denominators.

The first guarded CNN candidate improved Healthy on the original app regressions
and on the sealed TLID/Pakistan cohort, but added one unsupported false-known
decision there. It failed the strict gate. The second policy limited binary
knownness recovery to an original Healthy-leading supported class. It improved
the still-unopened binary author-validation Healthy results from 14 to 22 of 59,
with zero false Healthy decisions among 80 diseased photos. It nevertheless
increased unsupported false known decisions by two among 807 TLID/Pakistan
conditions. These were named disease calls, not additional Healthy calls. The
candidate was rejected for promotion. Those opened cohorts now serve as regressions.

Strict single-CNN confidence gates avoided the regressions but removed most of
the desired Healthy gain. The next comparison therefore requires agreement
between a supervised DINO feature head and the independent CNN. Positive votes
use the less confident Healthy evidence; negative votes also require both experts
to agree. The original encoder/disease head remains the reference branch.
Calibration uses grouped validation only; previously opened cohorts are regression
gates. The further 821-photo cohort stays unopened until a final policy freeze.

A further isolated-adapter comparison trains copies of the final DINO transformer
block with BCE, focal, and expected reward plus CE. LoRA rank 8 changes 49,922
parameters including its binary correction; the final norm remains frozen. The alternative trains the full
copied final block. The original prefix, final block and disease head remain
frozen in a separate reference branch. Both branches can share the first eleven
blocks on Android, avoiding a second complete DINO invocation. These trials are
reported only after completion and remain subject to the same regression gates.

## Runtime and verification contract

All deployable center-view candidates preserve one NHWC float32 input
`[1,224,224,3]`, nine float32 logits, the existing labels and Other index 8.
The temperature remains 0.8814700841903687, Other bias remains 0.75, and the energy
gate remains off. The detector is unchanged. The app's original preprocessing
must not be rewritten to accommodate a training cache.

Exported graphs contain stock LiteRT operations only. Verification covers random
inputs, all 13 canonical goldens, real validation tensors near gating boundaries,
all older selected regression crops through the actual exported bytes, and full
detector-plus-classifier replay on external photos. Restoring the self-contained
PyTorch bundle must exactly reproduce the pre-export model. Goldens must be
regenerated from the selected exported bytes in all three repository mirrors.

Phone downloaded-photo fixtures use metadata-free PNGs with verified RGB byte
hashes, because JPEG ICC colour management can otherwise change pixels between
PIL and Android. Fixture source labels are preserved; model expectations are
derived independently from exported bytes. A label parity test does not establish
that a publisher's disease label is agronomically correct.

## Reproduction entry points

Use the owner's existing `~/.venvs/agribot-train/bin/python`. See
`model_training/v3/requirements-campaign.txt` for direct pinned packages. The RTX
5090 requires the CUDA 12.8 PyTorch wheel used here, not a CPU-only replacement.

| Script | Purpose |
| --- | --- |
| `campaign_download.py` | Verify and extract the three initial source archives from pinned publisher metadata. |
| `campaign_data.py` | Audit source mappings, overlaps, contradictory labels and initial grouped membership. |
| `campaign_features.py` | Exact canonical DINO CLS/dense-token caches with logits and row-identity hashes. |
| `campaign_detector_data.py` | Detector-aligned fitting crops and photo-level validation replay inputs. |
| `campaign_heads.py` | Independent supervised/logit/reward/RL head controls. |
| `campaign_factorized.py` | Isolated binary corrections and validation selection. |
| `campaign_classical.py` | GPU tree residual controls with original log odds as base margins. |
| `campaign_cnn_data.py` | Exact center/letterbox pixel caches and prospective binary partition revision. |
| `campaign_cnn_train.py` | Six independently trained CNN controls. |
| `campaign_cnn_fusion.py` | Single-CNN gating and validation-only confidence calibration. |
| `campaign_consensus.py` | DINO-head/CNN agreement ablation and confidence calibration. |
| `campaign_fresh_data.py` | Further publisher-verified evaluation-only cohort and overlap audit. |
| `campaign_cnn_export.py` | Shared original DINO encoder plus optional confirmation head and tiny CNN export. |
| `campaign_verify.py` | Actual-byte regression classification, reusing unchanged detector outputs. |
| `campaign_report.py` | Compact experiment inventory, paired group bootstrap and outcome evidence. |

Long jobs run in persistent tmux windows. SSH can reconnect through the existing
Tailscale alias without losing the jobs. The inference/export environment,
source hashes, candidate parameters, seeds and checkpoints must travel with the
result. Do not export another candidate over an existing accepted artifact.

## Source attribution

The three initial sources used for additional training are CC BY 4.0. TLID v2:
Marcelo Pellenz, Grasielli Barreto Zimmermann, Alceu de Souza Britto Jr., Yandre
M. G. Costa, DOI 10.17632/kt64b2kh89.2. Pakistan v2: Mohammad Naeem Malik, Syed
Hashir Ali, Hamid Ali, DOI 10.17632/3mbnb82mxd.2. Bangladesh binary v1: Mangsura
Kabir Oni and Tabia Tanzin Prama, DOI 10.17632/rnbsw72zb5.1. Processing includes
deduplication, conservative class mapping, ROI extraction, normalization and
model training. No source photos are redistributed in this repository.

The classification source, initially evaluation-only and subsequently admitted
to phase-two training after its first opening, credits Nure Alam Siddiki
Rachin, Nasim Mahmud Likhon and Md Abdullah Hel Baki, DOI 10.17632/jttrv2w27r.1,
also CC BY 4.0. Source labels and uncertain inferred capture identities remain
limitations. Existing source/weight attribution and public-release review
requirements continue to apply.

SLIF-Tomato v1 is attributed to Romiyal George and collaborators, dataset DOI
10.34740/kaggle/ds/8911633, Kaggle owner `romiyalgeorge`, CC BY 4.0. Related paper:
https://doi.org/10.1007/s00521-026-12271-0. It was first independent evaluation,
then used in an aborted detector experiment with incompatible target semantics.
Those detector checkpoints are quarantined. Only
metadata and derived metrics are committed; source photos remain on the GPU PC.
The [LibreYOLO training documentation](https://www.libreyolo.com/docs/cli/train)
describes the checkpoint-based detector training used here; version 1.6.0 and the
existing owner's checkpoint are pinned for this continuation.
