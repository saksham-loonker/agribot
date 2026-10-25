# Leaf-scoped Healthy recovery: continuation and runtime contract

Revision: 25 October 2026. This supplements the chronological campaign record
and `RETRAINING_HANDOFF.md`. Research remains incomplete. The installed comparator
is 2.0.2; no candidate described here has been approved for installation.

## Why the training target changed

The original disease classifier is useful and stays frozen. Earlier supervised,
logit and reward controls did not establish a deployable improvement. A scene
classifier reaching a good validation score is insufficient: the frozen
prospective scene prototype recovered only 44/95 field Healthy photos, below its
70% requirement. That reservation is now consumed regression, never fresh
acceptance again.

The FieldPlant annotation audit found a concrete supervision mismatch: of 141
photos containing an explicitly Healthy leaf, 128 also contain unhealthy leaves.
A Healthy annotation for one leaf does not label every other detected leaf or
the entire plant Healthy. A contact sheet also shows questionable species among
negative symptom annotations. Negative genus targets are therefore masked.
These observations do not constitute independent clinical relabeling.

The new control uses explicit leaf boxes and masks all 2,858 weak TOM2024 fitting
rows. It reuses the 60,865-row pre-teacher corpus, excluding the 2,400 later
teacher-derived Healthy views. Regression images and their labels remain intact.
No failing example is removed to make a gate pass.

## Data, membership and lineage

Source: [FieldPlant publisher](https://universe.roboflow.com/plant-disease-detection/fieldplant),
paper DOI `10.1109/ACCESS.2023.3263042`, CC BY 4.0. A public Kaggle mirror,
`bloox2/fieldplant` version 1, carries an embedded README identifying that original
source and license. The original export records auto-orientation and no image
augmentation. Raw images stay on the GPU PC and are not redistributed in Git.

The selective download contains 1,403 tomato photos and 2,786 leaf annotations:
279 Healthy, 404 broad blight, 1,967 brown spots, 115 yellow virus, 19 mosaic and
2 bacterial wilt. Broad blight/spot labels are binary not-Healthy supervision;
they are not invented Early/Late Blight labels or nine-class Other labels.

Each selected ZIP member passed CRC verification. Only the 1,275,089,441-byte
compressed tomato subset was downloaded from the 5,912,207,630-byte ZIP; a full
archive SHA-256 is unavailable. Catalog SHA-256:
`3c96c8d0ca2c4dca51632df27f64cec4a7a85e00624211f9897d1ebeb61aeaec`.
Annotation CSV SHA-256:
`1b069168e3a982d98020fdf0e121305c84dc84880ebce543077a67d2cd83b433`.

Timestamp bursts, adjacent filename blocks and perceptual matches form 280
conservative families, including 59 Healthy-bearing families. These are inferred
capture families, not verified plant IDs. Whole-family splitting gives training
219 Healthy/2,009 negative leaf instances and validation 60 Healthy/498 negative
leaf instances. No prior-overlap family was found by the recorded audit; that
does not prove absence of every possible overlapping crop.

Leaf manifest SHA-256:
`8c1ad92eb488b444c576b60c8ef101d2164b15586f5d5495c766deb1a852fbd5`.
The combined canonical cache contains 66,437 rows, including 5,572 new center/full
ROI views. Row SHA-256:
`5019534db15e84ccef7ceed0bc9efb40f17ab2a849219559deb6c0e3db4dc10c`.
The two views of a leaf are not two independent validation instances.

The separate Abu Dhabi source remains unopened for diagnostic prediction or
fitting. Its sealed geometric manifest is
`5cf37861dfe835c918643b2753a05c229e9327cd750524435006384bdcfae6b6`.
Preparation compares hashes/fingerprints against this cold source to quarantine
overlap, without running a diagnosis. Preserve it until a complete policy and
acceptance protocol are frozen.

## Controls and current results

Six initial completed controls bring the previous 262 to 268: a leaf Healthy CNN,
an adapted genus CNN, and linear/MLP binary heads with BCE/focal losses over the
original DINO features. Policy grids do not count as trained controls.

The original-feature extraction reproduced 60,962 original-reference cases within
0.00002 logits. The best frozen MLP's leaf validation ROC AUC is about 0.962,
but its app replay still fails. This illustrates why AUC alone cannot authorize
deployment. The adapted genus gate improves its own validation; removing it also
fails the app target.

| Best retained policy family | TOM2024 Healthy, opened development | Paper Healthy, opened development |
|---|---:|---:|
| Leaf CNN plus adapted genus | 18/129 | 108/119 |
| Original-feature linear BCE | 21/129 | 96/119 |
| Original-feature linear focal | 14/129 | 91/119 |
| Original-feature MLP BCE | 14/129 | 97/119 |
| Original-feature MLP focal | 14/129 | 104/119 |
| Twelve copied DINO blocks, leaf focal | 14/129 | 86/119 |

The copied-tail focal control has now completed, bringing the total to 269.
Its best retained policy still fails field recall. Some best-ranked research
policies omit full-ROI confirmation; retention alone does not qualify them for
the schema-four deployment contract.

The distinct FieldPlant center-view oracle for interim 2.0.2 is 6/60 Healthy,
with 41 false Healthy calls among 498 negative leaves. This is an opened
annotation-box diagnostic, not detector-plus-plant accuracy or fresh acceptance.
The trained new verifiers perform better in isolation, but strict fusion retains
most legacy decisions and does not achieve the required end-to-end recall.

Two subsequent controls adapt a separate copy of all twelve DINO blocks using
proper leaf targets and focal loss. One uses horizontal flips; the other adds
training-only color jitter, Gaussian blur and downsampling to 112/160 pixels
before restoring 224. Neither updates the original nine-class branch. Source
Healthy misses inspected in an opened nine-photo contact sheet include original
images about 192–268 pixels across, blur and partial occlusion. This supports a
degradation control; it does not justify changing their labels.

Both copied-tail runs use 16 epochs, batch 64, seed 251, backbone LR 1e-5,
head LR 1e-4, weight decay 0.05, dropout 0.1 and negative cost 4. The ordinary
control caches FP16 training tokens and FP32 validation tokens. The augmented
control streams canonical images through a frozen patch embedding; validation
keeps exact FP32 canonical tensors. Optimizer and RNG states permit resumption.
Their completion and replay status are recorded in the machine evidence;
pending controls must not be counted as completed.

The baseline verdict audit finds TOM2024 Healthy outcomes of 83 Unknown,
23 no-leaf detections, 14 Healthy and 9 Unsure, with no named disease verdicts.
Thus reference aggregate disease priority does not impose a sub-70% ceiling on
this source. The 23 no-leaf photos are a separate detector-coverage limitation.
The four original false disease calls on paper Healthy remain protected by that
contract. This audit uses already opened development data only.

## App change: preserve the original aggregate disease verdict

The opt-in schema-four classifier contract emits 18 float32 values: proposal
logits for the existing nine labels followed by the complete frozen reference's
nine logits. It retains the existing NHWC float32 input, normalization, label
order, temperature, Other bias and detector. It requires full-ROI confirmation
and `classifier.reference_disease_priority: true`. Schemas two and three keep
their existing behavior.

The app accumulates proposal and reference evidence on the same actual tracks,
frames, detector scores and quality weights. If that reference accumulation
reports a named disease, it takes priority over a Healthy recovery proposal.
This closes an aggregation loophole: protecting individual disease leaves alone
does not prevent newly Healthy votes on other leaves from outvoting them.
It is a guarantee about the same observed frames, not unseen future leaves.

`AnalyzedLeaf`, `LeafObservation` and `VisionModelInfo` carry the optional reference
data. Both Check Plant and Walk Rows pass it through. Host replay implements the
same rule. Missing reference output fails explicitly when this contract is
enabled. No fake boxes, leaves or frame counts are added. Existing asset manifests
do not enable this feature, so the installed model is unchanged.

Four domain tests cover aggregate disease priority, Healthy recovery from Other,
missing reference data, and changing observations/quality over 180 frames. Native
validation passed 105 unit tests across domain, ML, scan and app, release lint,
and test-APK assembly. The Python suite ran 121 tests: 120 passed, one skipped.
The phone was absent from ADB at the latest check, so this continuation makes no
new device-validation or installation claim.

## Reproduction and next decisions

SSH alias: `agribot-5090`. Scripts: `/home/genesis/agribot/model_training/v3`.
Python: `/home/genesis/.venvs/agribot-train/bin/python`.
Artifact base: `/home/genesis/agribot-data/recognition_campaign_20261024`.

Run these commands from the script directory in persistent tmux. Preparation
requires the existing phase-five artifacts and complete recorded audits.

```bash
PY=/home/genesis/.venvs/agribot-train/bin/python
BASE=/home/genesis/agribot-data/recognition_campaign_20261024
$PY phase10_fieldplant.py --campaign "$BASE" --download
$PY phase10_fieldplant_audit.py --campaign "$BASE"
$PY phase10_leaf_prepare.py --campaign "$BASE"
$PY phase7_species_train.py --root "$BASE/phase10_leaf" --scene-work "$BASE/phase10_leaf/cnn_inputs" --task-scope annotated_leaf --primary-source fieldplant --run-name leaf_healthy --warmstart "$BASE/phase10_leaf/safe_cnn_warm.pt" --epochs 16 --negative-cost 4 --focal-gamma 2
$PY phase7_species_train.py --root "$BASE/phase10_leaf" --run-name leaf_species --warmstart "$BASE/phase6_roi/species/best.pt" --positive-source fieldplant --epochs 16 --negative-cost 2 --focal-gamma 2 --learning-rate 0.00005
$PY phase10_feature_train.py --root "$BASE/phase10_leaf"
$PY phase10_leaf_select.py --root "$BASE/phase10_leaf" --species-directory "$BASE/phase10_leaf/leaf_species" --out-name leaf_selection_joint
for HEAD in linear_bce linear_focal mlp_bce mlp_focal; do
  $PY phase10_leaf_select.py --root "$BASE/phase10_leaf" --head-directory "$BASE/phase10_leaf/leaf_feature_heads/$HEAD" --out-name "feature_selection_$HEAD"
done
$PY phase10_leaf_pipeline.py --root "$BASE/phase10_leaf"
$PY phase10_report.py --root "$BASE/phase10_leaf" --out "$BASE/phase10_evidence.json"
```

Read the audit/preparation argument parsers before regenerating membership;
never replace their manifests after inspecting candidate failures. The pipeline
resumes both copied-tail controls, runs app replay, and writes evidence. It does
not open cold acceptance data, export a model, install an APK or merge a PR.
Logs are `phase10_leaf/leaf_final_controls.log`, `leaf_sft_*.log` and
`feature_selection_*.log`; checkpoints and candidate JSONs live in their named
subdirectories. `phase10_report.py` records hashes, calibration, policy and scoped
results. Audit flags saying `training_used: false` describe the pre-fitting audit
snapshot; the report explicitly records subsequent binary training use.

Any qualifying candidate must pass unchanged per-source and per-disease
retention, at least 70% Healthy recall on each development field source, and
then a new frozen, family-aware acceptance evaluation. Export restoration,
actual LiteRT-byte replay and Android device parity are separate required gates.
Schema-four deployment requires the 18-output graph and native runtime together;
a nine-output asset must not be mislabeled schema four. Keep PR #6 a draft while
recognition remains unresolved. Do not substitute looser gates or a success
statement for evidence.
