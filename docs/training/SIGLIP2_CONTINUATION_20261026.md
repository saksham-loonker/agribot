# Apache-only recognition repair continuation

**Latest checkpoint:** [bounded recovery and safety audit](CONTEXT_GUARD_CONTINUATION_20261026.md) takes precedence over pending-control statements below. There are 318 completed distinct research controls. The stricter Apache-only image-context candidate passes fresh development/internal replay (38/53 Healthy); independent acceptance, exported-byte and native/device verification remain outstanding. The phone is disconnected and its model is unchanged.

Revision: 26 October 2026. This record supersedes deployment suggestions in the
MobileCLIP continuation. Recognition remains unresolved. The installed interim
2.0.2 classifier is unchanged, and no replacement model has been installed.

## Why the experiment changed

The full Apple Machine Learning Research Model license explicitly excludes
product development and commercial products. The 22 completed MobileCLIP-based
research controls, including the five-expert teacher, new-data fine-tunes,
distillation and union, are ineligible for this app. Do not export, deploy or
distill them into its replacement. No Apple model weights were added to Git or
the Android assets. Preserve their historical results as research evidence;
do not describe them as a production candidate. The original agreement and
attribution are retained in `licenses/APPLE_AMLR_RESEARCH_ONLY.txt`.

The last restricted union reached 98/129 Healthy field and 114/119 paper Healthy
on the old eight development cohorts while preserving the disease checks.
Correctly evaluating **the baseline app plus recovery** on 180 newly admitted
internal validation photos gave **31/53 Healthy**, below 70%. Earlier recovery
head-only recall was a diagnostic, not app recall. Do not reuse the consumed
246-photo benchmark or its related families as independent evidence after
admitting their human labels into training.

## Independent replacement and exact input contract

The replacement encoder is
[`timm/vit_base_patch16_siglip_256.v2_webli`](https://huggingface.co/timm/vit_base_patch16_siglip_256.v2_webli),
an Apache-2.0 port of
[`google/siglip2-base-patch16-256`](https://huggingface.co/google/siglip2-base-patch16-256).
Pinned repository revision: `ac4cc431026674dcac29e34a00d3bd2048390eb3`.
Source `model.safetensors` SHA-256:
`13ab9d238865f89b14005c5e92a099add04cbfab331a64245acab813509cb2b0`.
It has 92,930,304 parameters and produces a 768-dimensional image embedding.
Keep its Apache license and source attribution with any eventual model export.

The app's existing canonical tensor is RGB, NHWC, 224 pixels, divided by 255
and normalized with ImageNet mean/std. `permissive_encoder.py` explicitly
reverses that normalization, resizes RGB [0,1] to 256 using bilinear sampling,
then applies native mean `[.5,.5,.5]` and std `[.5,.5,.5]`. SigLIP2 therefore
receives RGB [-1,1]. Embeddings are L2-normalized and scaled by sqrt(768).
This is a recorded adaptation, different from the publisher's bicubic/crop
recipe. Training and inference use the same adaptation. Do not substitute a
different normalization at export, or feed ImageNet-normalized pixels directly
into the encoder. Seven solid RGB probes compare the wrapper with native
pretrained inference; a separately restored model checks 54 actual images
before any feature cache is accepted. Both checks passed.

The original nine-class DINO disease classifier never enters an optimizer.
Its class order, probability combination, temperature, Other bias and original
disease priority remain fixed. No 0–255 normalization defect has been established
in that classifier. New work concerns a separate image Healthy verifier.

## Data and contamination boundaries

GPU workspace: `~/agribot-data/recognition_campaign_20261024/phase13_siglip`.
Code: `~/agribot/model_training/v3`; environment:
`~/.venvs/agribot-train/bin/python`; SSH alias: `agribot-5090` over Tailscale.
Long jobs run under tmux session `agribot` so a laptop network change does not
kill training. Inspect windows and log files before relaunching anything.

The 23,259 rows are the same **human-only** phase-twelve image corpus, SHA-256
`aa429d09e6800af20e27b0ee1158179984b2bb6795780451ee61e0268055cfb9`.
Pixel storage reuses the original scene shard and the separately admitted
787-photo shard; do not duplicate the 13.5 GB cache. The consumed 246
representatives and all 787 related photos across 235 families are retired as
benchmarks. Their new grouped internal validation has 180 photos, including
53 Healthy, 72 coarse diseased TOM and 55 paper diseased. It is **not independent
acceptance**. Train and validation capture groups are disjoint; the 200 source
quarantines remain excluded. Coarse disease labels are binary negatives and
must not be fabricated into named conditions or nine-class Other targets.

No MobileCLIP features, teacher logits, checkpoints or distilled parameters are
used by SigLIP2. Reusing human labels and the unchanged original LiteRT baseline
records is allowed. Those records in `phase12_scene/union_app_validation/` are
original DINO/MobileNet/YOLO predictions, not the restricted union's outputs.

The independent Abu Dhabi source remains sealed at
`phase9_ad_source/sealed_geometric_manifest.json`, SHA-256
`5cf37861dfe835c918643b2753a05c229e9327cd750524435006384bdcfae6b6`.
Do not run diagnostic predictions on it until a licensed, exact candidate passes
the development and internal app gates and its independent acceptance protocol
is frozen. No cold-source threshold fitting or repeated acceptance attempts.

## Completed controls and current queue

Four frozen encoder heads completed: linear/MLP with BCE or focal loss. The best
training-validation head is MLP focal, field ROC AUC about 0.9925. Frozen heads
alone fail full app retention. High crop or image ROC AUC is insufficient.

Three monotone logit fusions completed with negative costs 1, 2 and 4. Inputs
are the **pre-Apple** DINO adapted linear head, original/adapted DINO MLP,
copied DINO12 Healthy scene model, genus CNN and the new SigLIP2 verifier.
Only the first four original human-data inputs and the new licensed fifth input
are read. Old MobileCLIP five-input arrays must never be substituted.
Fusion fits use the old human corpus's grouped calibration subset; the new
180-photo app cohort is evaluated separately. Calibration fitting and threshold
selection remain development, not independent evidence.

The coarse threshold grid's safe choice recovered 89/129 field Healthy.
A recorded development refinement selected a different fusion at threshold
0.98: **94/129 field Healthy and 112/119 paper Healthy**, with all eight
development retention checks passing. This still needs a fresh self-contained
graph replay and the 180-photo internal app check. It is not qualified.

Separate supervised SigLIP2 controls use BCE and focal loss, 16 epochs, batch
32, backbone LR 2e-5, head LR 1e-4, AdamW decay .01, negative cost 4, seed 557,
photometric jitter, blur and low-resolution/horizontal-flip augmentation.
Canonical training storage is FP16 with BF16 compute; validation is FP32.
Best checkpoints, optimizer and RNG state are saved atomically. Final Platt
calibration is applied after loading state; the entire restored validation
must match before a completed run is counted. Restricted teacher distillation
is explicitly rejected for this family. The focal control is running; the
controller queues BCE and both app validations afterward.

There were 292 valid completed research controls before SigLIP2; eight invalid
unloaded-encoder fits were excluded. The four new heads and three fusions bring
the current completed research count to 299 before supervised controls finish.
Counts measure work performed, not deployment eligibility or confidence.

## Reproduction and gates

From `~/agribot/model_training/v3`, with `P` set to the campaign directory:

```bash
PY=~/.venvs/agribot-train/bin/python
P=~/agribot-data/recognition_campaign_20261024
$PY phase10_feature_train.py --root "$P/phase13_siglip" --feature-family siglip2 --task-scope primary_image --primary-source tom2024
$PY phase11_logit_fusion.py --campaign "$P" --model-root "$P/phase13_siglip" --head-family siglip2
$PY phase11_scene_select.py --campaign "$P" --model-root "$P/phase13_siglip" --head-family siglip2 --fusion-run logit_fusion --out-name fusion_refined_selection --healthy-thresholds 0.951,0.955,0.96,0.965,0.968,0.975,0.98,0.985
$PY phase12_scene_sft.py --root "$P/phase13_siglip" --family siglip2 --loss focal --batch 32
$PY phase13_validate.py --campaign "$P" --selection fusion_refined_selection
$PY phase13_report.py --campaign "$P" --out "$P/phase13_siglip/evidence.json"
```

Do not rerun completed tmux jobs concurrently. The current remote
`verification_pipeline.py` waits for focal restore parity, performs app replay,
fresh graph/internal checks, then the BCE control and its corresponding checks.
It stops on errors. Inspect `verification_pipeline.log`, numbered
`verification_stage*.log`, `siglip2-sft_stage0.log` and candidate result files.

`phase13_context_model.py` builds a self-contained permitted fusion graph from
pinned source checkpoints. `phase13_validate.py` restores that graph, replays all
5,820 unique development images / 6,468 cohort observations, and requires exact
selected metric agreement. It separately scores **baseline plus recovery** on
the 180 internal photos. The retained original disease verdict takes priority.
The unchanged `phase2_select.retain` gates apply per cohort and named disease:
Healthy cannot fall, known disease loss is limited to 1 percentage point, and
additional false Healthy/Other/known to 0.5 points, using integer floors and
the correct diseased denominator. Small cohorts allow no additional mistakes.
Both field sources and the new internal Healthy subset must reach 70% before
independent acceptance is considered.

If internal performance fails, diagnose errors within development. Do not open
AD to obtain a more favorable number, remove hard examples post hoc, loosen
disease gates silently, or mix restricted teacher outputs into a new model.

## Android and delivery work still required

Whole-image Healthy evidence describes the visible primary view; it is not
clinical assurance that every leaf or the entire plant is healthy. The owner
has not yet answered the optional scope preference. Keep disease priority and
explain visible-view scope in the eventual UI.

The existing schema-four source support carries proposal and frozen reference
leaf distributions on real tracks/frames. It is disabled in shipped schema-two
assets. It does **not** implement the new whole-image context model. A qualified
context model needs a distinct image-evidence path and manifest contract, actual
quality/frame bookkeeping, original disease priority, tested abstention and
camera/rows behavior. Never fake leaf boxes, tracks or frame observations.

After independent acceptance, export the exact permitted model, compare actual
LiteRT bytes against PyTorch and all regression gates, then validate native
probabilities, decisions, timing, navigation and downloaded-photo fixtures on
the phone. Do not ship PyTorch success as native parity. Source attribution and
Apache notices must accompany new weights. Reconsider the fused graph's size
and latency only with an explicitly validated equivalent or licensed student.

The latest APKs already built with unchanged interim assets are in local
`dist/agribot-2.0.2-reference-priority-20261025/`; their build evidence says no
new model. The debug APK uses the existing development signing certificate.
No phone is visible to ADB at this checkpoint. Do not clear app data/uninstall
to install a future build, and do not ask for USB reconnection until a qualified
model actually needs device validation.

Local Python verification: 122 tests, 121 passed, one skipped. Android native
CI for `c955fba9664753663ec016bebbaa8f210a128bf1` passed, run `40219608964`.
Prior native unit count is 105; debug and unsigned release assemblies passed.
These checks concern code/build correctness and do not qualify model accuracy.
PR [#6](https://github.com/saksham-loonker/agribot/pull/6) stays a draft until the
recognition and native acceptance work is complete.
