# Agribot public-image investigation — 15 October 2026

**Conclusion:** false “Problem not in our list” decisions are reproducible on
some field photos, but this is not uniform over-rejection across supported
conditions. The positive Other bias contributes to some failures; removing it
also produces more incorrect known-condition decisions on unsupported inputs.
Blanket rewards for avoiding Other would optimize the wrong objective.

No model weights, app thresholds, or production inference rules were changed.

## What was tested

The exact shipped `disease_classifier_0.tflite` and `leaf_detector.tflite`, verified
against the app manifest's SHA-256 hashes, were evaluated on 952 downloaded,
author-labeled images:

| Source | Downloaded | Headline analysis | Coverage |
|---|---:|---:|---|
| [Tomato-Village](https://github.com/mamta-joshi-gehlot/Tomato-Village) | 461 | 461 | Entire Variant-A test split; all eight supported classes |
| [PlantDoc](https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset) | 236 | 225 | Field photos; 11 exact TRAIN/TEST overlaps excluded |
| [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) | 255 | 255 | Seeded lab sample; 150 supported and 105 unsupported-condition images |

The main results reproduce the app's leaf detection, top-six crop classification,
calibration, evidence weighting, partial-disease override, confidence/margin
gates, and stopping rules. Each photo is repeated as an identical frame to
exercise consensus. This is **static replay, not a live-camera trial**; repeated
frames are not independent observations.

Whole-image classifier outputs and author-box crop outputs are separate
diagnostics. A missed detection stays “No leaves”; no whole-image fallback is
invented for the app replay.

## Supported conditions: what the app would say

These denominators include only images labeled with supported tomato conditions.
“False Other” means an UNKNOWN_CONDITION verdict, not merely a low score or an
Unsure verdict.

| Source | Supported images | Correct verdict | False Other | Unsure | No leaves |
|---|---:|---:|---:|---:|---:|
| Tomato-Village | 461 | 425 / 461 = 92.2% | 0 / 461 = 0.0% | 10 | 1 |
| PlantDoc | 27 | 17 / 27 = 63.0% | 4 / 27 = 14.8% | 5 | 0 |
| PlantVillage | 150 | 143 / 150 = 95.3% | 2 / 150 = 1.3% | 2 | 0 |

The remaining supported-image errors were wrong known labels: 25, 1, and 3,
respectively. A reduction in Unknown wording alone would not fix those errors.

The PlantDoc subset is small: the nominal 95% Wilson interval for 4/27 is
5.9–32.5%, before accounting for source/plant correlations. The two PlantVillage
rejection photos have the same recorded leaf identifier, `GHLB2 Leaf 155`
(`155.1` and `155.2`); they should not be treated as two independent biological
failures. Keeping one photo per detected repeated leaf identifier changes that
lab result to 1/149 supported photos.

The sources have different class mixtures. Restricting all three to Early
blight, Late blight, and Healthy and weighting those classes equally gives
app-replay recall of 81.2% for Tomato-Village, 61.8% for PlantDoc, and 95.3% for
PlantVillage. The source difference remains, but this comparison does not prove
whether lighting, framing, labels, or another factor causes it.

## The classifier and crop diagnostics

Healthy recognition is a distinct weakness in the field-style sources: the app
gave the correct Healthy verdict on 11/22 Tomato-Village healthy photos, 3/8
PlantDoc healthy photos, and 50/50 lab photos. In Tomato-Village, seven of the
healthy-labeled photos were reported as Leaf Miner and four were Unsure. The
high aggregate accuracy therefore hides a healthy-class problem. These small,
source-specific cohorts do not establish a deployment error rate.

On the 27 supported-condition PlantDoc photos, whole-image argmax classification
chose Other 11 times. The full app replay displayed Unknown four times. Testing
only whole-image logits would therefore overstate this app's Unknown rate.

Using author-provided leaf boxes still produced Other on **13 of 60 supported
leaf crops**. Healthy-labeled crops accounted for 11 of those rejections:
**11/27 healthy leaf crops**, drawn from eight photos. These are correlated crop
observations, not 27 independent plants. Leaf detection matched 53/60 supported
reference boxes at IoU ≥0.5; the selected top-six boxes covered 52/60.

Thus, missed detection is not the whole explanation. Recognition, representation,
crop preprocessing, and source-label quality deserve attention. The gallery
shows the actual rejected photos and detected boxes; its labels are dataset
annotations, not a new agronomist diagnosis.

## What happens if the Other bias is removed?

The shipped classifier adds +0.75 to Other after combining logits. Temperature
is approximately 0.88147; the optional energy gate is disabled.

Holding images, model weights, detections, and the remaining app rules fixed:

| PlantDoc outcome | Shipped bias +0.75 | Bias 0 |
|---|---:|---:|
| Correct supported-condition verdicts | 17/27 | 20/27 |
| False Other on supported conditions | 4/27 | 2/27 |
| Unsupported inputs given known labels | 20/198 | 26/198 |
| Unsupported inputs explicitly rejected as Other | 166/198 | 157/198 |

Of the four original false Unknown verdicts, removing the bias made one correct,
one Unsure, and left two Unknown. Other gains came from previously undecided
cases. This is a genuine trade-off, not a free fix.

The unsupported PlantDoc group includes 39 tomato-condition photos and 159
non-tomato photos. At the shipped setting, known labels were incorrectly
assigned to 7/39 unsupported-tomato photos and 13/159 non-tomato photos. At zero
bias these counts became 8/39 and 18/159. All 105 unsupported-condition lab
photos were rejected at both settings.

Across this deliberately constructed sample, 271 of the 277 Unknown verdicts
were on out-of-scope source labels, and six were on supported labels. This mix
is chosen by the evaluation and **is not an estimate of the phone's rejection
frequency**.

## Robustness and software checks

- All 105 Android JVM tests passed across domain, camera, ML, data, scan, and app modules.
- All four on-device model golden tests passed using static images, without the camera.
- All 53 Python regression tests passed, including the optional downloaded-image check; no skips.
- All 13 golden-image controls passed. Sparse and original dense preprocessing
  matched every classifier/detector input pixel on these controls; logits matched
  the saved references within about 0.000002.
- Every one of the 952 saved app replays matched the actual compiled Kotlin
  decision classes, both at the shipped bias and at zero bias. Maximum
  confidence difference was below 0.00000004.
- All 952 downloads decoded, all source-file hashes reconciled, and no inference errors occurred.
- On 24 selected supported images, baseline whole-image accuracy was 23/24.
  Blur, 50% brightness, and 90° rotation each gave 22/24 correct, with no Other
  predictions. These 72 transformations are correlated probes, not new images.
- Uniform black, white, and gray controls produced no leaves. Random noise
  produced one false leaf detection, which the classifier rejected as Other.

## Recommendation and limits

Prioritize review of the failed field examples, particularly healthy-labeled
leaves. If their labels are confirmed, use targeted supervised fine-tuning with
representative field images, then calibrate on a separate validation set while
retaining unsupported-condition and non-tomato guardrails. Do not train merely
to minimize the frequency of the Other label. The bias sweep is exploratory and
does not justify deploying a new threshold from these test images.

This is a diagnostic replay of public sources used during model development,
not an independently held-out field-accuracy benchmark. The original full
training cache and the clean-test membership list are absent. Exact source
TRAIN/TEST overlaps were removed, but additional training/near-duplicate overlap
cannot be ruled out. Ten downloaded photos match the bundled golden controls.
Within-sample perceptual-hash grouping removes 22 additional correlated
Tomato-Village images in a sensitivity check; its false-Other count remains zero.
Source labels have not been newly adjudicated by an agronomist.

The test does not reproduce camera conversion, viewpoint changes, movement,
thermal behavior, or field prevalence. It cannot establish the cause or rate of
the user's phone-specific cases.

## Inspect and reproduce

- `agribot_model_diagnostics.html`: readable executed-notebook export with charts.
- `agribot_model_diagnostics.ipynb`: executed, reproducible analysis notebook.
- `image_results.csv`: all 952 image results, source URLs, and analysis-eligibility flags.
- `predictions.jsonl`: raw model logits, boxes, probabilities, crop results, and decisions.
- `bias_sensitivity.csv`, `scope_metrics.csv`, and `oracle_object_results.csv`: detailed diagnostics.
- `analysis_summary.json`, `validation.json`, and test logs: calculations and validation evidence.
- `source/`: pinned repository metadata and original source notices.
- `tests/model_evaluation/README.md` in the repository: commands and methodology.
