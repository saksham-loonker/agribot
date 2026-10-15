# Public-image model diagnostic

This harness evaluates the exact Android TFLite assets against author-labeled
public images. It does not train models or modify production thresholds.

The retraining agent handoff is in
[`docs/training/RETRAINING_HANDOFF.md`](../../docs/training/RETRAINING_HANDOFF.md).
The original findings are checked in beside it; generated datasets, notebooks,
and raw predictions under `dist/` are not included in a clone.

These scripts preserve the original Windows investigation. The Kotlin replay
driver currently expects the originating local JDK/Gradle cache layout, output
directories use the original date, and full validation requires device results.
Follow the handoff's portability and candidate-evaluation requirements before
using this harness on a Linux GPU host or with newly trained models. The downloader
records resolved revisions but initially uses branch names; explicitly use the
handoff's pinned revisions when reproducing the original sample in a fresh clone.

Activate the repository environment and install `requirements.txt` from this
directory if the notebook dependencies are missing. From the repository root:

For a full replication, run the Android regression commands below first; they
create the compiled Kotlin classes used by the replay verification.

```powershell
python tests/model_evaluation/download_public.py
python tests/model_evaluation/evaluate_public.py --workers 4 --threads 2
python tests/model_evaluation/evaluate_public.py --stress
python tests/model_evaluation/run_controls.py
python tests/model_evaluation/verify_kotlin_replay.py
python tests/model_evaluation/analyze_public.py
python tests/model_evaluation/verify_results.py
python -m ipykernel install --prefix .venv --name agribot-evaluation
python tests/model_evaluation/build_notebook.py
```

Outputs, cached images, source notices, exact download URLs, raw logits, CSVs,
figures, and the executed notebook are local under
`dist/model-evaluation-20261015/`. Network access is needed for the first
download; subsequent steps use cached local files. Inference resumes from
`predictions.jsonl`; move it aside before an intentional fresh inference run.
Keep the same source revisions, model manifest, and evaluator when resuming.

Full regression verification also requires the six Android JVM suites and the
four bundled on-device golden tests. Run these before `verify_results.py`:

```powershell
. .\Activate-Dev.ps1
.\agribot_android_app\android\gradlew.bat -p agribot_android_app/android :domain:test :ml:testDebugUnitTest :camera:testDebugUnitTest :data:testDebugUnitTest :feature-scan:testDebugUnitTest :app:testDebugUnitTest --no-daemon --max-workers=1
.\agribot_android_app\android\gradlew.bat -p agribot_android_app/android :app:connectedDebugAndroidTest '-Pandroid.testInstrumentationRunnerArguments.class=com.sakshyam.agribot.AgribotModelGoldenTest' --no-daemon --max-workers=1
```

The device tests require an ADB-authorized phone but do not use its camera.
`verify_kotlin_replay.py` compiles a small Java driver against the actual Kotlin
decision classes from those JVM builds. It checks every saved replay at both
the shipped Other bias and zero bias. Its TSVs and results remain in `dist/`.
The sparse preprocessing optimization uses the original canonical weights;
golden checks compare full input tensors against the dense NumPy reference.

The sample comprises the entire Tomato-Village Variant-A test split (461),
PlantDoc TEST photos with annotations (236), and a deterministic tomato-only
PlantVillage challenge sample (255). Whole-image predictions, detected-leaf
predictions, author-box oracle crops, and app consensus are distinct analyses.

The app replay uses stable tracks on repeated **identical** images and the
source's maximum of six leaves, minimum evidence, capped leaf weights, partial
disease override, probability/margin gates, and stopping rules. Repeated frames
do not increase the independent image count. A missed detection stays
`NO_LEAVES`; there is no invented whole-image fallback in the app replay.

These sources were used during model development. The original training cache
and image inventory are absent, so this is a diagnostic source replay and not
an independently held-out generalization benchmark. Exact and perceptual
duplicate checks are included, but cannot rule out all training overlap.
Unsupported conditions and non-tomato plants map to Other. Mixed-label photos
are excluded from scalar image accuracy; valid leaf annotations can still be
used for object-level diagnostics. Source labels are not a new agronomy review.
Known exact source TRAIN/TEST overlaps are excluded from primary metrics;
raw predictions remain available with an analysis-eligibility flag.
