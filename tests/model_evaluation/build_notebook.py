"""Create and execute the analysis notebook from reviewed inference records."""
import csv, json, os, sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client.kernelspec import KernelSpecManager

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'dist/model-evaluation-20261015'
for name,suffix in [('IPYTHONDIR','ipython'),('JUPYTER_CONFIG_DIR','config'),('JUPYTER_RUNTIME_DIR','runtime')]:
    folder=ROOT/'.toolchains/jupyter'/suffix; folder.mkdir(parents=True,exist_ok=True); os.environ[name]=str(folder)
os.environ['JUPYTER_PATH']=str(ROOT/'.venv/share/jupyter')

summary=json.loads((OUT/'analysis_summary.json').read_text())
lines=[]
for row in summary['source_summary']:
    if row['mode']!='app': continue
    lines.append(f"- **{row['source']}**: {row['known_false_other']}/{row['known_n']} ({row['known_false_other_rate']:.1%}) supported-condition images returned Unknown; {row['known_unsure']} were Unsure and {row['known_no_leaves']} had no detected leaf.")
field=next(row for row in summary['source_summary'] if row['source']=='plantdoc' and row['mode']=='app')
sweep=list(csv.DictReader((OUT/'bias_sensitivity.csv').open(encoding='utf-8-sig')))
zero=next(row for row in sweep if row['source']=='plantdoc' and float(row['other_logit_bias'])==0 and float(row['other_min_probability'])==.5)
healthy=next(row for row in summary['oracle_summary'] if row['label']=='Healthy')
healthy_cohorts=[row for row in summary['per_class'] if row['label']=='Healthy' and row['mode']=='app']
healthy_text='Healthy app verdicts were correct on '+', '.join(f"{row['correct']}/{row['n']} {row['source']} photos" for row in healthy_cohorts)+'. These are small source-specific cohorts, not deployment error rates.'
interpretation=(f"**Main finding:** false Unknown verdicts concentrate in the field-photo subset. Removing the Other bias improves its correct supported-condition verdicts "
                f"from {field['known_correct']} to {zero['known_correct']}, but increases unsupported inputs given known labels from {field['unknown_false_known']} to {zero['unknown_false_known']}. "
                f"Even author-box crops produce Other on {healthy['false_other']}/{healthy['n']} healthy-labeled crops. "
                "Prioritize label review and representative supervised data; do not optimize a blanket reward for fewer Unknown outputs.")
notebook=nbformat.v4.new_notebook()
cells=[]
def md(text): cells.append(nbformat.v4.new_markdown_cell(text))
def code(text): cells.append(nbformat.v4.new_code_cell(text))
md('# Agribot: why does the shipped model return “Problem not in our list”?\n\n'
   'Diagnostic evaluation of **952 downloaded images**, unchanged shipped TFLite models, and source-defined app decision rules.\n\n'
   '## Findings\n\n'+interpretation+'\n\n'+healthy_text+'\n\n'+'\n'.join(lines)+'\n\n'
   '**Interpretation limits:** these public sources overlap the development pipeline. This is a diagnostic source replay, '
   'not a new, independently held-out deployment benchmark. Static repeated-frame replay tests deterministic consensus; '
   'it does not create independent frames or reproduce a moving camera. Labels come from dataset authors, not a new agronomist review.')
md('## 1. Reproduce the investigation\n\n'
   'From the repository root, run the commands below in the project virtual environment. Downloads are pinned to source revisions and seeded where sampled. '
   'The classifier and detector SHA-256 hashes are checked against the shipped manifest before inference.\n\n'
   '```powershell\n'
   'python tests/model_evaluation/download_public.py\n'
   'python tests/model_evaluation/evaluate_public.py\n'
   'python tests/model_evaluation/evaluate_public.py --stress\n'
   'python tests/model_evaluation/run_controls.py\n'
   'python tests/model_evaluation/verify_kotlin_replay.py\n'
   'python tests/model_evaluation/analyze_public.py\n'
   'python tests/model_evaluation/build_notebook.py\n```\n\n'
   'The inference script resumes from per-image records. To intentionally rerun inference, choose a fresh output folder or move the existing predictions aside first.')
code("from pathlib import Path\nimport sys, json, io, contextlib\nimport pandas as pd\nfrom IPython.display import display, Image, Markdown\n"
     "root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'tests/model_evaluation/evaluate_public.py').exists())\n"
     "output = root / 'dist/model-evaluation-20261015'\n"
     "sys.path.insert(0, str(root/'tests/model_evaluation'))\n"
     "import analyze_public\n"
     "with contextlib.redirect_stdout(io.StringIO()):\n    summary = analyze_public.main()\n"
     "image_results = pd.read_csv(output / 'image_results.csv')\n"
     "assert len(image_results) == 952\n"
     "display(image_results.groupby('source').size().rename('Downloaded images').to_frame())")
md('## 2. Source and data-quality checks\n\n'
   'Tomato-Village uses the authors’ entire Variant-A test split. PlantDoc uses TEST images with usable XML annotations. '
   'PlantVillage uses a fixed seed (20261015), 50 images for each of Early blight, Late blight and Healthy, and 15 for each of seven unsupported tomato conditions. '
   'Non-tomato PlantDoc labels are out of scope for a tomato classifier and map to Other; mixed images are excluded from single-label image accuracy. '
   'The pooled rejection rate depends on this chosen class mix and must not be interpreted as a farmer-camera rejection rate.\n\n'
   'Sources: [Tomato-Village](https://github.com/mamta-joshi-gehlot/Tomato-Village), '
   '[PlantDoc](https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset), '
   '[PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset). '
   'Original source notices and readmes are retained in `source/`; downloaded images remain local.')
code("quality = summary['data_quality']\n"
     "display(pd.DataFrame([{'Check': key, 'Value': quality[key]} for key in ['downloaded', 'exact_unique', 'analysis_n', 'exact_duplicate_rows', 'near_duplicate_pairs_hamming4', 'near_unique_n', 'golden_pixel_overlap']]))\n"
     "display(Markdown('**Training-overlap caveat:** ' + quality['warning']))\n"
     "display(pd.DataFrame(summary['near_duplicate_sensitivity'])[['source','n','known_n','known_false_other','unknown_n','unknown_false_known']])")
code("display(pd.DataFrame(quality['plantvillage_repeated_leaf_identifiers']))\n"
     "display(pd.DataFrame([quality['plantvillage_leaf_identifier_sensitivity']])[['n','known_n','known_false_other']])")
md('Exact pixel duplicates are counted once in primary metrics; conflicting exact-label duplicates are excluded. '
   'Known exact file overlaps with original TRAIN/validation partitions are also excluded, including 11 PlantDoc TEST photos. '
   'All raw predictions are retained with an analysis-eligibility flag, and raw source metrics are saved in the summary JSON. '
   'The last table retains one image per connected perceptual-hash cluster (Hamming distance ≤4) as a sensitivity check. '
   'Perceptual similarity is a screening heuristic, not proof of identical plants or wrong labels. '
   'The nominal Wilson intervals below do not account for all source or plant correlations. '
   'The two lab rejection images share the filename leaf identifier GHLB2 Leaf 155; a separate conservative identifier-group sensitivity keeps one view. '
   'Filename grouping is not a complete physical-leaf inventory. '
   'The historical model manifest reports whole-image accuracy on 329 leak-cleaned Tomato-Village test images; '
   'the original clean-membership list is absent here, so this evaluation uses the full 461-image author test split. '
   'Do not interpret a numerical difference from that historical result as a model regression or improvement.')
md('## 3. Verify model identity and regressions')
code("golden = pd.read_json(output / 'golden_checks.json')\n"
     "assert golden.whole_pass.all() and golden.detector_pass.all()\n"
     "display(golden[['file','max_abs_logit_diff','box_px_diff','whole_pass','detector_pass']])\n"
     "display(pd.DataFrame(summary['tests']).T)")
md('The 13 golden images check preprocessing, logits, and detector outputs against shipped reference data. '
   'They are correctness controls and are not 13 independent accuracy-test samples. '
   'Python and Android test logs accompany this notebook; device tests use static bundled images and no camera. '
   'The optimized sparse resize is checked against every pixel of both dense-reference input tensors on all 13 controls. '
   'Raw saved logits are separately replayed through the actual compiled Kotlin EnsembleDecision and PlantEvidence classes.')
code("display(pd.read_json(output/'kotlin_replay_verification.json'))\n"
     "display(pd.DataFrame([summary['device_golden_tests']]))")
md('## 4. Outcomes on supported conditions\n\n'
   '“Unknown” here means the simulated app produces UNKNOWN_CONDITION. “Unsure” and “No leaves” are separate outcomes. '
   'Whole-image classification is a diagnostic path; the live Check a plant flow classifies detected leaves, without a whole-image fallback when detection is empty. '
   'Tomato-Village covers eight supported classes, whereas the other two sources cover three; the source totals have different class mixtures. '
   'The common-class table uses equal class weights for recall to make that comparison more interpretable, without proving the cause of any remaining source difference.')
code("metrics = pd.read_csv(output / 'source_metrics.csv')\n"
     "display(metrics[['source','mode','n','known_n','known_correct','known_false_other','known_unsure','known_no_leaves','unknown_n','unknown_rejected','unknown_false_known']])\n"
     "display(Image(filename=str(output/'figures/known_condition_outcomes.png')))\n"
     "display(pd.DataFrame(summary['common_class_comparison']))")
md('## 5. Class-specific errors and confusion')
code("class_metrics = pd.read_csv(output / 'class_metrics.csv')\n"
     "display(class_metrics.query(\"source == 'tomatovillage' and mode == 'app'\") [['label','n','correct','precision','recall','f1','known_false_other','unsure','no_leaves']])\n"
     "display(Image(filename=str(output/'figures/per_class_accuracy.png')))\n"
     "display(Image(filename=str(output/'figures/confusion_matrix.png')))")
md('## 6. Is the detector or crop responsible?\n\n'
   'The oracle diagnostic applies the same classifier to author-annotated leaf boxes, with the same 10% crop padding. '
   'Invalid boxes and mismatched annotation dimensions are excluded from this crop analysis; their image labels can still support image-level diagnostics. '
   'Detector coverage means a predicted leaf overlaps the annotation with IoU ≥0.5. Oracle results are object-level and must not be directly pooled with image-level rates.')
code("display(pd.read_csv(output/'oracle_class_metrics.csv'))\n"
     "display(Markdown(f\"Excluded annotation objects: **{summary['invalid_oracle_boxes']}**. Mixed-label images excluded from scalar image accuracy: **{summary['mixed_label_images']}**.\"))")
md('## 7. Does the positive Other bias explain rejection?\n\n'
   'The shipped decision adds **+0.75** to Other after combining classifier logits; temperature is about **0.88147** and the optional energy gate is **off**. '
   'With one classifier, the raw class ordering is unchanged by softmax; adding the bias favors Other against marginal known classes. '
   'The app additionally requires Other probability ≥0.50 before displaying an unknown-condition verdict. '
   'This sweep recomputes decisions from saved logits while holding model weights, images, and detections fixed. '
   'It is diagnostic counterfactual analysis, not a threshold fitted for release. Reducing unknown outcomes may increase false known-condition predictions on unsupported conditions.')
code("display(pd.DataFrame(summary['bias_paired']).T)\n"
     "display(Image(filename=str(output/'figures/bias_tradeoff.png')))\n"
     "bias_metrics = pd.read_csv(output/'bias_sensitivity.csv')\n"
     "display(bias_metrics.query(\"other_min_probability == 0.5 and other_logit_bias in [0.0, 0.75]\") [['source','other_logit_bias','known_n','known_false_other','unknown_n','unknown_false_known','unsure']])")
code("scope_metrics = pd.read_csv(output/'scope_metrics.csv')\n"
     "display(scope_metrics.query(\"scope != 'supported_tomato'\") [['source','scope','other_logit_bias','n','unknown_rejected','unknown_false_known']])")
md('## 8. Controlled robustness probes\n\n'
   'Three source-test images per supported class (24 total) are each darkened to 50% brightness, blurred at radius 2, and rotated by 90°. '
   'This isolates whole-image classifier sensitivity. These are correlated transformations of existing test images, not 72 new independent samples.')
code("display(pd.DataFrame(summary.get('stress', [])))")
md('### Non-plant controls\n\n'
   'These four generated inputs are not downloaded photos and are not included in any source-accuracy denominator. '
   'They test whether a blank or noisy image is silently converted into a known disease. A false leaf detection and a classifier rejection are distinct events.')
code("display(pd.DataFrame(summary.get('synthetic_controls', [])))")
md('## 9. Confidence and the rejection mechanism\n\n'
   'Expected calibration error (ECE) uses 10 equal-width confidence bins, weighted by sample count. '
   'Negative log likelihood and the unnormalized nine-class Brier score use author labels. '
   'These are source-mix diagnostics, not a claim that deployment confidence is calibrated. '
   'The mechanism table distinguishes raw classifier Other predictions from cases flipped to Other by the explicit bias.')
code("display(pd.DataFrame(summary['calibration']))\n"
     "display(pd.DataFrame(summary['whole_rejection_mechanism']))\n"
     "display(Image(filename=str(output/'figures/reliability.png')))")
md('## 10. Inspect rejection examples')
code("example = output/'figures/false_other_examples.png'\n"
     "if example.exists(): display(Image(filename=str(example)))\n"
     "display(image_results.query(\"app_kind == 'UNKNOWN_CONDITION' and label != 'Other' and label != 'Mixed'\") [['id','source','label','app_confidence','classified_leaves']].sort_values('app_confidence',ascending=False).head(12))")
md('## Scope and next decision\n\n'
   'Use these results to distinguish false Other predictions on supported conditions, correct rejection of unsupported conditions, detector failures, '
   'and genuinely low-confidence disagreements. Retraining is justified only if labeled supported-condition errors remain after a clean pipeline check. '
   'Any bias or threshold candidate needs a separately held-out validation set, with an explicit guardrail for falsely assigning known diseases to unsupported conditions. '
   'No model weight, app threshold, or production decision rule was changed by this evaluation.\n\n'
   'Raw evidence: `images_manifest.json`, `predictions.jsonl`, `image_results.csv`, `bias_sensitivity.csv`, '
   '`oracle_object_results.csv`, `analysis_summary.json`, and regression logs. '
   'Host inference elapsed times are diagnostic runtime measurements, not phone-camera FPS or field latency.')
notebook.cells=cells
notebook.metadata={'kernelspec':{'display_name':'Agribot evaluation','language':'python','name':'agribot-evaluation'},'language_info':{'name':'python','version':sys.version.split()[0]}}
path=OUT/'agribot_model_diagnostics.ipynb'; nbformat.write(notebook,path)
ksm=KernelSpecManager(kernel_dirs=[str(ROOT/'.venv/share/jupyter/kernels')])
client=NotebookClient(notebook,timeout=300,kernel_name='agribot-evaluation',resources={'metadata':{'path':str(ROOT)}})
client.execute(); nbformat.validate(notebook); nbformat.write(notebook,path)
html,_=HTMLExporter().from_notebook_node(notebook)
(OUT/'agribot_model_diagnostics.html').write_text(html,encoding='utf-8')
print('Executed notebook:',path,flush=True)
print('HTML preview:',OUT/'agribot_model_diagnostics.html',flush=True)
