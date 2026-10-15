"""Record executable source identities and verify the delivered notebook artifacts."""
import datetime, hashlib, json, subprocess
from collections import Counter
from pathlib import Path
import nbformat

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'dist/model-evaluation-20261015'
files=list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('*.java'))
files += [ROOT/'model_training/v2/final/canon.py',ROOT/'agribot_android_app/android/app/src/main/assets/model_manifest.json',
          ROOT/'agribot_android_app/android/ml/src/main/java/com/sakshyam/agribot/ml/vision/EnsembleDecision.kt',
          ROOT/'agribot_android_app/android/domain/src/main/java/com/sakshyam/agribot/domain/scan/PlantEvidence.kt',
          ROOT/'agribot_android_app/android/featurescan/src/main/java/com/sakshyam/agribot/featurescan/check/CheckPlantViewModel.kt']
hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
notebook=nbformat.read(OUT/'agribot_model_diagnostics.ipynb',as_version=4); nbformat.validate(notebook)
code=[cell for cell in notebook.cells if cell.cell_type=='code']
assert all(cell.execution_count is not None for cell in code)
assert not any(output.output_type=='error' for cell in code for output in cell.outputs)
images=sum('image/png' in output.get('data',{}) for cell in code for output in cell.outputs)
assert images==6, f'Expected six figure outputs; got {images}'
html=(OUT/'agribot_model_diagnostics.html').read_text(encoding='utf-8')
assert html.count('data:image/png;base64,')>=6
for text in ['Main finding','PlantDoc','Does the positive Other bias explain rejection','Scope and next decision']:
    assert text in html, 'Missing notebook section: '+text
rows=[json.loads(line) for line in (OUT/'predictions.jsonl').read_text().splitlines()]
context=dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_hashes=hashes,
    model_manifest=json.loads((ROOT/'agribot_android_app/android/app/src/main/assets/model_manifest.json').read_text()),
    source_revisions=json.loads((OUT/'source/revisions.json').read_text()),
    threads_by_image=dict(Counter(str(row.get('runtime_threads',4)) for row in rows)),
    preprocessing_by_image=dict(Counter(row.get('preprocessing_backend','dense_reference') for row in rows)),
    notebook_validation=dict(executed_code_cells=len(code),embedded_figures=images,error_outputs=0,html_embedded_figures=html.count('data:image/png;base64,')),
    visual_review='All six standalone figures visually inspected. HTML sections and embedded outputs checked programmatically; no browser layout inspection was performed.')
(OUT/'provenance.json').write_text(json.dumps(context,indent=2),encoding='utf-8')
print(json.dumps(context['notebook_validation'],indent=2))
print('Recorded source identities and runtime provenance.')
