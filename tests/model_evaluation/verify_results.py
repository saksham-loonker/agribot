"""Independent count/normalization/provenance checks; no shared metric helper."""
from pathlib import Path
import hashlib, json
from collections import Counter
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'dist/model-evaluation-20261015'
rows=[json.loads(line) for line in (OUT/'predictions.jsonl').read_text().splitlines()]
source=json.loads((OUT/'images_manifest.json').read_text())
summary=json.loads((OUT/'analysis_summary.json').read_text())
assert len(rows)==len(source)==952
assert len({r['id'] for r in rows})==len(rows)
checks=[]
repositories={'tomatovillage':'Tomato-Village','plantdoc':'PlantDoc-Object-Detection-Dataset','plantvillage':'PlantVillage-Dataset'}
revisions={source:json.loads((OUT/'source'/(repo+'_commit.json')).read_text())['sha'] for source,repo in repositories.items()}
for row in rows:
    path=ROOT/row['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
    assert row['source_commit']==revisions[row['source']]
    assert abs(sum(row['whole_probabilities'])-1)<1e-6
    for leaf in row['leaves']: assert abs(sum(leaf['probabilities'])-1)<1e-6
    # Independent one-member softmax computation directly from raw logits.
    z=np.array(row['whole_member_logits'][0],dtype=float)
    z[-1]+=.75; z/=.8814700841903687
    p=np.exp(z-z.max()); p/=p.sum()
    assert np.max(np.abs(p-row['whole_probabilities']))<1e-6
    assert len(row['leaves'])<=6
checks.append('All source-file SHA-256 values and probability normalizations verified; direct single-member logit formula agrees.')
groups={}
for row in rows: groups.setdefault(row['pixel_sha256'],[]).append(row)
unique=[g[0] for g in groups.values() if len({r['label'] for r in g})==1]
tv_tree=json.loads((OUT/'source/Tomato-Village_subtree0.json').read_text())['tree']
tv_train={e['sha'] for e in tv_tree if e['type']=='blob' and e['path'].split('/')[0] in ('train','val')}
tv_overlap={e['path'] for e in tv_tree if e['type']=='blob' and e['path'].startswith('test/') and e['sha'] in tv_train}
pd_tree=json.loads((OUT/'source/PlantDoc-Object-Detection-Dataset_tree.json').read_text())['tree']
pd_train={e['sha'] for e in pd_tree if e['path'].startswith('TRAIN/') and e['path'].lower().endswith('.jpg')}
pd_overlap={e['path'] for e in pd_tree if e['path'].startswith('TEST/') and e['path'].lower().endswith('.jpg') and e['sha'] in pd_train}
excluded={r['id'] for r in unique if (r['source']=='plantdoc' and r['source_path'] in pd_overlap) or (r['source']=='tomatovillage' and r['source_path'].split('/',1)[1] in tv_overlap)}
assert excluded==set(summary['data_quality']['excluded_training_overlap_ids'])
unique=[r for r in unique if r['id'] not in excluded]
for item in summary['source_summary']:
    selected=[r for r in unique if r['source']==item['source'] and r['label']!='Mixed']
    known=[r for r in selected if r['label']!='Other']; unknown=[r for r in selected if r['label']=='Other']
    assert item['n']==len(selected) and item['known_n']==len(known) and item['unknown_n']==len(unknown)
    if item['mode']=='whole':
        false_other=sum(r['whole_label']=='Other' for r in known)
        correct=sum(r['whole_label']==r['label'] for r in selected)
    else:
        false_other=sum(r['app_replay']['kind']=='UNKNOWN_CONDITION' for r in known)
        correct=sum(r['app_replay']['label']==r['label'] and r['app_replay']['kind'] not in ['UNSURE','NO_LEAVES','NEED_MORE_VIEWS'] for r in selected)
    assert item['known_false_other']==false_other and item['correct']==correct
    if known: assert abs(item['known_false_other_rate']-false_other/len(known))<1e-12
checks.append('Image-level source denominators, correctness counts, and false-Other counts independently reconcile.')
for item in summary['source_summary']:
    if item['mode']=='app':
        assert item['known_correct']+item['known_false_other']+item['known_unsure']+item['known_no_leaves']<=item['known_n']
reliability=list(__import__('csv').DictReader((OUT/'reliability_bins.csv').open(encoding='utf-8-sig')))
for source in repositories:
    assert sum(int(r['n']) for r in reliability if r['source']==source)==next(r['n'] for r in summary['source_summary'] if r['source']==source and r['mode']=='whole')
for module,result in summary['tests'].items():
    assert result['tests']>0, module+' did not run any tests'
    assert result['failures']==result['errors']==0, module+' has regression failures'
assert all(r['whole_pass'] and r['detector_pass'] for r in json.loads((OUT/'golden_checks.json').read_text()))
assert json.loads((OUT/'download_failures.json').read_text())==[]
assert json.loads((OUT/'evaluation_errors.json').read_text())==[]
kotlin=json.loads((OUT/'kotlin_replay_verification.json').read_text())
assert all(r['images']==952 and not r['mismatches'] for r in kotlin)
device=summary['device_golden_tests']
assert device['tests']==4 and device['failures']==device['errors']==0
checks.append('All six JVM module suites and 13 golden-image controls passed; no download or inference errors.')
checks.append('All 952 static replays match compiled Kotlin decision classes at shipped and zero Other bias; all four device golden tests passed.')
(OUT/'validation.json').write_text(json.dumps(dict(status='Share with caveats',checks=checks,
    caveats=['Public source overlap with model development; no new independent held-out benchmark.',
             'Static replay uses repeated copies of each photo, not independent views or a real camera.',
             'Source labels have not been re-adjudicated by an agronomist.',
             'Nominal per-image intervals do not account for all image/plant correlations.']),indent=2))
print('\n'.join(checks))
