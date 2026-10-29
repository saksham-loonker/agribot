"""Explain unused consumed supervision without fitting or weakening quarantine."""
from pathlib import Path
import argparse, collections, hashlib, json


def audit(p):
    read = lambda path: json.loads(path.read_text())
    rows = read(p/'phase84_population_head/fitting_rows.json')
    protocol = read(p/'phase127_clinical_anchors/protocol.json')
    selected = set(protocol['train_indices'])
    bags = read(p/'phase87_diagnostic_bag_features/bags.json')
    descriptors = read(p/'phase29_compact/inputs.json') + read(p/'phase87_diagnostic_bag_features/new_inputs.json')
    sources = read(p/'phase29_compact/source_files.json')
    sources.update({r['path']: r['file_sha256'] for r in read(p/'phase24_olid_source/sealed_manifest.json')})
    validation = [r for r in rows if r['validation']]
    files = {sources[r['path']] for r in validation if r['path'] in sources}
    groups = {r['group'] for r in validation}
    def inputs(row):
        return {descriptors[k]['input_sha256'] for view in bags.get(row['path'], {}).values() for k in view}
    pixels = set().union(*(inputs(r) for r in validation))
    excluded = set(protocol['quarantined_overlap_groups'])
    cases = []
    for i, r in enumerate(rows):
        if r['source'] != 'olid_tomato' or r['validation']:
            continue
        reasons = []
        if r['group'] in groups: reasons.append('validation_family')
        if sources.get(r['path']) in files: reasons.append('validation_file')
        if inputs(r) & pixels: reasons.append('validation_roi_input')
        if r['group'] in excluded: reasons.append('quarantined_overlap_family')
        if not bags.get(r['path'], {}).get('center'): reasons.append('no_cached_detected_roi')
        cases.append(dict(index=i, label=r['label'], selected=i in selected, reasons=reasons))
    summary = {}
    for label in sorted({r['label'] for r in cases}):
        part = [r for r in cases if r['label'] == label]
        omitted = [r for r in part if not r['selected']]
        summary[label] = dict(nominal_training=len(part), selected=sum(r['selected'] for r in part), omitted=len(omitted),
                              omitted_reasons=dict(collections.Counter(reason for r in omitted for reason in r['reasons'])),
                              unexplained_indices=[r['index'] for r in omitted if not r['reasons']])
    return dict(summary=summary, cases=cases, diagnostic_only=True, new_encoder_inputs=0,
                ad_independent=False, olid_independent=False, acceptance_data_used=False,
                model_promoted=False, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--campaign', required=True)
    p = Path(parser.parse_args().campaign)
    result = audit(p); out = p/'phase242_training_coverage_audit'; out.mkdir(exist_ok=True)
    target = out/'result.json'
    if target.exists():
        assert json.loads(target.read_text()) == result, 'Audit identity changed; preserve prior result first'
    else:
        target.write_text(json.dumps(result, indent=2))
    print(json.dumps(result['summary'], indent=2))
