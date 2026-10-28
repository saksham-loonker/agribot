"""Separate clinical-label disagreements from exact-verdict retention failures."""
from pathlib import Path
from collections import Counter
import argparse,json,time
from opened_context_data import load,read
from phase85_population_sft import evaluation_data
from permissive_encoder import digest
from population_siglip_tail import freeze


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True);p=Path(parser.parse_args().campaign);out=p/'phase197_veto_label_audit';out.mkdir(exist_ok=True);started=time.monotonic()
    parents=['phase192_confident_veto_audit','phase194_exhaustive_veto_audit'];protocol=dict(source_sha256=digest(__file__),parents={name:digest(p/name/'result.json') for name in parents},diagnostic_only=True,labels_consumed_development=True,gates_unchanged=True,new_encoder_inputs=0,acceptance_data_used=False)
    freeze(out/'protocol.json',protocol)
    data=load(p);entries,_,_,_,_=evaluation_data(p,data);cases=[r for group in list(data['base'].values())+[data['internal']] for r in group if entries[id(r)][1]['app_replay']['kind']=='HEALTHY'];controls=[]
    for name in parents:
        for result in read(p/name/'result.json')['controls']:
            paths=Counter(result['protected_changes']);affected=[r for r in cases if r['path'] in paths];assert Counter(r['path'] for r in affected)==paths
            counts=Counter(r.get('label','MISSING') for r in affected);controls.append(dict(parent=name,method=result['method'],protected_vetoes=len(affected),publisher_labeled_healthy=counts.get('Healthy',0),publisher_labeled_nonhealthy=sum(n for label,n in counts.items() if label not in ('Healthy','MISSING')),missing_clinical_label=counts.get('MISSING',0),by_label=dict(counts),by_source_and_label=dict(Counter(str(r['source'])+'|'+str(r.get('label','MISSING')) for r in affected)),unique_paths=len(paths)))
    result=dict(protocol_sha256=digest(out/'protocol.json'),controls=controls,diagnostic_only=True,model_promoted=False,original_gates_unchanged=True,new_encoder_inputs=0,acceptance_data_used=False,elapsed_seconds=time.monotonic()-started,
        interpretation='A changed protected verdict is an exact-retention failure, but not automatically a loss of clinically correct recognition. Publisher labels are not new clinical adjudication. Potential corrections do not authorize weakening any existing gate.')
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
