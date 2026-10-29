"""One bounded two-snapshot alternative after the single-snapshot OR failed."""
from pathlib import Path
import argparse,json,time
import numpy as np
from conditional_recovery_policy import recover_only
from opened_context_data import load,read
from phase85_population_sft import evaluation_data
from permissive_encoder import digest
from population_siglip_tail import freeze


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True);p=Path(parser.parse_args().campaign);out=p/'phase201_two_snapshot_coverage';out.mkdir(exist_ok=True);started=time.monotonic();parent=p/'phase200_cached_admission_policy_preflight';pp=read(parent/'protocol.json')
    assert read(parent/'result.json')['maximum_or_healthy']<28
    protocol=dict(source_sha256=digest(__file__),parent_protocol_sha256=digest(parent/'protocol.json'),native_branches=pp['native_prediction_sha256'],mobile_branches=pp['mobile_prediction_sha256'],
        fixed_policy='Exactly two saved native snapshots OR the fixed four-head mobile OR, followed by the separately fixed .05 rotation-consistent negative veto on NEW admissions only',maximum_native_branches=2,healthy_admission=.95,original_gates_unchanged=True,
        stopping='Compute the two-snapshot coverage ceiling from existing predictions before any new inference; do not expand to three or more snapshots. No validation cutoff fitting or independent-acceptance claim.',new_encoder_inputs=0,acceptance_data_used=False,ad_independent=False,olid_independent=False)
    freeze(out/'protocol.json',protocol)
    if (out/'result.json').exists():assert read(out/'result.json')['protocol_sha256']==digest(out/'protocol.json');return
    selection=read(p/'phase118_conditional_recovery/protocol.json');ids=np.array(selection['validation_indices']);rows=read(p/'phase84_population_head/fitting_rows.json')
    def vector(name,sha):
        path=p/name;assert digest(path)==sha;z=np.load(path);assert np.array_equal(z['indices'],ids);return z['probability']
    mobile=np.stack([vector(name,sha) for name,sha in pp['mobile_prediction_sha256'].items()]).max(0);names=list(pp['native_prediction_sha256']);bank=np.stack([np.maximum(mobile,vector(name,pp['native_prediction_sha256'][name])) for name in names])
    data=load(p);entries,_,packets,before,previous=evaluation_data(p,data);items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({r['path']:(r,packets[r['id']],retained) for r,retained in zip(before,previous)})
    admissible=np.array([recover_only(*[items[rows[i]['path']][0],items[rows[i]['path']][2],items[rows[i]['path']][1]],1.,data['manifest'])['app_replay']['kind']=='HEALTHY' for i in ids])
    positive=admissible&np.array([rows[i]['y'] and rows[i]['source']=='olid_tomato' for i in ids]);negative=admissible&np.array([not rows[i]['y'] for i in ids]);active=(bank>=.95)&admissible;reports=[];maximum=0
    for first in range(len(names)):
        for second in range(first+1,len(names)):
            union=active[first]|active[second];healthy=selection['fixed_healthy_validation_correct']+int((union&positive).sum());maximum=max(maximum,healthy)
            if healthy>=28:reports.append(dict(first=names[first],second=names[second],healthy_before_veto=healthy,new_false_healthy_before_veto=int((union&negative).sum())))
    np.savez(out/'candidate_bank.npz',indices=ids,native_names=np.array(names),probability=bank,admissible=admissible,protocol_sha256=digest(out/'protocol.json'))
    result=dict(protocol_sha256=digest(out/'protocol.json'),candidate_bank_sha256=digest(out/'candidate_bank.npz'),saved_native_branches=len(names),pairs_tested=len(names)*(len(names)-1)//2,maximum_two_snapshot_healthy=maximum,coverage_capable_pairs=len(reports),controls=reports,
        next_action='Evaluate only the predeclared fixed rotation-veto alternatives' if reports else 'Stop two-snapshot policy; even pre-veto coverage misses the gate',model_promoted=False,new_encoder_inputs=0,acceptance_data_used=False,elapsed_seconds=time.monotonic()-started)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='controls'}),flush=True)


if __name__=='__main__':main()
