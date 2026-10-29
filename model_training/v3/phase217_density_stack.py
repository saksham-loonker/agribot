"""Small meta-heads trained only on cross-fitted density evidence."""
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from clinical_replay_balance import clinical_replay_weights
from conditional_recovery_policy import recover_only
from density_stack import stack_basis
from opened_context_data import load,read
from phase85_population_sft import evaluation_data
from permissive_encoder import digest
from population_siglip_tail import freeze

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True);p=Path(parser.parse_args().campaign);base=p/'phase215_shrinkage_density';out=p/'phase217_density_stack';out.mkdir(exist_ok=True);started=time.monotonic()
    pp={kind:read(base/kind/'protocol.json') for kind in ('mobile','native')};rr={kind:read(p/('phase84_population_head/fitting_rows.json' if kind=='mobile' else 'phase168_early_cnn_prefix/rows.json')) for kind in ('mobile','native')}
    # Field rows in the mobile namespace are appended after source107; rebuild
    # through the same immutable helper rather than joining numeric row IDs.
    from exhaustive_replay_data import exhaustive_replay_data
    _,_,rr['mobile'],_,_,_,_=exhaustive_replay_data(p)
    lookup={kind:{rr[kind][i]['path']:j for j,i in enumerate(pp[kind]['train_indices'])} for kind in pp};paths=sorted(set(lookup['mobile'])&set(lookup['native']));assert len(paths)>100
    rows=[rr['mobile'][pp['mobile']['train_indices'][lookup['mobile'][path]]] for path in paths]
    for path,row in zip(paths,rows):
        other=rr['native'][pp['native']['train_indices'][lookup['native'][path]]];assert (row['y'],row['label'],row['group'])==(other['y'],other['label'],other['group'])
    val=pp['mobile']['validation_indices'];assert val==pp['native']['validation_indices'];predictions={};x=[];v=[]
    for kind in ('mobile','native'):
        for mode in ('shared','class_shrinkage'):
            path=base/kind/mode/'predictions.npz';predictions[f'{kind}/{mode}']=digest(path);z=np.load(path);assert np.array_equal(z['indices'],val) and str(z['protocol_sha256'])==digest(base/kind/'protocol.json');assert digest(path.with_name('model.npz'))==str(z['model_sha256'])
            x.append(z['oof_training_margin'][[lookup[kind][path_] for path_ in paths]]);v.append(z['validation_margin'])
    x=np.stack(x,axis=-1);v=np.stack(v,axis=-1);assert x.shape==(len(rows),2,4) and v.shape==(len(val),2,4);w=clinical_replay_weights(rows,list(range(len(rows))))*len(rows);y=np.array([r['y'] for r in rows])
    protocol=dict(source_sha256=digest(__file__),basis_source_sha256=digest(Path(__file__).with_name('density_stack.py')),prediction_sha256=predictions,parent_protocol_sha256={kind:digest(base/kind/'protocol.json') for kind in pp},training_paths=paths,training_weights=w.tolist(),validation_indices=val,methods=['linear','quadratic'],regularization_C=1.,fit='Logistic meta-head on cross-fitted base evidence only; weighted center/scale learned on training evidence; two crops have half photo weight; no extra meta-OOF or independent calibration claim',basis='Four density log-likelihood margins per crop; linear versus all four linear terms and ten quadratic products',joins='Exact photo path, y/clinical label/family agreement; never field row indices across namespaces',hypothesis='Shared and class-specific covariance models on different encoder representations may make complementary errors. Fit their evidence jointly instead of another convex probability mixture or validation-tuned threshold.',healthy_admission=.95,original_gates_unchanged=True,existing_healthy_and_named_disease_exact=True,new_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False,stopping='Exactly linear and degree-two logistic C=1; no interaction/coefficient/threshold grid. Only >=28/40 Healthy and zero new false Healthy permits full original qualification. Stop this evidence family if both fail.')
    freeze(out/'protocol.json',protocol)
    if (out/'result.json').exists():assert read(out/'result.json')['protocol_sha256']==digest(out/'protocol.json');return
    flat=x.reshape(-1,4);mass=np.repeat(w/2,2);center=np.average(flat,axis=0,weights=mass);scale=np.sqrt(np.maximum(np.average((flat-center)**2,axis=0,weights=mass),1e-12));data=load(p);entries,_,packets,before,previous=evaluation_data(p,data);items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({r['path']:(r,packets[r['id']],retained) for r,retained in zip(before,previous)});selection=read(p/'phase118_conditional_recovery/protocol.json');controls=[]
    for degree in (1,2):
        folder=out/('linear' if degree==1 else 'quadratic');folder.mkdir(exist_ok=True);model=LogisticRegression(C=1.,max_iter=2000).fit(stack_basis(flat,center,scale,degree),np.repeat(y,2),sample_weight=mass);raw=model.decision_function(stack_basis(v.reshape(-1,4),center,scale,degree)).reshape(len(val),2);probability=expit(raw).min(1);healthy=selection['fixed_healthy_validation_correct'];errors={};old_h=0
        for i,score in zip(val,probability):
            row=rr['mobile'][i];original,packet,retained=items[row['path']];now=recover_only(original,retained,packet,float(score),data['manifest'])['app_replay']['kind']=='HEALTHY'
            if not row['y']:errors[row['source']]=errors.get(row['source'],0)+int(now)
            elif row['source']=='olid_tomato':healthy+=int(now)
            else:old_h+=int(now)
        path=folder/'model.npz';np.savez(path,center=center,scale=scale,degree=np.array(degree),coef=model.coef_,intercept=model.intercept_);restored=np.load(path,allow_pickle=False);replay=(stack_basis(v.reshape(-1,4),restored['center'],restored['scale'],int(restored['degree']))@restored['coef'].T+restored['intercept']).reshape(len(val),2);error=float(np.abs(raw-replay).max());assert error<1e-10 and np.array_equal(expit(replay).min(1)>=.95,probability>=.95)
        np.savez(folder/'predictions.npz',indices=np.array(val),probability=probability,validation_margin=raw,model_sha256=digest(path),protocol_sha256=digest(out/'protocol.json'))
        control=dict(method=folder.name,healthy_validation_correct=healthy,new_false_healthy_by_source=errors,old_healthy_correct=old_h,model_sha256=digest(path),restore_maximum_margin_error=error,cheap_pass=healthy>=28 and sum(errors.values())==0);controls.append(control);print(json.dumps(control),flush=True)
    result=dict(protocol_sha256=digest(out/'protocol.json'),controls=controls,model_promoted=False,new_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False,elapsed_seconds=time.monotonic()-started);(out/'result.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
