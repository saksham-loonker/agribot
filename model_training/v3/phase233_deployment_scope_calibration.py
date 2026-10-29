"""Training-only deployment-scope calibration of fixed auxiliary-trained heads."""
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from phase215_shrinkage_density import inputs
from clinical_replay_balance import clinical_replay_weights
from conditional_recovery_policy import recover_only
from opened_context_data import load,read
from phase85_population_sft import evaluation_data
from permissive_encoder import digest
from population_siglip_tail import freeze

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',required=True);p=Path(ap.parse_args().campaign);out=p/'phase233_deployment_scope_calibration';out.mkdir(exist_ok=True);started=time.monotonic();rows,tr,val,x,v,w,keys,parents=inputs(p,'mobile');parent=p/'phase232_auxiliary_nutrition_heads';pp=read(parent/'protocol.json');assert tr==pp['original_training_indices'] and val==pp['validation_indices'];methods=pp['methods'];paths={m:parent/m/'predictions.npz' for m in methods};full=read(p/'phase193_exhaustive_mobile_replay/protocol.json');roles=[full['target_indices'],full['old_anchor_indices'],full['field_training_indices']];assert sum(roles,[])==tr;lookup={i:j for j,i in enumerate(tr)};protocol=dict(source_sha256=digest(__file__),parent_protocol_sha256=digest(parent/'protocol.json'),prediction_sha256={m:digest(path) for m,path in paths.items()},base_checkpoint_sha256={m:digest(parent/m/'full/last.pt') for m in methods},methods=methods,original_training_indices=tr,validation_indices=val,hypothesis='Cucumber auxiliary labels teach generic stress but are outside the app deployment crop. Compare one fixed original-scope OOF confidence mapping against the already frozen all-source calibration, with unchanged trained weights and .95 admission.',calibration='Logistic C1 on only original-scope selected training OOF margins. Exclude every appended auxiliary position; preserve .5 target/.25 replay/.25 field mass. No validation labels fit coefficients. The held auxiliary fold remains diagnostic and not deployment probability supervision.',healthy_admission=.95,original_gates_unchanged=True,existing_healthy_and_named_disease_exact=True,auxiliary_source_calibration_used=False,new_encoder_inputs=0,acceptance_data_used=False,ad_independent=False,olid_independent=False,stopping='Exactly original-scope calibration for the two fixed auxiliary-trained heads; no source subset/C/threshold grid. Nutrient specialist remains negative-only. Full qualification only after >=28/40 and zero new falseHealthy.')
    freeze(out/'protocol.json',protocol)
    if (out/'result.json').exists():assert read(out/'result.json')['protocol_sha256']==digest(out/'protocol.json');return
    data=load(p);entries,_,packets,before,previous=evaluation_data(p,data);items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({r['path']:(r,packets[r['id']],retained) for r,retained in zip(before,previous)});selection=read(p/'phase118_conditional_recovery/protocol.json');controls=[]
    for method,path in paths.items():
        z=np.load(path);assert np.array_equal(z['indices'],val) and str(z['model_sha256'])==protocol['base_checkpoint_sha256'][method];selected=z['selected_training_positions'];selected=selected[selected<len(tr)];selected_set=set(selected);weight=np.zeros(len(tr));calibration_roles=[]
        for mass,role in zip((.5,.25,.25),roles):
            ids=[i for i in role if lookup[i] in selected_set];positions=[lookup[i] for i in ids];weight[positions]=mass*clinical_replay_weights(rows,ids);calibration_roles.append(ids)
        weight*=len(selected);oof=z['oof_training_margin'][selected];assert np.isfinite(oof).all();y=np.array([rows[tr[i]]['y'] for i in selected]);cal=LogisticRegression(C=1.,max_iter=2000).fit(oof.reshape(-1,1),np.repeat(y,2),sample_weight=np.repeat(weight[selected]/2,2));a=float(cal.coef_[0,0]);b=float(cal.intercept_[0]);assert a>0;raw=z['validation_margin'].astype(np.float64);prob=expit(a*raw+b).min(1);folder=out/method;folder.mkdir(exist_ok=True);np.savez(folder/'model.npz',calibration=np.array([a,b]),base_checkpoint_sha256=protocol['base_checkpoint_sha256'][method]);restored=np.load(folder/'model.npz',allow_pickle=False)['calibration'];assert np.array_equal(expit(restored[0]*raw+restored[1]).min(1),prob);healthy=selection['fixed_healthy_validation_correct'];old_h=0;errors={}
        for i,score in zip(val,prob):
            row=rows[i];original,packet,retained=items[row['path']];now=recover_only(original,retained,packet,float(score),data['manifest'])['app_replay']['kind']=='HEALTHY'
            if not row['y']:errors[row['source']]=errors.get(row['source'],0)+int(now)
            elif row['source']=='olid_tomato':healthy+=int(now)
            else:old_h+=int(now)
        np.savez(folder/'predictions.npz',indices=np.array(val),probability=prob,validation_margin=raw,model_sha256=digest(folder/'model.npz'),base_checkpoint_sha256=protocol['base_checkpoint_sha256'][method],protocol_sha256=digest(out/'protocol.json'));r=dict(method=method,healthy_validation_correct=healthy,new_false_healthy_by_source=errors,old_healthy_correct=old_h,calibration=[a,b],calibration_training_positions=selected.tolist(),calibration_roles=calibration_roles,model_sha256=digest(folder/'model.npz'),diagnostic_standalone_verdicts_only=method=='nutrition_veto_only',cheap_pass=method=='general_auxiliary' and healthy>=28 and sum(errors.values())==0);controls.append(r);print(json.dumps({k:v for k,v in r.items() if k not in ('calibration_training_positions','calibration_roles')}),flush=True)
        if r['cheap_pass']:freeze(folder/'candidate.json',dict(**r,full_qualification_required=True,model_promoted=False))
    (out/'result.json').write_text(json.dumps(dict(protocol_sha256=digest(out/'protocol.json'),controls=controls,elapsed_seconds=time.monotonic()-started,model_promoted=False,new_encoder_inputs=0,acceptance_data_used=False,ad_independent=False,olid_independent=False),indent=2))

if __name__=='__main__':main()
