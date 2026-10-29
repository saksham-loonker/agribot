"""Bounded cross-crop physiology transfer, with a nutrition-only veto specialist."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np,torch
from torch.nn import functional as F
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from clinical_replay_balance import clinical_replay_weights
from phase215_shrinkage_density import inputs
from leaf_feature_expert import LeafFeatureHead
from hard_negative_ranking import hard_negative_rank_loss
from conditional_recovery_policy import recover_only
from opened_context_data import load,read
from phase85_population_sft import evaluation_data
from permissive_encoder import digest
from population_siglip_tail import freeze
from training_support import atomic_save,capture_rng

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',required=True);p=Path(ap.parse_args().campaign);out=p/'phase232_auxiliary_nutrition_heads';out.mkdir(exist_ok=True);started=time.monotonic()
    if (out/'result.json').exists():assert read(out/'result.json')['source_sha256']==digest(__file__);return
    rows,tr,val,x,v,w,keys,parents=inputs(p,'mobile');parent_path=p/'phase215_shrinkage_density/mobile/protocol.json';prior=read(parent_path);assert tr==prior['train_indices'] and val==prior['validation_indices'];root=p/'phase231_cross_crop_features';complete=read(root/'complete.json');assert complete['protocol_sha256']==digest(root/'protocol.json') and complete['features_sha256']==digest(root/'features.npy') and complete['bags_sha256']==digest(root/'bags.json') and complete['identities_sha256']==digest(root/'input_sha256.json');assert complete['overlap_images']==0 and not read(root/'overlap_audit.json')['whole_collection_quarantined'];identities=read(root/'input_sha256.json');assert not set(identities)&{k for values in keys.values() for k in values},'Auxiliary input overlaps original training/held inputs; quarantine source';values=np.load(root/'features.npy');lookup={key:i for i,key in enumerate(identities)};aux=[];aux_rows=[];family_labels={}
    for name,bag in sorted(read(root/'bags.json').items()):
        record=bag['record'];label='Healthy' if record['y'] else 'Auxiliary_'+record['label'];family=hashlib.sha256(json.dumps(bag['views']).encode()).hexdigest();family_labels.setdefault(family,set()).add(record['y']);aux_rows.append(dict(path=record['path'],label=label,y=record['y'],source='olid_cucumber_aux',crop='Cucumber',group='aux_exact_inputs:'+family,collection='OLID_CUCUMBER_ENTIRE_COLLECTION',validation=False));views=[]
        for key in bag['views']:
            raw=values[lookup[key]].astype(np.float64);views.append(np.concatenate([b/max(float(np.linalg.norm(b)),1e-8) for b in (raw[:384],raw[384:])])/2**.5)
        aux.append(views)
    assert all(len(s)==1 for s in family_labels.values()),'Contradictory auxiliary binary labels for exact model input family';training_rows=[rows[i] for i in tr]+aux_rows;xx=np.concatenate([x,np.array(aux)]);aux_ids=list(range(len(tr),len(xx)));yy=np.array([r['y'] for r in training_rows]);pos={i:j for j,i in enumerate(tr)};full=read(p/'phase193_exhaustive_mobile_replay/protocol.json');parts=[full['target_indices'],full['old_anchor_indices'],full['field_training_indices']];assert sum(parts,[])==tr;roles=[[pos[i] for i in part] for part in parts]+[aux_ids];folds=[]
    for n,f in enumerate(prior['calibration_folds']):folds.append((np.array(f['fit']+(aux_ids if n!=0 else [])),np.array(f['test']+(aux_ids if n==0 else []))))
    methods=['general_auxiliary','nutrition_veto_only'];nutrient_labels={'Nitrogen Deficiency','Pottassium Deficiency','Potassium Deficiency','Magnesium Deficiency'};protocol=dict(source_sha256=digest(__file__),source_feature_complete_sha256=digest(root/'complete.json'),overlap_audit_sha256=digest(root/'overlap_audit.json'),parent_protocol_sha256=digest(parent_path),parents=parents,original_training_indices=tr,validation_indices=val,auxiliary_records=aux_rows,auxiliary_unique_input_families=len(family_labels),calibration_folds=[dict(fit=a.tolist(),test=b.tolist()) for a,b in folds],cv_groups=prior['cv_groups']+[max(prior['cv_groups'])+1]*len(aux_rows),methods=methods,masses=[.375,.1875,.1875,.25],epochs=40,draws_per_epoch=2048,batch=64,learning_rate=.001,weight_decay=.01,ranking_weight=.2,seed=2280,calibration='Grouped training-only cold-start OOF Platt C1. All cucumber images held together in fold0; all original family/input/source93 folds preserved. Final endpoint epoch39, no validation snapshot selection.',nutrition_scope='Nutrition specialist trained only on original Healthy and explicitly named N/K/Mg deficiencies plus author-labelled auxiliary nutrient stress. It can only supply a negative veto, never a new positive Healthy branch. No Other image is relabelled as a named deficiency.',general_scope='Original full clinical replay plus Healthy/stressed cucumber auxiliary supervision; foreign-crop genus and all app guards remain unchanged.',hypothesis='Expert-labelled other-crop nutrient stress adds N/K visual evidence absent from the small tomato-only stress subsets. Test general supervised transfer and a nutrient-specific negative expert, using only new frozen-encoder inputs and matched fixed optimization.',healthy_admission=.95,original_gates_unchanged=True,existing_healthy_and_named_disease_exact=True,acceptance_data_used=False,ad_independent=False,olid_independent=False,stopping='Exactly general and nutrient-specific cold-start40epoch models, each four grouped calibration fits and one full fit. No hyperparameter/threshold/epoch grid. Specialist never supplies positive Healthy. Full qualification only after >=28/40 and zero new falseHealthy under unchanged guards.')
    freeze(out/'protocol.json',protocol);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;features=torch.tensor(xx,dtype=torch.float32,device='cuda');validation=torch.tensor(v,dtype=torch.float32,device='cuda');truth=torch.tensor(yy,dtype=torch.float32,device='cuda');data=load(p);entries,_,packets,before,previous=evaluation_data(p,data);items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({r['path']:(r,packets[r['id']],retained) for r,retained in zip(before,previous)});selection=read(p/'phase118_conditional_recovery/protocol.json');controls=[]
    def predict(model,value):
        model.eval()
        with torch.inference_mode():return model(value.reshape(-1,1408)).reshape(len(value),2).cpu().numpy()
    for method in methods:
        folder=out/method;folder.mkdir(exist_ok=True);selected=np.array([i for i,r in enumerate(training_rows) if method=='general_auxiliary' or r['y'] or r['label'] in nutrient_labels or r['source']=='olid_cucumber_aux']);selected_set=set(selected);weights=np.zeros(len(xx))
        for mass,role in zip(protocol['masses'],roles):
            ids=[i for i in role if i in selected_set];assert ids;weights[ids]=mass*clinical_replay_weights(training_rows,ids)
        weights*=len(selected)
        def fit(ids,name,save_epochs=False):
            dst=folder/name;dst.mkdir(exist_ok=True);torch.manual_seed(2280);model=LeafFeatureHead('mlp',1408).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);sampler=torch.tensor(weights[ids],dtype=torch.float64);history=[]
            if (dst/'complete.json').exists():
                r=read(dst/'complete.json');assert r['protocol_sha256']==digest(out/'protocol.json') and r['model_sha256']==digest(dst/'last.pt');model.load_state_dict(torch.load(dst/'last.pt',weights_only=True,map_location='cuda')['model']);return model
            for epoch in range(40):
                model.train();order=torch.multinomial(sampler,2048,replacement=True,generator=torch.Generator().manual_seed(228000+epoch));losses=[]
                for batch in order.split(64):
                    samples=ids[batch.numpy()];labels=truth[samples];z=model(features[samples].reshape(-1,1408)).reshape(len(samples),2);lo=z.min(1).values;hi=z.max(1).values;raw=torch.where(labels==1,lo,hi);loss=F.binary_cross_entropy_with_logits(raw,labels)+.2*hard_negative_rank_loss(torch.stack([lo,hi],1),labels);assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
                history.append(dict(epoch=epoch,training_loss=float(np.mean(losses))))
                if save_epochs:atomic_save(dict(model={k:z.detach().cpu().clone() for k,z in model.state_dict().items()},epoch=epoch,protocol_sha256=digest(out/'protocol.json')),dst/f'epoch_{epoch:03d}.pt')
            atomic_save(dict(model={k:z.detach().cpu().clone() for k,z in model.state_dict().items()},optimizer=optimizer.state_dict(),rng=capture_rng(),epoch=39,protocol_sha256=digest(out/'protocol.json')),dst/'last.pt');(dst/'complete.json').write_text(json.dumps(dict(protocol_sha256=digest(out/'protocol.json'),model_sha256=digest(dst/'last.pt'),training_indices=ids.tolist(),history=history),indent=2));print(json.dumps(dict(method=method,fit=name,training_loss=history[-1]['training_loss'])),flush=True);return model
        oof=np.full((len(xx),2),np.nan)
        for n,(a,b) in enumerate(folds):
            a=np.array([i for i in a if i in selected_set]);b=np.array([i for i in b if i in selected_set]);assert set(yy[a])=={0,1};oof[b]=predict(fit(a,f'fold_{n}'),features[b])
        assert np.isfinite(oof[selected]).all();cal=LogisticRegression(C=1.,max_iter=2000).fit(oof[selected].reshape(-1,1),np.repeat(yy[selected],2),sample_weight=np.repeat(weights[selected]/2,2));a=float(cal.coef_[0,0]);b=float(cal.intercept_[0]);assert a>0;model=fit(selected,'full',True);raw=predict(model,validation);prob=expit(a*raw+b).min(1);healthy=selection['fixed_healthy_validation_correct'];old_h=0;errors={}
        for i,score in zip(val,prob):
            row=rows[i];original,packet,retained=items[row['path']];now=recover_only(original,retained,packet,float(score),data['manifest'])['app_replay']['kind']=='HEALTHY'
            if not row['y']:errors[row['source']]=errors.get(row['source'],0)+int(now)
            elif row['source']=='olid_tomato':healthy+=int(now)
            else:old_h+=int(now)
        restored=LeafFeatureHead('mlp',1408).cuda();restored.load_state_dict(torch.load(folder/'full/last.pt',weights_only=True,map_location='cuda')['model']);replay=predict(restored,validation);error=float(abs(raw-replay).max());assert error<1e-6 and np.array_equal(expit(a*replay+b).min(1)>=.95,prob>=.95);np.savez(folder/'predictions.npz',indices=np.array(val),probability=prob,validation_margin=raw,oof_training_margin=oof,selected_training_positions=selected,calibration=np.array([a,b]),model_sha256=digest(folder/'full/last.pt'),protocol_sha256=digest(out/'protocol.json'));r=dict(method=method,healthy_validation_correct=healthy,new_false_healthy_by_source=errors,old_healthy_correct=old_h,calibration=[a,b],model_sha256=digest(folder/'full/last.pt'),restored_head_max_error=error,diagnostic_standalone_verdicts_only=method=='nutrition_veto_only',cheap_pass=method=='general_auxiliary' and healthy>=28 and sum(errors.values())==0);controls.append(r);print(json.dumps(r),flush=True)
        if r['cheap_pass']:freeze(folder/'candidate.json',dict(**r,full_qualification_required=True,model_promoted=False))
    (out/'result.json').write_text(json.dumps(dict(source_sha256=digest(__file__),protocol_sha256=digest(out/'protocol.json'),controls=controls,elapsed_seconds=time.monotonic()-started,model_promoted=False,acceptance_data_used=False,ad_independent=False,olid_independent=False),indent=2))

if __name__=='__main__':main()
