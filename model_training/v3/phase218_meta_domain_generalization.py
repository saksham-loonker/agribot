"""Matched ERM and differentiable held-training-source episodes over cached ROIs."""
from pathlib import Path
import argparse,collections,json,shutil,time
import numpy as np
import torch
from torch.nn import functional as F
from torch.func import functional_call
from clinical_replay_balance import clinical_replay_weights
from conditional_recovery_policy import recover_only
from exhaustive_replay_data import exhaustive_replay_data
from leaf_feature_expert import LeafFeatureHead
from meta_domain_loss import meta_domain_loss
from opened_context_data import load,read
from phase85_population_sft import evaluation_data
from permissive_encoder import digest
from population_siglip_tail import freeze
from training_support import atomic_save,capture_rng,restore_rng

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True);p=Path(parser.parse_args().campaign);out=Path('/dev/shm/agribot-phase218-meta-domain');out.mkdir(exist_ok=True);link=p/'phase218_meta_domain_generalization'
    if not link.exists():link.symlink_to(out,target_is_directory=True)
    assert link.resolve()==out and shutil.disk_usage(out).free>96<<20
    if (out/'result.json').exists():assert read(out/'result.json')['source_sha256']==digest(__file__);return
    started=time.monotonic();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    values,identities,rows,tables,train,val,meta=exhaustive_replay_data(p);parent=list(range(len(train)));owner={}
    def find(j):
        while parent[j]!=j:parent[j]=parent[parent[j]];j=parent[j]
        return j
    for j,i in enumerate(train):
        keys=[('family',rows[i]['group'])]+[('input',identities[key]) for pair in tables[i] for view in pair for key in view]
        for key in keys:
            if key in owner:parent[find(j)]=find(owner[key])
            else:owner[key]=j
    family={i:find(j) for j,i in enumerate(train)};sources=sorted({rows[i]['source'] for i in train if {rows[j]['y'] for j in train if rows[j]['source']==rows[i]['source']}=={0,1}});pools={}
    for source in sources:
        query=[i for i in train if rows[i]['source']==source];quarantine={family[i] for i in query};support=[i for i in train if rows[i]['source']!=source and family[i] not in quarantine]
        assert set(rows[i]['y'] for i in support)=={0,1} and not {family[i] for i in support}&{family[i] for i in query};pools[source]=(np.array(support),np.array(query))
    assert len(sources)>=3
    warm=p/'phase88_diagnostic_bag_head/mlp';wr=read(warm/'result.json');assert digest(warm/'best.pt')==wr['checkpoint_sha256'];calibration=wr['calibration']
    protocol=dict(source_sha256=digest(__file__),meta_loss_source_sha256=digest(Path(__file__).with_name('meta_domain_loss.py')),data_source_sha256=digest(Path(__file__).with_name('exhaustive_replay_data.py')),full_replay_protocol_sha256=meta['full_replay_protocol_sha256'],original_parents=meta['original_parents'],warm_checkpoint_sha256=digest(warm/'best.pt'),train_indices=train,validation_indices=val,source_pools={s:dict(support=a.tolist(),query=b.tolist()) for s,(a,b) in pools.items()},methods=['episodic_erm','mldg'],inner_learning_rate=.05,objective='Half support BCE plus half held-training-source BCE after one differentiable support update; matched ERM uses same episodes without that update',episode_sampling='Uniform source schedule; 32 support and32 query photos, clinical/binary/source/family balanced within each pool; entire held-source family/input components excluded from support',steps_per_epoch=32,maximum_epochs=40,minimum_epochs=20,training_ema_patience=12,minimum_ema_improvement=.001,outer_optimizer='AdamW lr .001 weight_decay .01, gradient norm1',seed=2218,calibration=calibration,healthy_admission=.95,new_encoder_inputs=0,original_gates_unchanged=True,existing_healthy_and_named_disease_exact=True,ad_independent=False,olid_independent=False,acceptance_data_used=False,output_volatile=True,hypothesis='Lower ordinary training loss has not generalized to field Healthy/deficiency cases. Explicitly simulate source shift inside training, aligning a training-domain update with different training-source supervision. This differs from earlier source-adversarial or worst-group reweighting and uses no evaluation labels in episodes.',stopping='Exactly matched ERM and full differentiable one-step meta-domain objectives; at most forty epochs, training-EMA stopping only. No inner-step/outer-rate grid or full encoder meta-training without a passing cheap actual-verdict screen.')
    freeze(out/'protocol.json',protocol);features=torch.from_numpy(values).cuda();index=np.zeros((len(rows),2,6),np.int64);mask=np.zeros(index.shape,np.float32)
    for i in train+val:
        assert len(tables[i][0])==2
        for view,keys in enumerate(tables[i][0]):index[i,view,:len(keys)]=keys;mask[i,view,:len(keys)]=1.
    index=torch.from_numpy(index).cuda();mask=torch.from_numpy(mask).cuda();valid=mask.sum(-1)>0;truth=torch.tensor([r['y'] for r in rows],dtype=torch.float32,device='cuda');pool_weights={s:tuple(torch.from_numpy(clinical_replay_weights(rows,ids)) for ids in pair) for s,pair in pools.items()};data=load(p);entries,_,packets,before,previous=evaluation_data(p,data);items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({r['path']:(r,packets[r['id']],retained) for r,retained in zip(before,previous)});selection=read(p/'phase118_conditional_recovery/protocol.json');controls=[]
    for method in protocol['methods']:
        folder=out/method;folder.mkdir(exist_ok=True);torch.manual_seed(2218);model=LeafFeatureHead('mlp',1408).cuda();model.load_state_dict(torch.load(warm/'best.pt',weights_only=True,map_location='cpu')['model']);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);history=[];best=None;start=0
        def margins(ids,params=None):
            x=features[index[ids]].reshape(-1,1408);raw=(model(x) if params is None else functional_call(model,params,(x,))).reshape(len(ids),2,6)*calibration[0]+calibration[1];means=(raw*mask[ids]).sum(-1)/mask[ids].sum(-1).clamp_min(1)
            return means.masked_fill(~valid[ids],float('inf')).min(1).values,means.masked_fill(~valid[ids],-float('inf')).max(1).values
        def loss(ids,params):
            positive,negative=margins(ids,params);return F.binary_cross_entropy_with_logits(torch.where(truth[ids]==1,positive,negative),truth[ids])
        if (folder/'last.pt').exists():
            saved=torch.load(folder/'last.pt',weights_only=True,map_location='cpu');assert saved['protocol_sha256']==digest(out/'protocol.json');model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng']);history,best,start=saved['history'],saved['best'],saved['epoch']+1
        ema=history[-1]['training_loss_ema'] if history else None;training_best=min((h['training_loss_ema'] for h in history),default=float('inf'));stale=history[-1]['training_stale'] if history else 0
        for epoch in range(start,40):
            if epoch>=20 and stale>=12:break
            generator=torch.Generator().manual_seed(221800+epoch);schedule=torch.randperm(len(sources),generator=generator).tolist();losses=[];support_losses=[];query_losses=[];model.train()
            for step in range(32):
                source=sources[schedule[step%len(schedule)]];support,query=pools[source];wa,wb=pool_weights[source];a=support[torch.multinomial(wa,32,replacement=True,generator=generator).numpy()];b=query[torch.multinomial(wb,32,replacement=True,generator=generator).numpy()]
                if method=='mldg':objective,la,lb=meta_domain_loss(model,lambda params:loss(a,params),lambda params:loss(b,params),.05)
                else:la=loss(a,None);lb=loss(b,None);objective=.5*(la+lb)
                assert torch.isfinite(objective);losses.append(float(objective.detach()));support_losses.append(float(la.detach()));query_losses.append(float(lb.detach()));optimizer.zero_grad(set_to_none=True);objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            model.eval()
            with torch.inference_mode():positive,negative=margins(np.array(val));probability=positive.sigmoid().cpu().numpy();validation_loss=float(F.binary_cross_entropy_with_logits(torch.where(truth[val]==1,positive,negative),truth[val]))
            healthy=selection['fixed_healthy_validation_correct'];errors={};old_h=0
            for i,score in zip(val,probability):
                row=rows[i];original,packet,retained=items[row['path']];now=recover_only(original,retained,packet,float(score),data['manifest'])['app_replay']['kind']=='HEALTHY'
                if not row['y']:errors[row['source']]=errors.get(row['source'],0)+int(now)
                elif row['source']=='olid_tomato':healthy+=int(now)
                else:old_h+=int(now)
            score=(-sum(errors.values()),healthy,old_h,-validation_loss);state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if best is None or score>tuple(best):
                best=score;atomic_save(dict(model=state,epoch=epoch,protocol_sha256=digest(out/'protocol.json')),folder/'best.pt');np.savez(folder/'best_validation_predictions.npz',indices=np.array(val),probability=probability,checkpoint_sha256=digest(folder/'best.pt'),protocol_sha256=digest(out/'protocol.json'))
            checkpoint=folder/f'epoch_{epoch:03d}.pt';atomic_save(dict(model=state,epoch=epoch,protocol_sha256=digest(out/'protocol.json')),checkpoint);np.savez(folder/f'epoch_{epoch:03d}_predictions.npz',indices=np.array(val),probability=probability,checkpoint_sha256=digest(checkpoint),protocol_sha256=digest(out/'protocol.json'));train_loss=float(np.mean(losses));ema=train_loss if ema is None else .9*ema+.1*train_loss
            if ema<training_best-.001:training_best=ema;stale=0
            else:stale+=1
            history.append(dict(epoch=epoch,decision=score,new_false_healthy_by_source=errors,training_loss=train_loss,training_loss_ema=ema,training_stale=stale,support_loss=float(np.mean(support_losses)),query_loss=float(np.mean(query_losses)),checkpoint_sha256=digest(checkpoint)));print(json.dumps(dict(method=method,**history[-1])),flush=True);atomic_save(dict(model=state,optimizer=optimizer.state_dict(),rng=capture_rng(),epoch=epoch,history=history,best=best,protocol_sha256=digest(out/'protocol.json')),folder/'last.pt')
            if best[0]==0 and best[1]>=28:break
        result=dict(method=method,best_decision=best,history=history,cheap_pass=best[0]==0 and best[1]>=28,checkpoint_sha256=digest(folder/'best.pt'),protocol_sha256=digest(out/'protocol.json'),model_promoted=False);(folder/'result.json').write_text(json.dumps(result,indent=2));controls.append(result)
    result=dict(source_sha256=digest(__file__),protocol_sha256=digest(out/'protocol.json'),controls=controls,new_encoder_inputs=0,model_promoted=False,acceptance_data_used=False,output_volatile=True,elapsed_seconds=time.monotonic()-started);(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(controls=[{k:r[k] for k in ('method','best_decision','cheap_pass')} for r in controls])),flush=True)

if __name__=='__main__':main()
