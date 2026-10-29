"""Matched binary and clinical auxiliary heads on all quarantined whole photos."""
from pathlib import Path
import argparse,json,time
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from clinical_replay_balance import clinical_replay_weights
from conditional_recovery_policy import recover_only
from opened_context_data import load,read
from permissive_encoder import digest
from phase85_population_sft import evaluation_data
from population_siglip_tail import freeze
from training_support import atomic_save,capture_rng,restore_rng


class WholeClinicalHead(nn.Module):
    def __init__(self,dimension,classes):
        super().__init__();self.hidden=nn.Sequential(nn.Linear(dimension,128),nn.SiLU());self.binary=nn.Linear(128,1);self.clinical=nn.Linear(128,classes)
    def forward(self,x):
        h=self.hidden(x);return self.binary(h).ravel(),self.clinical(h)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',required=True);p=Path(ap.parse_args().campaign);out=p/'phase245_all_photo_nonlinear_supervision';out.mkdir(exist_ok=True);started=time.monotonic()
    if (out/'result.json').exists():assert read(out/'result.json')['source_sha256']==digest(__file__);return
    parent=p/'phase243_all_photo_context_supervision';cp=read(parent/'protocol.json');assert read(parent/'result.json')['protocol_sha256']==digest(parent/'protocol.json')
    rows=read(p/'phase84_population_head/fitting_rows.json');train=cp['train_indices'];val=cp['validation_indices'];whole={}
    for name,sha in cp['parents'].items():
        root=p/name;assert digest(root/'complete.json')==sha;cc=read(root/'complete.json');assert digest(root/'features.npy')==cc['features_sha256'] and digest(root/'inputs.json')==cc['inputs_sha256']
        values=np.load(root/'features.npy',mmap_mode='r');whole.update({r['path']:v for r,v in zip(read(root/'inputs.json'),values)})
    x=np.asarray([whole[r['path']] if r['path'] in whole else np.zeros(768) for r in rows],dtype=np.float32);x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)
    labels=sorted({rows[i]['label'] for i in train});fine=np.array([labels.index(r['label']) if r['label'] in labels else -1 for r in rows]);y=np.array([r['y'] for r in rows]);methods=['binary','clinical_auxiliary']
    folds=[([train[j] for j in f['fit']],[train[j] for j in f['test']]) for f in cp['calibration_folds']]
    protocol=dict(source_sha256=digest(__file__),parent_protocol_sha256=digest(parent/'protocol.json'),methods=methods,clinical_labels=labels,train_indices=train,validation_indices=val,
        calibration_folds=[dict(fit=a,test=b) for a,b in folds],epochs=80,draws_per_epoch=1024,batch_images=64,learning_rate=.001,weight_decay=.01,clinical_loss_weight=.3,
        calibration='Cold grouped training-only OOF Platt C1; exact same all-photo supervision/quarantine as phase243. Per-fold clinical/source/family weights. Fixed endpoint, no validation epoch selection.',
        hypothesis='Whole-photo nonlinear/clinical boundaries may benefit from legitimate no-ROI supervision absent from previous ROI learners; compare matched binary and clinical auxiliary heads without another encoder call.',
        healthy_admission=.95,new_encoder_inputs=0,original_gates_unchanged=True,existing_healthy_and_named_disease_exact=True,ad_independent=False,olid_independent=False,acceptance_data_used=False,
        stopping='Exactly2heads,4cold folds plus full fit each,80epochs. Preserve every10th full-fit snapshot and final optimizer/RNG for each fit. No rate/threshold/snapshot grid. Full qualification only after28/40 Healthy and zero new falseHealthy; no export on failure.')
    freeze(out/'protocol.json',protocol);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;x=torch.tensor(x,device='cuda');truth=torch.tensor(y,dtype=torch.float32,device='cuda');clinical=torch.tensor(fine,device='cuda');controls=[]
    data=load(p);entries,_,packets,before,previous=evaluation_data(p,data);items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({r['path']:(r,packets[r['id']],old) for r,old in zip(before,previous)})
    for method in methods:
        folder=out/method;folder.mkdir(exist_ok=True)
        def fit(ids,name):
            dst=folder/name;dst.mkdir(exist_ok=True);torch.manual_seed(2450);model=WholeClinicalHead(768,len(labels)).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);start=0
            if (dst/'last.pt').exists():
                saved=torch.load(dst/'last.pt',weights_only=True,map_location='cpu');assert saved['protocol_sha256']==digest(out/'protocol.json');model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng']);start=saved['epoch']+1
            sampling=torch.tensor(clinical_replay_weights(rows,ids));ids=np.asarray(ids)
            for epoch in range(start,80):
                model.train();order=torch.multinomial(sampling,1024,replacement=True,generator=torch.Generator().manual_seed(245000+epoch))
                for batch in order.split(64):
                    selected=ids[batch.numpy()];z,f=model(x[selected]);loss=F.binary_cross_entropy_with_logits(z,truth[selected])
                    if method=='clinical_auxiliary':loss=loss+.3*F.cross_entropy(f,clinical[selected])
                    assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                atomic_save(dict(model=state,optimizer=optimizer.state_dict(),rng=capture_rng(),epoch=epoch,protocol_sha256=digest(out/'protocol.json')),dst/'last.pt')
                if name=='full' and (epoch+1)%10==0:atomic_save(dict(model=state,epoch=epoch,protocol_sha256=digest(out/'protocol.json')),dst/f'epoch_{epoch:03d}.pt')
            return model.eval()
        oof=np.full(len(rows),np.nan)
        with torch.no_grad():pass
        for n,(a,b) in enumerate(folds):
            model=fit(a,f'fold_{n}')
            with torch.inference_mode():oof[b]=model(x[b])[0].cpu().numpy()
            print(json.dumps(dict(method=method,completed_fold=n)),flush=True)
        weights=clinical_replay_weights(rows,train)*len(train);cal=LogisticRegression(C=1.,max_iter=2000).fit(oof[train,None],y[train],sample_weight=weights);a=float(cal.coef_[0,0]);b=float(cal.intercept_[0]);model=fit(train,'full')
        with torch.inference_mode():raw=model(x[val])[0].cpu().numpy().astype(np.float64)
        restored=WholeClinicalHead(768,len(labels)).cuda().eval();restored.load_state_dict(torch.load(folder/'full/last.pt',weights_only=True,map_location='cpu')['model'])
        with torch.inference_mode():assert np.array_equal(raw,restored(x[val])[0].cpu().numpy().astype(np.float64))
        probability=expit(a*raw+b) if a>0 else np.zeros(len(val));healthy=13;errors={};old_h=0
        for i,score in zip(val,probability):
            r=rows[i];original,packet,retained=items[r['path']];now=recover_only(original,retained,packet,float(score),data['manifest'])['app_replay']['kind']=='HEALTHY'
            if not r['y']:errors[r['source']]=errors.get(r['source'],0)+int(now)
            elif r['source']=='olid_tomato':healthy+=int(now)
            else:old_h+=int(now)
        model_sha=digest(folder/'full/last.pt');np.savez(folder/'predictions.npz',indices=val,probability=probability,validation_margin=raw,oof_indices=train,oof_training_margin=oof[train],calibration=np.array([a,b]),model_sha256=model_sha)
        control=dict(method=method,healthy_validation_correct=healthy,new_false_healthy_by_source=errors,old_healthy_correct=old_h,calibration_slope=a,cheap_pass=healthy>=28 and sum(errors.values())==0,model_sha256=model_sha);controls.append(control);print(json.dumps(control),flush=True)
    result=dict(source_sha256=digest(__file__),protocol_sha256=digest(out/'protocol.json'),controls=controls,elapsed_seconds=time.monotonic()-started,new_encoder_inputs=0,model_promoted=False,acceptance_data_used=False);(out/'result.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
