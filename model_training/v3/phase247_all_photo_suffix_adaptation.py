"""Matched binary and clinical auxiliary heads on all quarantined whole photos."""
from pathlib import Path
import argparse,json,time
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
import timm
from sklearn.model_selection import StratifiedGroupKFold
from partial_primary_siglip import PartialPrimarySiglip,prefix_identity
from permissive_encoder import CanonicalPermissiveEncoder
from large_photometric_dino import trainable_state,restore_trainable
from clinical_replay_balance import clinical_replay_weights
from conditional_recovery_policy import recover_only
from opened_context_data import load,read
from permissive_encoder import digest
from phase85_population_sft import evaluation_data
from population_siglip_tail import freeze
from training_support import atomic_save,capture_rng,restore_rng


class WholeSuffixHead(PartialPrimarySiglip):
    def __init__(self,encoder,mode):
        super().__init__(encoder)
        if mode=='frozen':self.encoder.requires_grad_(False)
    def forward(self,tokens):return super().forward(tokens),None


class TokenRows:
    def __init__(self,tokens,row_index):self.tokens=tokens;self.row_index=row_index
    def __getitem__(self,indices):return self.tokens[self.row_index[indices]]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',required=True);p=Path(ap.parse_args().campaign);out=p/'phase247_all_photo_suffix_adaptation';out.mkdir(exist_ok=True);started=time.monotonic()
    if (out/'result.json').exists():assert read(out/'result.json')['source_sha256']==digest(__file__);return
    parent=p/'phase246_whole_prefix_coverage';cp=read(parent/'protocol.json');cc=read(parent/'complete.json');assert cc['protocol_sha256']==digest(parent/'protocol.json') and cc['tokens_sha256']==digest(cp['storage'])
    assert cc['locations_sha256']==digest(parent/'locations.json') and cc['path_indices_sha256']==digest(parent/'path_indices.json')
    rows=read(p/'phase84_population_head/fitting_rows.json');train=cp['train_indices'];val=cp['validation_indices'];labels=sorted({rows[i]['label'] for i in train});fine=np.array([labels.index(r['label']) if r['label'] in labels else -1 for r in rows]);y=np.array([r['y'] for r in rows]);methods=['frozen','full_suffix']
    locations=read(parent/'locations.json');path_index=read(parent/'path_indices.json');parent_group=list(range(len(train)));owner={}
    def find(j):
        while parent_group[j]!=j:parent_group[j]=parent_group[parent_group[j]];j=parent_group[j]
        return j
    for j,i in enumerate(train):
        for key in [('family',rows[i]['group']),('input',path_index[rows[i]['path']])]:
            if key in owner:parent_group[find(j)]=find(owner[key])
            else:owner[key]=j
    groups=np.array([find(j) for j in range(len(train))]);splits=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=2470).split(np.zeros(len(train)),y[train],groups)
    folds=[]
    for a,b in splits:
        assert not set(groups[a])&set(groups[b]);folds.append(([train[j] for j in a],[train[j] for j in b]))
    protocol=dict(source_sha256=digest(__file__),parent_protocol_sha256=digest(parent/'protocol.json'),methods=methods,clinical_labels=labels,train_indices=train,validation_indices=val,
        calibration_folds=[dict(fit=a,test=b) for a,b in folds],epochs=40,draws_per_epoch=1024,batch_images=16,head_learning_rate=.001,suffix_learning_rate=.00001,weight_decay=.01,clinical_loss_weight=0.,
        calibration='Cold grouped training-only OOF Platt C1; exact same all-photo supervision/quarantine as phase243. Per-fold clinical/source/family weights. Fixed endpoint, no validation epoch selection.',
        hypothesis='Adapt copied whole-SigLIP suffixes using the previously omitted no-ROI target supervision after all-photo fixed embeddings failed. Compare a matched frozen head with copied last2blocks/norm/pool, retaining exact original prefix and all gates. Old replay is explicitly bounded to already cached whole prefixes.',
        healthy_admission=.95,new_encoder_inputs=0,original_gates_unchanged=True,existing_healthy_and_named_disease_exact=True,ad_independent=False,olid_independent=False,acceptance_data_used=False,
        stopping='Exactly frozen/full_suffix,4cold folds plus full fit each,40epochs. Preserve every10th full-fit snapshot and final optimizer/RNG. No rate/threshold/snapshot grid;28/40 Healthy and zero new falseHealthy before full qualification. No export on failure.')
    freeze(out/'protocol.json',protocol);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;timm.layers.set_fused_attn(False)
    source=p/'phase23_siglip/siglip2_feature_heads/encoder.pt';assert digest(source)==cp['encoder_sha256'];state=torch.load(source,weights_only=True,map_location='cpu');encoder=CanonicalPermissiveEncoder(state['spec']);encoder.load_state_dict(state['model']);assert prefix_identity(encoder)==cp['prefix_state_sha256'];encoder.eval();del state
    banks={name:np.load(name,mmap_mode='r') for name in {r['file'] for r in locations}};tokens=torch.empty((len(locations),256,768),device='cuda')
    for begin in range(0,len(locations),32):tokens[begin:begin+32]=torch.from_numpy(np.stack([banks[r['file']][r['index']] for r in locations[begin:begin+32]])).cuda()
    indices=torch.tensor([path_index.get(r['path'],0) for r in rows],device='cuda');x=TokenRows(tokens,indices);truth=torch.tensor(y,dtype=torch.float32,device='cuda');controls=[]
    data=load(p);entries,_,packets,before,previous=evaluation_data(p,data);items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({r['path']:(r,packets[r['id']],old) for r,old in zip(before,previous)})
    for method in methods:
        folder=out/method;folder.mkdir(exist_ok=True)
        def fit(ids,name):
            dst=folder/name;dst.mkdir(exist_ok=True);torch.manual_seed(2450);model=WholeSuffixHead(encoder,method).cuda();optimizer=torch.optim.AdamW([dict(params=list(model.head.parameters()),lr=.001)]+([dict(params=[v for k,v in model.named_parameters() if v.requires_grad and not k.startswith('head.')],lr=.00001)] if method=='full_suffix' else []),weight_decay=.01);start=0
            if (dst/'last.pt').exists():
                saved=torch.load(dst/'last.pt',weights_only=True,map_location='cpu');assert saved['protocol_sha256']==digest(out/'protocol.json');restore_trainable(model,saved['model']);optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng']);start=saved['epoch']+1
            sampling=torch.tensor(clinical_replay_weights(rows,ids));ids=np.asarray(ids)
            for epoch in range(start,40):
                model.train();order=torch.multinomial(sampling,1024,replacement=True,generator=torch.Generator().manual_seed(245000+epoch))
                for batch in order.split(16):
                    selected=ids[batch.numpy()];z,f=model(x[selected]);loss=F.binary_cross_entropy_with_logits(z,truth[selected])
                    if method=='clinical_auxiliary':loss=loss+.3*F.cross_entropy(f,clinical[selected])
                    assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                state=trainable_state(model)
                atomic_save(dict(model=state,optimizer=optimizer.state_dict(),rng=capture_rng(),epoch=epoch,protocol_sha256=digest(out/'protocol.json')),dst/'last.pt')
                if name=='full' and (epoch+1)%10==0:atomic_save(dict(model=state,epoch=epoch,protocol_sha256=digest(out/'protocol.json')),dst/f'epoch_{epoch:03d}.pt')
            assert prefix_identity(model.encoder)==cp['prefix_state_sha256'];return model.eval()
        oof=np.full(len(rows),np.nan)
        for n,(a,b) in enumerate(folds):
            model=fit(a,f'fold_{n}')
            with torch.inference_mode():oof[b]=model(x[b])[0].cpu().numpy()
            print(json.dumps(dict(method=method,completed_fold=n)),flush=True)
        weights=clinical_replay_weights(rows,train)*len(train);cal=LogisticRegression(C=1.,max_iter=2000).fit(oof[train,None],y[train],sample_weight=weights);a=float(cal.coef_[0,0]);b=float(cal.intercept_[0]);model=fit(train,'full')
        with torch.inference_mode():raw=model(x[val])[0].cpu().numpy().astype(np.float64)
        restored=WholeSuffixHead(encoder,method).cuda().eval();restore_trainable(restored,torch.load(folder/'full/last.pt',weights_only=True,map_location='cpu')['model'])
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
