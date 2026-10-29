"""Image-level disease witnesses and universal Healthy supervision on cached ROIs.

Additional audited field supervision; no encoder construction, pixel decoding or threshold
search. Continue the completed head warm-up, not its adapted encoder weights.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from torch.nn import functional as F
from image_bag_supervision import image_bag_losses
from joint_clinical_head import JointClinicalHead
from conditional_recovery_policy import recover_only
from hard_negative_ranking import hard_negative_rank_loss
from opened_context_data import load,read
from permissive_encoder import digest
from phase85_population_sft import evaluation_data
from phase114_new_supervision_features import original_features
from population_siglip_tail import freeze
from source_balance import source_class_group_weights
from training_support import atomic_save,capture_rng,restore_rng


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);out=p/'phase166_new_field_bag_teacher';out.mkdir(exist_ok=True);started=time.monotonic()
    if (out/'result.json').exists():assert read(out/'result.json')['source_sha256']==digest(__file__);return
    torch.set_num_threads(4);torch.manual_seed(2159);torch.backends.cuda.matmul.allow_tf32=False
    parent=p/'phase152_balanced_photo_teacher';pp=read(parent/'protocol.json');warm=parent/'balanced_photo_clinical_teacher/head_warm_last.pt'
    saved=torch.load(warm,weights_only=True,map_location='cpu');assert saved['protocol_sha256']==digest(parent/'protocol.json')
    lookup={};features=[];contracts={}
    for name in ('phase131_large_roi_cache','phase147_large_teacher_features'):
        root=p/name;protocol=read(root/'protocol.json');complete=read(root/'complete.json')
        assert complete['protocol_sha256']==digest(root/'protocol.json') and complete['features_sha256']==digest(root/'features.npy')
        assert protocol['encoder_sha256']==pp['dino_metadata']['encoder_sha256']
        values=np.load(root/'features.npy');assert values.shape==(len(protocol['input_sha256']),1024)
        for sha,value in zip(protocol['input_sha256'],values):
            if sha in lookup:assert np.array_equal(features[lookup[sha]],value)
            else:lookup[sha]=len(features);features.append(value)
        contracts[name]=digest(root/'complete.json')
    cnn=p/'phase154_convnext_features';cp=read(cnn/'protocol.json');cc=read(cnn/'complete.json')
    assert cc['protocol_sha256']==digest(cnn/'protocol.json') and cc['features_sha256']==digest(cnn/'features.npy')
    cnn_features=np.load(cnn/'features.npy');assert cnn_features.shape==(len(cp['input_sha256']),768)
    cnn_lookup={sha:i for i,sha in enumerate(cp['input_sha256'])};assert len(cnn_lookup)==len(cnn_features)
    common=[sha for sha in lookup if sha in cnn_lookup]
    combined=np.stack([np.concatenate((features[lookup[sha]],cnn_features[cnn_lookup[sha]])) for sha in common]);lookup={sha:i for i,sha in enumerate(common)}
    field=p/'phase164_new_field_features';fc=read(field/'complete.json');assert fc['protocol_sha256']==digest(field/'protocol.json')
    for filename,key in [('features.npy','features_sha256'),('input_sha256.json','input_sha256_sha256'),('bags.json','bags_sha256')]:assert digest(field/filename)==fc[key]
    field_keys=read(field/'input_sha256.json');field_values=np.load(field/'features.npy');assert field_values.shape==(len(field_keys),3200)
    parts=list(combined)
    for sha,row in zip(field_keys,field_values):
        joint=row[1408:]
        if sha in lookup:assert np.array_equal(parts[lookup[sha]],joint)
        else:lookup[sha]=len(parts);parts.append(joint)
    features=torch.from_numpy(np.stack(parts)).cuda();contracts['phase154_convnext_features']=digest(cnn/'complete.json');contracts['phase164_new_field_features']=digest(field/'complete.json')
    _,bags,descriptors,_=original_features(p)
    rows=read(p/'phase84_population_head/fitting_rows.json');extra=p/'phase114_new_supervision_features'
    identities=[r['input_sha256'] for r in descriptors]+read(extra/'new_input_sha256.json')
    train=np.array(pp['train_indices']);val=np.array(pp['validation_indices']);target=pp['target_indices'];anchors=pp['old_anchor_indices']
    tables={i:[[bags[rows[i]['path']][view] for view in ('center','full')]] for i in target+anchors+val.tolist()};new=[]
    for path,bag in read(extra/'bags.json').items():
        if not bag['actual_leaves']:continue
        i=len(rows);new.append(i);tables[i]=bag['views'];record=bag['source_record']
        rows.append(dict(path=path,y=int(record['label']=='Healthy'),source='new107_development',group='new107:entire_publisher_collection',validation=False,label=record['label']))
    assert train.tolist()==target+anchors+new and len(new)==pp['new_source_training_images']
    field_indices=[];field_offset=len(identities);identities+=field_keys
    for path,bag in read(field/'bags.json').items():
        if not bag['actual_leaves']:continue
        i=len(rows);field_indices.append(i);tables[i]=[[[field_offset+key for key in view] for view in pair] for pair in bag['views']];record=bag['source_record']
        rows.append(dict(path=path,y=int(record['label']=='Healthy'),source='field93',group=record['group'],validation=False,label=record['label']))
    train=np.array(target+anchors+new+field_indices);assert len(field_indices)>100 and sum(rows[i]['y'] for i in field_indices)>30
    keys=lambda ids:{identities[key] for i in ids for pair in tables[i] for view in pair for key in view}
    assert not keys(train)&keys(val) and keys(train)|keys(val)<=set(lookup)
    data=load(p);entries,records,packets,before,previous=evaluation_data(p,data)
    items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({b['path']:(b,packets[b['id']],r) for b,r in zip(before,previous)})
    labels=data['manifest']['classifier']['labels'];assert labels[1]=='Healthy'
    truth=torch.tensor([r['y'] for r in rows],dtype=torch.float32,device='cuda');fine=torch.tensor([labels.index(r['label']) if r['label'] in labels else -1 for r in rows],device='cuda')
    index=np.zeros((len(rows),8,6),np.int64);mask=np.zeros(index.shape,np.float32)
    for i,pairs in tables.items():
        for rotation,pair in enumerate(pairs):
            for view,keys_ in enumerate(pair):
                ids=[lookup[identities[key]] for key in keys_];index[i,rotation*2+view,:len(ids)]=ids;mask[i,rotation*2+view,:len(ids)]=1
    index=torch.from_numpy(index).cuda();mask=torch.from_numpy(mask).cuda();active=mask.sum(-1)>0
    weights=torch.from_numpy(np.concatenate([.5*source_class_group_weights(rows,target),.125*source_class_group_weights(rows,anchors),.125*source_class_group_weights(rows,new),.25*source_class_group_weights(rows,field_indices)]))
    model=JointClinicalHead().cuda();model.load_warm_head(saved['head']);selection=read(p/'phase118_conditional_recovery/protocol.json')
    protocol=dict(source_sha256=digest(__file__),parent_protocol_sha256=digest(parent/'protocol.json'),warm_head_sha256=digest(warm),feature_contracts=contracts,field_training_indices=field_indices,field_training_images=len(field_indices),field_healthy_photos=sum(rows[i]['y'] for i in field_indices),training_mass=dict(target=.5,old_anchor=.125,old107=.125,field93=.25),new_field_source_role='Entire publisher collection development-only, no independent remainder',
        train_indices=train.tolist(),validation_indices=val.tolist(),clinical_loss_sha256=digest(Path(__file__).with_name('image_bag_supervision.py')),supervision='Photo-level named/coarse disease requires one matching ROI witness per complete rotation; Healthy supervises every actual ROI; missing/padded rotations excluded; strict minimum over all real ROI margins at admission',training_healthy_photos=sum(rows[i]['y']==1 for i in train),training_disease_photos=sum(rows[i]['y']==0 for i in train),training_multi_roi_photos=sum(any(len(view)>1 for pair in tables[i] for view in pair) for i in train),ranking_source_sha256=digest(Path(__file__).with_name('hard_negative_ranking.py')),
        joint_head_source_sha256=digest(Path(__file__).with_name('joint_clinical_head.py')),feature_dimensions=[1024,768],clinical_loss_weight=1.,binary_loss_weight=.3,ranking_loss_weight=.2,ranking_margin=1.,maximum_hard_negatives=4,
        maximum_epochs=30,minimum_epochs=6,patience=6,draws_per_epoch=1024,batch_images=64,learning_rate=.0003,seed=2159,
        existing_verdicts_preserved=True,new_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False,
        hypothesis='Add screened, publisher-verified field supervision to the completed image-level witness objective. Keep the same frozen representation, clinical class semantics, optimization and strict all-ROI admission; preserve every existing guard and named verdict. The source is development-only and its remaining photos are never independent acceptance.',
        stopping='One image-bag witness-supervision control; stop after six stale actual-verdict epochs, minimum six and maximum thirty. >=28/40 Healthy and zero new false Healthy required before complete unchanged gates; no model export from this probe.')
    freeze(out/'protocol.json',protocol)
    def forward(ids):
        logits=model(features[index[ids]]).reshape(len(ids),4,2,6,9)
        clinical,binary,admission=image_bag_losses(logits,fine[ids],mask[ids].reshape(len(ids),4,2,6)>0,1)
        return admission,admission,clinical+.3*binary
    def decision(probability,loss):
        errors={};healthy=selection['fixed_healthy_validation_correct'];old_h=0
        for i,score in zip(val,probability):
            row=rows[i];original,packet,retained=items[row['path']];now=recover_only(original,retained,packet,float(score),data['manifest'])['app_replay']['kind']=='HEALTHY'
            if not row['y']:errors[row['source']]=errors.get(row['source'],0)+int(now)
            elif row['source']=='olid_tomato':healthy+=int(now)
            else:old_h+=int(now)
        return (-sum(errors.values()),healthy,old_h,-loss),errors
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01);history=[];best=None;stale=0;start=0
    if (out/'last.pt').exists():
        saved=torch.load(out/'last.pt',weights_only=True,map_location='cpu');assert saved['protocol_sha256']==digest(out/'protocol.json')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng']);history,best,stale,start=saved['history'],saved['best'],saved['stale'],saved['epoch']+1
    for epoch in range(start,30):
        if epoch>=6 and stale>=6:break
        model.train();order=torch.multinomial(weights,1024,replacement=True,generator=torch.Generator().manual_seed(215900+epoch))
        for batch in order.split(64):
            ids=train[batch.numpy()];positive,negative,clinical=forward(ids);raw=torch.where(truth[ids]==1,positive,negative)
            rank=hard_negative_rank_loss(torch.stack((positive,negative),1),truth[ids],margin=1.,maximum_negatives=4)
            loss=clinical+.2*rank
            assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        model.eval()
        with torch.inference_mode():
            positive,negative,_=forward(val);raw=torch.where(truth[val]==1,positive,negative);loss=float(F.binary_cross_entropy_with_logits(raw,truth[val]));probability=positive.sigmoid().cpu().numpy();score,errors=decision(probability,loss)
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if best is None or score>tuple(best):
            best=score;stale=0;atomic_save(dict(model=state,epoch=epoch,protocol_sha256=digest(out/'protocol.json')),out/'best.pt')
            np.savez(out/'best_validation_predictions.npz',indices=val,probability=probability,protocol_sha256=digest(out/'protocol.json'),checkpoint_sha256=digest(out/'best.pt'))
        else:stale+=1
        history.append(dict(epoch=epoch,decision=score,new_false_healthy_by_source=errors));print(json.dumps(history[-1]),flush=True)
        atomic_save(dict(model=state,optimizer=optimizer.state_dict(),rng=capture_rng(),epoch=epoch,history=history,best=best,stale=stale,protocol_sha256=digest(out/'protocol.json')),out/'last.pt')
    result=dict(source_sha256=digest(__file__),protocol_sha256=digest(out/'protocol.json'),best_decision=best,selected_epoch=torch.load(out/'best.pt',weights_only=True)['epoch'],history=history,
        cheap_pass=best[0]==0 and best[1]>=28,passed=False,model_promoted=False,new_encoder_inputs=0,elapsed_seconds=time.monotonic()-started,ad_independent=False,olid_independent=False,acceptance_data_used=False)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)


if __name__=='__main__':main()
