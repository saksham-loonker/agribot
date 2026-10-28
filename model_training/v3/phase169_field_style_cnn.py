"""Controlled MobileNet Healthy-axis SFT, with/without photo-style mixing."""
from pathlib import Path
import argparse,json,shutil,time
import numpy as np
import torch
from torch.nn import functional as F
from conditional_recovery_policy import recover_only
from large_photometric_dino import trainable_state,restore_trainable
from opened_context_data import load,read
from partial_cnn_cache import original_cnn
from partial_cnn_healthy import PartialCnnHealthyTail,prefix_cut
from permissive_encoder import digest
from phase85_population_sft import evaluation_data
from phase168_early_cnn_prefix import prefix_state_identity
from photo_mixstyle import PhotoMixStyle
from population_siglip_tail import freeze
from source_balance import source_class_group_weights
from training_support import atomic_save,capture_rng,restore_rng


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);root=p/'phase169_field_style_cnn';root.mkdir(exist_ok=True);started=time.monotonic()
    if (root/'result.json').exists():assert read(root/'result.json')['source_sha256']==digest(__file__);return
    source=p/'phase168_early_cnn_prefix';cp=read(source/'protocol.json');cc=read(source/'complete.json');assert cc['protocol_sha256']==digest(source/'protocol.json')
    for filename,key in [('features.npy','features_sha256'),('rows.json','rows_sha256'),('bags.json','bags_sha256')]:assert digest(source/filename)==cc[key]
    rows=read(source/'rows.json');tables=read(source/'bags.json');train=np.array(cp['train_indices']);val=np.array(cp['validation_indices']);selection=read(p/'phase118_conditional_recovery/protocol.json');assert val.tolist()==selection['validation_indices']
    keys=cp['input_sha256'];assert not ({keys[key] for i in train for pair in tables[str(i)] for view in pair for key in view}&{keys[key] for i in val for pair in tables[str(i)] for view in pair for key in view})
    assert not ({rows[i]['group'] for i in train}&{rows[i]['group'] for i in val})
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    backbone,architecture,_=original_cnn(p);assert prefix_state_identity(backbone,cp['cut'])==cp['prefix_state_sha256'] and 0<cp['cut']<prefix_cut(backbone)
    backbone=backbone.cuda().eval();prefix=torch.from_numpy(np.array(np.load(source/'features.npy',mmap_mode='r'),copy=True)).cuda().contiguous(memory_format=torch.channels_last)
    index=np.zeros((len(rows),4,2,6),np.int64);mask=np.zeros(index.shape,np.float32)
    for key,pairs in tables.items():
        i=int(key)
        for rotation,pair in enumerate(pairs):
            for view,ids in enumerate(pair):index[i,rotation,view,:len(ids)]=ids;mask[i,rotation,view,:len(ids)]=1.
    index=torch.from_numpy(index).cuda();mask=torch.from_numpy(mask).cuda();valid=mask.sum(-1)>0;truth=torch.tensor([r['y'] for r in rows],dtype=torch.float32,device='cuda')
    target=cp['target_indices'];anchors=cp['anchor_indices'];old107=cp['old107_indices'];field=cp['field_indices'];assert train.tolist()==target+anchors+old107+field
    sampling=torch.from_numpy(np.concatenate([.5*source_class_group_weights(rows,target),.125*source_class_group_weights(rows,anchors),.125*source_class_group_weights(rows,old107),.25*source_class_group_weights(rows,field)]));vw=torch.tensor(source_class_group_weights(rows,val),device='cuda')
    data=load(p);entries,records,packets,before,previous=evaluation_data(p,data);items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({b['path']:(b,packets[b['id']],r) for b,r in zip(before,previous)})
    methods=['earlier_tail_sft','photo_mixstyle_tail_sft']
    protocol=dict(source_sha256=digest(__file__),prefix_complete_sha256=digest(source/'complete.json'),prefix_protocol_sha256=digest(source/'protocol.json'),architecture=architecture,cut=cp['cut'],original_later_cut=prefix_cut(backbone),
        policy_source_sha256=digest(Path(__file__).with_name('conditional_recovery_policy.py')),style_source_sha256=digest(Path(__file__).with_name('photo_mixstyle.py')),wrapper_source_sha256=digest(Path(__file__).with_name('partial_cnn_healthy.py')),
        methods=methods,train_indices=train.tolist(),validation_indices=val.tolist(),healthy_output_column=0,knownness_output_not_subtracted=True,trainable_state_only=True,batchnorm_running_statistics_frozen=True,
        training_mass=dict(target=.5,old_anchor=.125,old107=.125,field93=.25),maximum_epochs=40,minimum_epochs=12,patience=8,draws_per_epoch=1024,batch_photos=16,learning_rate=.0001,weight_decay=.01,seed=2169,
        style=dict(probability_per_photo=.5,beta_alpha=.1,epsilon=1e-6,donor='Random training photo, independent of clinical labels',same_pairing_and_coefficient_across_photo_rois=True,statistics_detached=True,variance_correction=0,applied_at='Frozen MobileNetV3 prefix after block 3; three remaining grouped blocks plus head',validation_and_inference_augmented=False),
        supervision='Original photo-level binary objective: minimum crop mean for Healthy, maximum for disease; actual ROI copies retain dropout independence in training; evaluation deduplicates identical inputs',
        hypothesis='Frozen pooled heads still fail with added field supervision. Test an earlier MobileNet adaptation and a matched, training-only camera-style statistic mixing alternative; clinical invariance is a hypothesis tested by unchanged verdict/safety gates, not an assumption of deployment correctness.',
        existing_healthy_and_named_disease_preserved=True,healthy_admission=.95,new_encoder_inputs=0,new_detector_inputs=0,new_dino_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False,
        stopping='Two fixed supervised controls, minimum twelve epochs, eight stale actual-verdict epochs, forty maximum. Stop the campaign at the first >=28/40 Healthy with zero new false Healthy and run complete unchanged development gates. No restored full-model verification, export or Android integration from a failed screen.')
    freeze(root/'protocol.json',protocol)
    def decision(probability,loss):
        errors={};healthy=selection['fixed_healthy_validation_correct'];old_h=0
        for i,score in zip(val,probability):
            row=rows[i];original,packet,retained=items[row['path']];now=recover_only(original,retained,packet,float(score),data['manifest'])['app_replay']['kind']=='HEALTHY'
            if not row['y']:errors[row['source']]=errors.get(row['source'],0)+int(now)
            elif row['source']=='olid_tomato':healthy+=int(now)
            else:old_h+=int(now)
        return (-sum(errors.values()),healthy,old_h,-loss),errors
    controls=[]
    for method in methods:
        out=root/method;out.mkdir(exist_ok=True)
        if (out/'result.json').exists():
            result=read(out/'result.json');assert result['protocol_sha256']==digest(root/'protocol.json');controls.append(result)
            if result['cheap_pass']:break
            continue
        torch.manual_seed(2169);model=PartialCnnHealthyTail(backbone).cuda();model.cut=cp['cut'];model.cnn.blocks[model.cut:].requires_grad_(True);model.eval();style=PhotoMixStyle().cuda()
        with torch.inference_mode():
            expected=backbone.forward_head(backbone.blocks[model.cut:](prefix[:16]))[:,0];error=float((model(prefix[:16])-expected).abs().max());assert error<.0001,('Earlier copied Healthy output does not restore',error)
        print(json.dumps(dict(stage='earlier_healthy_row_restore',method=method,maximum_error=error,passed=True)),flush=True)
        def forward(ids,training=False):
            actual=mask[ids]>0;flat=index[ids][actual]
            if training:
                value=prefix[flat].contiguous(memory_format=torch.channels_last)
                if method=='photo_mixstyle_tail_sft':
                    owners=torch.arange(len(ids),device='cuda')[:,None,None,None].expand_as(index[ids])[actual];value=style(value,owners).contiguous(memory_format=torch.channels_last)
                raw=model(value)
            else:
                unique,inverse=flat.unique(return_inverse=True);raw=model(prefix[unique].contiguous(memory_format=torch.channels_last))[inverse]
            z=prefix.new_zeros((len(ids),4,2,6));z[actual]=raw;means=(z*mask[ids]).sum(-1)/mask[ids].sum(-1).clamp_min(1)
            return means.masked_fill(~valid[ids],float('inf')).amin((1,2)),means.masked_fill(~valid[ids],-float('inf')).amax((1,2))
        optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=.0001,weight_decay=.01);history=[];best=None;stale=0;start=0
        if (out/'last.pt').exists():
            saved=torch.load(out/'last.pt',weights_only=True,map_location='cpu');assert saved['protocol_sha256']==digest(root/'protocol.json');restore_trainable(model,saved['model']);optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng']);history,best,stale,start=saved['history'],saved['best'],saved['stale'],saved['epoch']+1
        parameter_bytes=sum(v.numel()*v.element_size() for v in model.parameters() if v.requires_grad);assert shutil.disk_usage(out).free>parameter_bytes*5+(1<<30),'Preserve checkpoint/cache disk reserve'
        for epoch in range(start,40):
            if epoch>=12 and stale>=8:break
            model.train();style.train();order=torch.multinomial(sampling,1024,replacement=True,generator=torch.Generator().manual_seed(216900+epoch))
            for batch in order.split(16):
                ids=train[batch.numpy()];positive,negative=forward(ids,training=True);raw=torch.where(truth[ids]==1,positive,negative);loss=F.binary_cross_entropy_with_logits(raw,truth[ids])
                assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            model.eval();style.eval()
            with torch.inference_mode():
                pairs=[forward(part) for part in np.array_split(val,max(1,(len(val)+63)//64))];positive=torch.cat([pair[0] for pair in pairs]);negative=torch.cat([pair[1] for pair in pairs]);raw=torch.where(truth[val]==1,positive,negative)
                loss=float((F.binary_cross_entropy_with_logits(raw,truth[val],reduction='none')*vw).sum());probability=positive.sigmoid().cpu().numpy();score,errors=decision(probability,loss)
            state=trainable_state(model)
            if best is None or score>tuple(best):
                best=score;stale=0;atomic_save(dict(model=state,epoch=epoch,protocol_sha256=digest(root/'protocol.json')),out/'best.pt');np.savez(out/'best_validation_predictions.npz',indices=val,probability=probability,protocol_sha256=digest(root/'protocol.json'),checkpoint_sha256=digest(out/'best.pt'))
            else:stale+=1
            history.append(dict(epoch=epoch,decision=score,new_false_healthy_by_source=errors));print(json.dumps(dict(method=method,**history[-1])),flush=True)
            atomic_save(dict(model=state,optimizer=optimizer.state_dict(),rng=capture_rng(),epoch=epoch,history=history,best=best,stale=stale,protocol_sha256=digest(root/'protocol.json')),out/'last.pt')
            if best[0]==0 and best[1]>=28:break
        chosen=torch.load(out/'best.pt',weights_only=True,map_location='cpu');cheap=best[0]==0 and best[1]>=28
        result=dict(method=method,protocol_sha256=digest(root/'protocol.json'),checkpoint_sha256=digest(out/'best.pt'),best_decision=best,selected_epoch=chosen['epoch'],history=history,cheap_pass=cheap,passed=False,model_promoted=False,new_encoder_inputs=0,new_detector_inputs=0)
        if cheap:result['next_action']='Extend exact earlier-prefix identities/predictions only for missing applicable consumed cases, run every original development recognition/safety gate, then restored full-model parity and suitable frozen prospective acceptance before export'
        (out/'result.json').write_text(json.dumps(result,indent=2));controls.append(result)
        if cheap:break
    result=dict(source_sha256=digest(__file__),protocol_sha256=digest(root/'protocol.json'),controls=controls,model_promoted=False,elapsed_seconds=time.monotonic()-started,new_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False)
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(controls=[{k:r[k] for k in ('method','best_decision','cheap_pass')} for r in controls])),flush=True)


if __name__=='__main__':main()
