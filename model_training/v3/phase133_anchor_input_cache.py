"""Extend the unchanged pre-transformer cache for class-balanced disease replay."""
from pathlib import Path
import argparse,collections,hashlib,json,shutil,time
import numpy as np
import torch,timm
from torch.nn import functional as F
from partial_dino_cache import CanonicalInputs,original_dino
from phase114_new_supervision_features import original_features
from opened_context_data import read,load
from phase85_population_sft import evaluation_data
from permissive_encoder import digest
from population_siglip_tail import freeze
from safe_feature_memmap import feature_memmap


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);out=p/'phase133_anchor_input_cache';out.mkdir(exist_ok=True);started=time.monotonic()
    values,bags,descriptors,parent=original_features(p);rows=read(p/'phase84_population_head/fitting_rows.json');clinical=read(p/'phase127_clinical_anchors/protocol.json');selection=read(p/'phase118_conditional_recovery/protocol.json')
    data=load(p);entries,records,packets,before,previous=evaluation_data(p,data);originals={r['path']:r for group in list(data['base'].values())+[data['internal']] for r in group};originals.update({r['path']:r for r in before})
    target=[i for i in selection['train_indices'] if rows[i]['source']=='olid_tomato' and i in set(clinical['train_indices'])];pool=collections.defaultdict(list)
    for i in clinical['train_indices']:
        if i in set(target):continue
        row=rows[i];label=row['label']
        if label=='Other':label+='|'+str(originals[row['path']].get('raw_label','Other')).strip().lower()
        pool[(row['source'],label)].append(i)
    anchors=[]
    for group,ids in sorted(pool.items()):
        families=set()
        for i in sorted(ids,key=lambda i:hashlib.sha256(('phase133:'+rows[i]['path']).encode()).digest()):
            if rows[i]['group'] in families:continue
            anchors.append(i);families.add(rows[i]['group'])
            if len(families)==8:break
    train=target+anchors;val=selection['validation_indices'];all_rows=sorted(set(train+val));assert set(train)<=set(clinical['train_indices'])
    assert not ({rows[i]['group'] for i in train}&{rows[i]['group'] for i in val})
    selected=sorted({key for i in all_rows for keys in bags[rows[i]['path']].values() for key in keys})
    donor=p/'phase123_lora_input_cache';dp=read(donor/'protocol.json');dc=read(donor/'complete.json');assert dc['protocol_sha256']==digest(donor/'protocol.json') and dc['tokens_sha256']==digest(donor/'tokens.npy')
    donor_lookup={sha:i for i,sha in enumerate(dp['input_sha256'])};donor_tokens=np.load(donor/'tokens.npy',mmap_mode='r')
    protocol=dict(source_sha256=digest(__file__),parents=parent,clinical_protocol_sha256=digest(p/'phase127_clinical_anchors/protocol.json'),donor_complete_sha256=digest(donor/'complete.json'),
        train_indices=train,validation_indices=val,target_indices=target,anchor_indices=anchors,maximum_anchor_families_per_source_condition=8,
        anchor_source_condition_counts={str(group):min(8,len({rows[i]['group'] for i in ids})) for group,ids in pool.items()},
        selected_tables=selected,input_sha256=[descriptors[key]['input_sha256'] for key in selected],shape=[len(selected),257,384],
        representation=dp['representation'],input_prefix_state_sha256=dp['input_prefix_state_sha256'],model_sha256=digest(p/'export_cnn_safe/pytorch_bundle.pt'),
        torch_version=torch.__version__,timm_version=timm.__version__,tf32=False,fused_attention=False,
        all_validation_raw_and_roi_identities_quarantined=True,ad_independent=False,olid_independent=False,acceptance_data_used=False)
    freeze(out/'protocol.json',protocol)
    if (out/'complete.json').exists():
        done=read(out/'complete.json');assert done['protocol_sha256']==digest(out/'protocol.json') and done['tokens_sha256']==digest(out/'tokens.npy');return
    assert (out/'tokens.npy').exists() or shutil.disk_usage(out).free>len(selected)*257*384*4+(1<<30)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;timm.layers.set_fused_attn(False)
    model=original_dino(p).cuda().eval();state=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        if name in ('cls_token','reg_token','pos_embed') or name.startswith(('patch_embed.','norm_pre.')):
            array=value.detach().cpu().contiguous().numpy();state.update(str((name,str(array.dtype),array.shape)).encode());state.update(array.tobytes())
    assert state.hexdigest()==dp['input_prefix_state_sha256']
    tokens=feature_memmap(out/'tokens.npy',protocol['shape']);progress=read(out/'progress.json') if (out/'progress.json').exists() else dict(completed=0,reused=0,new_input_embeddings=0)
    if (out/'progress.json').exists():assert progress['protocol_sha256']==digest(out/'protocol.json')
    pixels=CanonicalInputs(p)
    try:
        with torch.inference_mode():
            for start in range(progress['completed'],len(selected),16):
                batch=[];missing=[];reused=0
                for offset,key in enumerate(selected[start:start+16]):
                    row=descriptors[key];old=donor_lookup.get(row['input_sha256'])
                    if old is not None:tokens[start+offset]=donor_tokens[old];reused+=1
                    else:batch.append(pixels(row if 'kind' in row else dict(row,kind='new',policy=row['view'])));missing.append(offset)
                if missing:
                    x=torch.from_numpy(np.concatenate(batch)).cuda().permute(0,3,1,2)
                    tokens[np.array(missing)+start]=model.norm_pre(model.patch_drop(model._pos_embed(model.patch_embed(x)))).cpu().numpy()
                tokens.flush();progress.update(completed=start+len(missing)+reused,reused=progress['reused']+reused,new_input_embeddings=progress['new_input_embeddings']+len(missing),protocol_sha256=digest(out/'protocol.json'))
                tmp=out/'progress.tmp';tmp.write_text(json.dumps(progress));tmp.replace(out/'progress.json')
            audit=np.unique(np.linspace(0,len(selected)-1,16,dtype=int));raw=model.norm(model.blocks(torch.from_numpy(np.array(tokens[audit],copy=True)).cuda()))
            error=float(np.abs((F.normalize(raw[:,0],dim=1)*384**.5).cpu().numpy()-values[np.array(selected)[audit],:384]).max());assert error<.002
    finally:pixels.close()
    result=dict(protocol_sha256=digest(out/'protocol.json'),tokens_sha256=digest(out/'tokens.npy'),**{k:v for k,v in progress.items() if k!='protocol_sha256'},
        original_transformer_audit_inputs=len(audit),maximum_original_cls_error=error,elapsed_seconds=time.monotonic()-started,ad_independent=False,olid_independent=False,acceptance_data_used=False)
    (out/'complete.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
