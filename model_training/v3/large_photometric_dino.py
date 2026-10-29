"""Licensed DINOv2-L teacher: cached clinical-head warm-up, then adapter-only fitting."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch,timm
from torch import nn
from torch.nn import functional as F
from dino_large_encoder import DinoLargeEncoder
from feature_cache_reuse import validate_completed_contract,runtime_identity
from lora_dino import inject_qv_lora
from opened_context_data import read
from partial_dino_healthy import healthy_margin
from permissive_encoder import digest
from population_siglip_tail import freeze
from safe_feature_memmap import feature_memmap


class LargeClinicalTeacher(nn.Module):
    def __init__(self,network):
        super().__init__();self.network=inject_qv_lora(network);self.classifier=nn.Linear(1024,9)

    def stem(self,x):
        # Match the source encoder's NHWC -> NCHW view exactly. Conv2d
        # accumulation layout can amplify into a visible DINO feature drift.
        x=x.contiguous(memory_format=torch.channels_last)
        return self.network.norm_pre(self.network.patch_drop(self.network._pos_embed(self.network.patch_embed(x))))

    def features(self,tokens):return F.normalize(self.network.norm(self.network.blocks(tokens))[:,0],dim=1)*(1024**.5)

    def from_tokens(self,tokens):return self.classifier(self.features(tokens))

    def margin(self,logits):return healthy_margin(logits)


def trainable_state(model):
    names={name for name,value in model.named_parameters() if value.requires_grad}
    return {key:value.detach().cpu().clone() for key,value in model.state_dict().items() if key in names}


def restore_trainable(model,state):
    names={name for name,value in model.named_parameters() if value.requires_grad};assert set(state)==names,'Adapter checkpoint trainable parameter contract changed'
    result=model.load_state_dict(state,strict=False)
    assert not result.unexpected_keys and set(result.missing_keys)==set(model.state_dict())-names


def prepare(p,pixels,selected,input_sha256):
    p=Path(p);source=p/'phase23_dino';directory=source/'dinov2_large_feature_heads';validate_completed_contract(source,directory,'dinov2_large')
    torch.manual_seed(2139);timm.layers.set_fused_attn(False)
    state=torch.load(directory/'encoder.pt',weights_only=True,map_location='cpu');encoder=DinoLargeEncoder(state['spec']);encoder.load_state_dict(state['model'],strict=True);del state
    model=LargeClinicalTeacher(encoder.network).cuda().eval();del encoder
    # This is a different encoder/representation from the small-DINO prefixes.
    # Keep it in GPU memory, preserving disk reserve and every existing cache.
    tokens=torch.empty((len(selected),257,1024),dtype=torch.float32,device='cuda')
    with torch.inference_mode():
        for start in range(0,len(selected),16):tokens[start:start+16]=model.stem(pixels[start:start+16])
    donor=p/'phase131_large_roi_cache';dp=read(donor/'protocol.json');dc=read(donor/'complete.json')
    assert dp['encoder_sha256']==digest(directory/'encoder.pt') and dc['protocol_sha256']==digest(donor/'protocol.json') and dc['features_sha256']==digest(donor/'features.npy')
    lookup={sha:i for i,sha in enumerate(dp['input_sha256'])};old=np.load(donor/'features.npy',mmap_mode='r');missing=[i for i,key in enumerate(selected) if input_sha256[key] not in lookup]
    reused_masters=[i for i,key in enumerate(selected) if input_sha256[key] in lookup];assert reused_masters
    preflight=np.array(reused_masters)[np.unique(np.linspace(0,len(reused_masters)-1,min(16,len(reused_masters)),dtype=int))]
    with torch.inference_mode():
        reference=torch.from_numpy(np.array(old[[lookup[input_sha256[selected[i]]] for i in preflight]],copy=True)).cuda()
        preflight_error=float((model.features(tokens[preflight])-reference).abs().max());assert preflight_error<.002,('Large encoder layout preflight differs from verified original cache',preflight_error)
    print(json.dumps(dict(stage='large_encoder_layout_preflight',maximum_feature_error=preflight_error,inputs=len(preflight),passed=True)),flush=True)
    out=p/'phase147_large_teacher_features';out.mkdir(exist_ok=True)
    protocol=dict(source_sha256=digest(__file__),encoder_sha256=digest(directory/'encoder.pt'),donor_complete_sha256=digest(donor/'complete.json'),
        input_sha256=[input_sha256[selected[i]] for i in missing],shape=[len(missing),1024],patch_input_memory_layout='channels_last_NCHW matching original NHWC permutation',representation='Frozen official DINOv2-L normalized CLS, before clinical-head adaptation',
        runtime=runtime_identity(),tf32=False,fused_attention=False,ad_independent=False,olid_independent=False,acceptance_data_used=False)
    freeze(out/'protocol.json',protocol);new=0
    if (out/'complete.json').exists():
        complete=read(out/'complete.json');assert complete['protocol_sha256']==digest(out/'protocol.json') and complete['features_sha256']==digest(out/'features.npy')
    else:
        values=feature_memmap(out/'features.npy',protocol['shape']);progress=read(out/'progress.json') if (out/'progress.json').exists() else dict(completed=0,protocol_sha256=digest(out/'protocol.json'))
        assert progress['protocol_sha256']==digest(out/'protocol.json')
        with torch.inference_mode():
            for start in range(progress['completed'],len(missing),8):
                part=missing[start:start+8];values[start:start+len(part)]=model.features(tokens[part]).cpu().numpy();values.flush();new+=len(part)
                tmp=out/'progress.tmp';tmp.write_text(json.dumps(dict(completed=start+len(part),protocol_sha256=digest(out/'protocol.json'))));tmp.replace(out/'progress.json')
                if start%256==0:print(json.dumps(dict(stage='missing_large_teacher_features',computed=start+len(part),total=len(missing))),flush=True)
        (out/'complete.json').write_text(json.dumps(dict(protocol_sha256=digest(out/'protocol.json'),features_sha256=digest(out/'features.npy')),indent=2))
    extra=np.load(out/'features.npy',mmap_mode='r');new_lookup={master:i for i,master in enumerate(missing)};features=torch.empty((len(selected),1024),dtype=torch.float32,device='cuda');reused=[]
    for i,key in enumerate(selected):
        if input_sha256[key] in lookup:value=old[lookup[input_sha256[key]]];reused.append(i)
        else:value=extra[new_lookup[i]]
        features[i]=torch.from_numpy(np.array(value,copy=True)).cuda()
    assert reused
    audit=np.array(reused)[np.unique(np.linspace(0,len(reused)-1,min(16,len(reused)),dtype=int))]
    with torch.inference_mode():error=float((model.features(tokens[audit])-features[audit]).abs().max());assert error<.002,('Large encoder/canonical prefix cache differs',error)
    # Ordinary attributes deliberately stay out of checkpoints: only adapters,
    # final LayerNorm and the nine-class head are mutable/saved.
    model.frozen_features=features
    return model,tokens,dict(encoder_sha256=digest(directory/'encoder.pt'),donor_complete_sha256=digest(donor/'complete.json'),extension_complete_sha256=digest(out/'complete.json'),
        original_cls_features_reused=len(reused),missing_original_cls_features_computed=new,new_large_input_embeddings=len(selected),large_encoder_audit_inputs=len(audit),maximum_initial_feature_error=error,layout_preflight_maximum_error=preflight_error,patch_input_memory_layout='channels_last_NCHW',
        rank=8,alpha=16.,dropout=.05,blocks=len(model.network.blocks),dimension=1024,calibration=[1.,0.],role='Capacity teacher probe; student distillation and all unchanged gates required before mobile deployment')
