"""Warm rank-eight DINO recovery with reusable unchanged input embeddings."""
import hashlib
from pathlib import Path
import numpy as np
import torch,timm
from torch import nn
from continued_diagnostic_expert import ContinuedDiagnosticExpert
from lora_dino import inject_qv_lora
from opened_context_data import read
from partial_dino_cache import original_dino
from partial_dino_healthy import healthy_margin
from permissive_encoder import digest


class PhotometricDino(nn.Module):
    def __init__(self,network,calibration):
        super().__init__();self.network=network;self.calibration=calibration

    def stem(self,x):
        return self.network.norm_pre(self.network.patch_drop(self.network._pos_embed(self.network.patch_embed(x))))

    def from_tokens(self,tokens):
        logits=self.network.forward_head(self.network.norm(self.network.blocks(tokens)))
        margin=healthy_margin(logits)*self.calibration[0]+self.calibration[1]
        return torch.stack([margin,torch.zeros_like(margin)],1)


def prepare(p,pixels,selected,input_sha256):
    p=Path(p);timm.layers.set_fused_attn(False)
    initial=torch.load(p/'phase55_dino_continuation/head/candidate.pt',weights_only=True,map_location='cpu')
    warm=p/'phase97_continued_verified_expert/best.pt';assert read(warm.parent/'result.json')['checkpoint_sha256']==digest(warm)
    original=original_dino(p);expert=ContinuedDiagnosticExpert(original,initial);expert.load_state_dict(torch.load(warm,weights_only=True,map_location='cpu')['model'])
    for offset,module in enumerate(expert.blocks,start=len(original.blocks)-2):original.blocks[offset]=module
    original.norm=expert.norm;original.fc_norm=expert.head[0];original.head=expert.head[1];del expert
    original=original.cuda().eval();calibration=initial['calibration']
    with torch.inference_mode():expected=healthy_margin(original(pixels[:16]))*calibration[0]+calibration[1]
    model=PhotometricDino(inject_qv_lora(original),calibration).cuda().eval()
    cache=p/'phase133_anchor_input_cache';cp=read(cache/'protocol.json');cc=read(cache/'complete.json');assert cc['protocol_sha256']==digest(cache/'protocol.json') and cc['tokens_sha256']==digest(cache/'tokens.npy')
    state=hashlib.sha256()
    for name,value in sorted(model.network.state_dict().items()):
        if name in ('cls_token','reg_token','pos_embed') or name.startswith(('patch_embed.','norm_pre.')):
            array=value.detach().cpu().contiguous().numpy();state.update(str((name,str(array.dtype),array.shape)).encode());state.update(array.tobytes())
    assert state.hexdigest()==cp['input_prefix_state_sha256'],'Cannot reuse changed patch/position embeddings'
    donor=np.load(cache/'tokens.npy',mmap_mode='r');lookup={sha:i for i,sha in enumerate(cp['input_sha256'])};tokens=torch.empty((len(selected),257,384),dtype=torch.float32,device='cuda');reused=0;new=0
    with torch.inference_mode():
        for start in range(0,len(selected),32):
            missing=[]
            for offset,key in enumerate(selected[start:start+32]):
                if input_sha256[key] in lookup:tokens[start+offset]=torch.from_numpy(np.array(donor[lookup[input_sha256[key]]],copy=True)).cuda();reused+=1
                else:missing.append(start+offset)
            if missing:tokens[missing]=model.stem(pixels[missing]);new+=len(missing)
        actual=model.from_tokens(tokens[:16])[:,0];error=float((actual-expected).abs().max());assert error<.0002,('LoRA/prefix restore changes warm margin',error)
    return model,tokens,dict(warm_expert_sha256=digest(warm),donor_complete_sha256=digest(cache/'complete.json'),input_prefix_state_sha256=state.hexdigest(),original_input_prefixes_reused=reused,
        missing_original_input_embeddings=new,maximum_initial_margin_error=error,rank=8,alpha=16.,dropout=.05,blocks=12,calibration=calibration)
