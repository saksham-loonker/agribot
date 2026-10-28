"""Original nine-class supervised LoRA with identity-pinned shared input embeddings."""
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
from population_siglip_tail import freeze
from safe_feature_memmap import feature_memmap
import shutil,json


class ClinicalPhotometricDino(nn.Module):
    def __init__(self,network,calibration):
        super().__init__();self.network=network;self.calibration=calibration

    def stem(self,x):
        return self.network.norm_pre(self.network.patch_drop(self.network._pos_embed(self.network.patch_embed(x))))

    def from_tokens(self,tokens):
        logits=self.network.forward_head(self.network.norm(self.network.blocks(tokens)))
        return logits

    def margin(self,logits):return healthy_margin(logits)


def prepare(p,pixels,selected,input_sha256):
    p=Path(p);timm.layers.set_fused_attn(False)
    original=original_dino(p).cuda().eval();calibration=[1.,0.]
    warm=p/'export_cnn_safe/pytorch_bundle.pt'
    with torch.inference_mode():expected=healthy_margin(original(pixels[:16]))
    model=ClinicalPhotometricDino(inject_qv_lora(original),calibration).cuda().eval()
    cache=p/'phase133_anchor_input_cache';cp=read(cache/'protocol.json');cc=read(cache/'complete.json');assert cc['protocol_sha256']==digest(cache/'protocol.json') and cc['tokens_sha256']==digest(cache/'tokens.npy')
    state=hashlib.sha256()
    for name,value in sorted(model.network.state_dict().items()):
        if name in ('cls_token','reg_token','pos_embed') or name.startswith(('patch_embed.','norm_pre.')):
            array=value.detach().cpu().contiguous().numpy();state.update(str((name,str(array.dtype),array.shape)).encode());state.update(array.tobytes())
    assert state.hexdigest()==cp['input_prefix_state_sha256'],'Cannot reuse changed patch/position embeddings'
    donor=np.load(cache/'tokens.npy',mmap_mode='r');lookup={sha:i for i,sha in enumerate(cp['input_sha256'])}
    missing=[i for i,key in enumerate(selected) if input_sha256[key] not in lookup]
    extension=p/'phase146_new_input_prefix_cache';extension.mkdir(exist_ok=True)
    protocol=dict(input_prefix_state_sha256=state.hexdigest(),input_sha256=[input_sha256[selected[i]] for i in missing],shape=[len(missing),257,384],dtype='float32',torch_version=torch.__version__,timm_version=timm.__version__,tf32=False,source_sha256=digest(__file__),ad_independent=False,olid_independent=False,acceptance_data_used=False)
    freeze(extension/'protocol.json',protocol);done_path=extension/'complete.json';new=0
    if done_path.exists():
        complete=read(done_path);assert complete['protocol_sha256']==digest(extension/'protocol.json') and complete['tokens_sha256']==digest(extension/'tokens.npy')
    else:
        assert (extension/'tokens.npy').exists() or shutil.disk_usage(extension).free>len(missing)*257*384*4+(1<<30)
        values=feature_memmap(extension/'tokens.npy',protocol['shape']);progress=read(extension/'progress.json') if (extension/'progress.json').exists() else dict(completed=0,protocol_sha256=digest(extension/'protocol.json'))
        assert progress['protocol_sha256']==digest(extension/'protocol.json')
        with torch.inference_mode():
            for start in range(progress['completed'],len(missing),32):
                part=missing[start:start+32];values[start:start+len(part)]=model.stem(pixels[part]).cpu().numpy();values.flush();new+=len(part)
                tmp=extension/'progress.tmp';tmp.write_text(json.dumps(dict(completed=start+len(part),protocol_sha256=digest(extension/'protocol.json'))));tmp.replace(extension/'progress.json')
        done_path.write_text(json.dumps(dict(protocol_sha256=digest(extension/'protocol.json'),tokens_sha256=digest(extension/'tokens.npy')),indent=2))
    extra=np.load(extension/'tokens.npy',mmap_mode='r');new_lookup={master:i for i,master in enumerate(missing)}
    tokens=torch.empty((len(selected),257,384),dtype=torch.float32,device='cuda');reused=0
    with torch.inference_mode():
        for i,key in enumerate(selected):
            if input_sha256[key] in lookup:value=donor[lookup[input_sha256[key]]];reused+=1
            else:value=extra[new_lookup[i]]
            tokens[i]=torch.from_numpy(np.array(value,copy=True)).cuda()
        actual=model.margin(model.from_tokens(tokens[:16]));error=float((actual-expected).abs().max());assert error<.0002,('LoRA/prefix restore changes original nine-class margin',error)
    return model,tokens,dict(warm_original_model_sha256=digest(warm),donor_complete_sha256=digest(cache/'complete.json'),extension_complete_sha256=digest(done_path),input_prefix_state_sha256=state.hexdigest(),original_input_prefixes_reused=reused,
        newly_computed_original_input_embeddings=new,maximum_initial_margin_error=error,rank=8,alpha=16.,dropout=.05,blocks=12,calibration=calibration)
