"""Pretrained local-texture CNN alternative with mutable-only adapter checkpoints."""
from pathlib import Path
import json,shutil
import numpy as np
import torch,torchvision
from torch import nn
from torchvision.models import convnext_tiny,ConvNeXt_Tiny_Weights
from torchvision.models.convnext import CNBlock
from lora_linear import LoraLinear
from opened_context_data import read
from partial_dino_healthy import healthy_margin
from permissive_encoder import digest
from population_siglip_tail import freeze
from safe_feature_memmap import feature_memmap


class ConvNextClinicalTeacher(nn.Module):
    def __init__(self,backbone):
        super().__init__();backbone.requires_grad_(False)
        self.stem_module=backbone.features[0];self.blocks=backbone.features[1:];self.avgpool=backbone.avgpool
        self.classifier=nn.Sequential(nn.LayerNorm(768,eps=backbone.classifier[0].eps),nn.Linear(768,9));self.classifier[0].load_state_dict(backbone.classifier[0].state_dict())
        self.adapted_layers=0
        for block in self.blocks.modules():
            if isinstance(block,CNBlock):
                for i in (3,5):block.block[i]=LoraLinear(block.block[i]);self.adapted_layers+=1
                block.block[2].requires_grad_(True)

    def stem(self,x):return self.stem_module(x.contiguous())

    def features(self,prefix):return self.avgpool(self.blocks(prefix.contiguous(memory_format=torch.channels_last))).flatten(1)

    def from_tokens(self,prefix):return self.classifier(self.features(prefix))

    def margin(self,logits):return healthy_margin(logits)


def prepare(p,pixels,selected,input_sha256):
    p=Path(p);torch.manual_seed(2139);weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1;url=weights.url
    checkpoint=Path(torch.hub.get_dir())/'checkpoints'/url.rsplit('/',1)[1];checkpoint.parent.mkdir(parents=True,exist_ok=True)
    if not checkpoint.exists():assert shutil.disk_usage(checkpoint.parent).free>114419221+(64<<20)+(1<<30),'Preserve disk/cache reserve'
    state=torch.hub.load_state_dict_from_url(url,check_hash=True,weights_only=True,map_location='cpu',progress=False)
    sha=digest(checkpoint);assert sha.startswith('983f1562'),'Official pretrained checkpoint integrity mismatch'
    backbone=convnext_tiny(weights=None).cuda().eval();backbone.load_state_dict(state,strict=True);del state
    with torch.inference_mode():expected=backbone.avgpool(backbone.features(pixels[:16])).flatten(1)
    model=ConvNextClinicalTeacher(backbone).cuda().eval();del backbone
    sample=model.stem(pixels[:1]).shape[1:]
    prefix=torch.empty((len(selected),*sample),dtype=torch.float32,device='cuda').contiguous(memory_format=torch.channels_last)
    with torch.inference_mode():
        for start in range(0,len(selected),16):prefix[start:start+16]=model.stem(pixels[start:start+16])
        error=float((model.features(prefix[:16])-expected).abs().max());assert error<.002,('Zero ConvNeXt adapters/prefix change pretrained features',error)
    out=p/'phase154_convnext_features';out.mkdir(exist_ok=True)
    protocol=dict(source_sha256=digest(__file__),adapter_source_sha256=digest(Path(__file__).with_name('lora_linear.py')),pretrained_url=url,pretrained_sha256=sha,
        input_sha256=[input_sha256[key] for key in selected],shape=[len(selected),768],representation='Frozen original ConvNeXt-Tiny pooled spatial features before the clinical LayerNorm/head',
        input='Exact existing normalized canonical 224 ROI tensors; ImageNet mean/std; no additional 236px evaluation resize/crop',torch_version=torch.__version__,torchvision_version=torchvision.__version__,
        tf32=False,dtype='float32',raw_input_layout='contiguous_NCHW',stem_prefix_layout='channels_last_NCHW',ad_independent=False,olid_independent=False,acceptance_data_used=False)
    freeze(out/'protocol.json',protocol);new=0
    if (out/'complete.json').exists():
        result=read(out/'complete.json');assert result['protocol_sha256']==digest(out/'protocol.json') and result['features_sha256']==digest(out/'features.npy')
    else:
        values=feature_memmap(out/'features.npy',protocol['shape']);progress=read(out/'progress.json') if (out/'progress.json').exists() else dict(completed=0,protocol_sha256=digest(out/'protocol.json'))
        assert progress['protocol_sha256']==digest(out/'protocol.json')
        with torch.inference_mode():
            for start in range(progress['completed'],len(selected),16):
                values[start:start+16]=model.features(prefix[start:start+16]).cpu().numpy();values.flush();new+=min(16,len(selected)-start)
                tmp=out/'progress.tmp';tmp.write_text(json.dumps(dict(completed=start+min(16,len(selected)-start),protocol_sha256=digest(out/'protocol.json'))));tmp.replace(out/'progress.json')
                if start%256==0:print(json.dumps(dict(stage='new_convnext_frozen_features',computed=start+min(16,len(selected)-start),total=len(selected))),flush=True)
        (out/'complete.json').write_text(json.dumps(dict(protocol_sha256=digest(out/'protocol.json'),features_sha256=digest(out/'features.npy')),indent=2))
    model.frozen_features=torch.from_numpy(np.array(np.load(out/'features.npy',mmap_mode='r'),copy=True)).cuda()
    return model,prefix,dict(architecture='torchvision.convnext_tiny',pretrained_url=url,pretrained_sha256=sha,feature_cache_complete_sha256=digest(out/'complete.json'),
        missing_frozen_features_computed=new,frozen_features_reused=len(selected)-new,stem_inputs=len(selected),maximum_initial_feature_error=error,
        rank=8,alpha=16.,dropout=.05,adapted_channel_mixing_layers=model.adapted_layers,stem_and_downsample_frozen=True,block_layernorm_and_clinical_head_trainable=True,
        dimension=768,calibration=[1.,0.],role='Alternative local-texture teacher; mobile deployment or distillation requires all unchanged recognition/safety and acceptance gates')
