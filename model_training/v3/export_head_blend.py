"""Export two supervised heads sharing an unchanged verified DINOv2 backbone.

The retained original head and adapted head are mixed in probability space.
Backbone equality is mandatory; this is not a second ViT invocation on Android.
This tool creates an evaluation candidate, not deployment approval.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import timm
import torch

from train import LABELS
from training_support import load_initial_weights


def conservative_mix(base, adapted, ratio, max_reference_alternative, reference_bias, reference_temperature):
    mixture=((1-ratio)*base+ratio*adapted).clamp_min(1e-12)
    if max_reference_alternative is not None:
        reference=((base.clamp_min(1e-12).log()+reference_bias)/reference_temperature).softmax(-1)
        alternatives=torch.cat([reference[:,:1],reference[:,2:]],dim=1).amax(1)
        eligible=(alternatives<max_reference_alternative)&(mixture[:,1]>base[:,1])
        mixture=torch.where(eligible[:,None],mixture,base)
    return mixture.clamp_min(1e-12)


class SharedHeadBlend(torch.nn.Module):
    def __init__(self, original, adapted, ratio, max_reference_alternative=None, reference_temperature=.8814700841903687, reference_other_bias=.75):
        super().__init__()
        if not 0 < ratio < 1:
            raise ValueError('Blend ratio must be between zero and one')
        if max_reference_alternative is not None and not 0 < max_reference_alternative < 1:
            raise ValueError('Alternative confidence threshold must be between zero and one')
        a,b=original.state_dict(),adapted.state_dict()
        if any(not torch.equal(a[k],b[k]) for k in a if not k.startswith('head.')):
            raise ValueError('Backbones differ: shared-feature export would change model behavior')
        self.backbone=original
        self.adapted_head=copy.deepcopy(adapted.head)
        self.ratio=ratio
        self.max_reference_alternative=max_reference_alternative
        self.reference_temperature=reference_temperature
        self.register_buffer('reference_bias',torch.tensor([0.]*8+[reference_other_bias]))

    def forward(self, nhwc):
        features=self.backbone.forward_head(self.backbone.forward_features(nhwc.permute(0,3,1,2)),pre_logits=True)
        base=self.backbone.head(features).softmax(-1)
        adapted=self.adapted_head(features).softmax(-1)
        mixture=conservative_mix(base,adapted,self.ratio,self.max_reference_alternative,self.reference_bias,self.reference_temperature)
        return mixture.clamp_min(1e-12).log()


def load_export_bundle(path):
    """Restore a self-contained trainable bundle without the legacy checkpoint directory."""
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state.get('architecture')!='SharedHeadBlend' or state.get('labels')!=LABELS:
        raise ValueError('Unsupported trainable bundle or label order')
    base=timm.create_model('vit_small_patch14_dinov2.lvd142m',pretrained=False,num_classes=9,img_size=224).eval()
    model=SharedHeadBlend(base,copy.deepcopy(base),state['ratio'],state['max_reference_alternative'],
                          state['reference_temperature'],state['reference_other_bias']).eval()
    model.load_state_dict(state['model'],strict=True)
    return model


def main():
    from ai_edge_litert.interpreter import Interpreter
    import litert_torch
    p=argparse.ArgumentParser();p.add_argument('--original',required=True);p.add_argument('--adapted',required=True)
    p.add_argument('--out',required=True);p.add_argument('--ratio',type=float,default=.5)
    p.add_argument('--max-reference-alternative',type=float)
    a=p.parse_args()
    torch.set_num_threads(4);timm.layers.set_fused_attn(False)
    create=lambda:timm.create_model('vit_small_patch14_dinov2.lvd142m',pretrained=False,num_classes=9,img_size=224).eval()
    base,adapt=create(),create();load_initial_weights(base,a.original,LABELS,'vit_small_patch14_dinov2.lvd142m',224)
    state=torch.load(a.adapted,map_location='cpu',weights_only=False)
    if state['labels']!=LABELS: raise ValueError('Adapted class order differs')
    adapt.load_state_dict(state.get('ema',state['model']),strict=True)
    model=SharedHeadBlend(base,adapt,a.ratio,a.max_reference_alternative).eval();out=Path(a.out);(out/'models').mkdir(parents=True,exist_ok=True)
    torch.save(dict(architecture='SharedHeadBlend',model=model.state_dict(),labels=LABELS,
                    ratio=a.ratio,max_reference_alternative=a.max_reference_alternative,
                    reference_temperature=model.reference_temperature,reference_other_bias=.75),out/'pytorch_bundle.pt')
    x=(torch.rand(1,224,224,3)-torch.tensor([.485,.456,.406]))/torch.tensor([.229,.224,.225])
    target=out/'models/disease_classifier_0.tflite';litert_torch.convert(model,(x,)).export(str(target))
    it=Interpreter(model_path=str(target),num_threads=4);it.allocate_tensors();rows=[]
    cases=[]
    for seed in range(5):
        torch.manual_seed(seed);x=(torch.rand(1,224,224,3)-torch.tensor([.485,.456,.406]))/torch.tensor([.229,.224,.225])
        cases.append(x)
    import sys
    from PIL import Image
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'v2/final'))
    import canon
    for image in sorted((Path(__file__).resolve().parents[1]/'v2/bundle/golden').glob('*.png')):
        cases.append(torch.from_numpy(canon.classifier_input(np.asarray(Image.open(image).convert('RGB')),224)))
    for x in cases:
        with torch.no_grad(): expected=model(x).numpy()
        it.set_tensor(it.get_input_details()[0]['index'],x.numpy());it.invoke();actual=it.get_tensor(it.get_output_details()[0]['index'])
        rows.append(float(abs(actual-expected).max()))
    result=dict(ratio=a.ratio,max_reference_alternative=a.max_reference_alternative,
                reference_temperature=.8814700841903687,reference_other_bias=.75,
                original=a.original,adapted=a.adapted,backbone_shared=True,parity_cases=len(rows),max_abs_logit_error=max(rows),
                sha256=hashlib.sha256(target.read_bytes()).hexdigest(),size_bytes=target.stat().st_size,passed=max(rows)<1e-3)
    (out/'parity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));assert result['passed']


if __name__=='__main__': main()
