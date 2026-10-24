"""Run real-GPU tiny-set, interrupted-resume and classifier-preservation checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import timm
import torch

from train import LABELS
from runner import normalized_cache_input
from training_support import load_initial_weights, sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--work',required=True)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    work=Path(args.work)
    records=json.loads((work/'data_manifest.json').read_text())
    cache=np.load(work/'cache_256.npy',mmap_mode='r')
    picked=[]
    for split,count in [('train',3),('val',1),('calib',1)]:
        for label in LABELS:
            candidates=[r for r in records if r['split']==split and r['label']==label]
            seen=set()
            for row in candidates:
                if row['group'] not in seen:
                    picked.append(dict(row)); seen.add(row['group'])
                if len(seen)==count: break
            if len(seen)!=count: raise ValueError(f'Insufficient {split}/{label} groups')
    pixels=np.stack([cache[r['cache_index']] for r in picked])
    for i,row in enumerate(picked): row['cache_index']=i
    tiny=out/'tiny';tiny.mkdir()
    np.save(tiny/'cache_256.npy',pixels)
    (tiny/'data_manifest.json').write_text(json.dumps(picked))
    torch.set_num_threads(4);torch.manual_seed(17)
    m=timm.create_model('vit_small_patch14_dinov2.lvd142m',pretrained=False,num_classes=9,img_size=224).cuda().eval()
    init=load_initial_weights(m,args.checkpoint,LABELS,'vit_small_patch14_dinov2.lvd142m',224)
    m.requires_grad_(False);m.head.requires_grad_(True)
    ix=[i for i,r in enumerate(picked) if r['split']=='train']
    x=normalized_cache_input(torch.from_numpy(pixels[ix]).cuda(),224)
    y=torch.tensor([LABELS.index(picked[i]['label']) for i in ix],device='cuda')
    with torch.no_grad(): features=m.forward_head(m.forward_features(x),pre_logits=True)
    opt=torch.optim.AdamW(m.head.parameters(),lr=1e-3,weight_decay=0.)
    for step in range(500):
        logits=m.head(features);loss=torch.nn.functional.cross_entropy(logits,y)
        opt.zero_grad();loss.backward();opt.step()
        accuracy=float((logits.argmax(1)==y).float().mean())
        if accuracy>=.999: break
    result=dict(initialization=init,tiny_n=len(ix),tiny_accuracy=accuracy,tiny_steps=step+1,tiny_loss=float(loss.detach()))
    if accuracy<.95: raise RuntimeError(f'Tiny-set fit failed: {result}')
    del m,features,opt,x,y
    torch.cuda.empty_cache()
    base=[sys.executable,str(Path(__file__).with_name('train.py')),'--work',str(tiny),'--runs',str(out/'runs'),
          '--init-checkpoint',args.checkpoint,'--epochs','3','--warmup-epochs','1','--unfreeze-blocks','1',
          '--bs','16','--max-steps','2','--no-augment','--precision','fp32','--drop-path','0',
          '--lr-head','0.00001','--lr-backbone','0.000001']
    for suffix in [['--run-name','continuous'],['--run-name','interrupted','--stop-after-epoch','0'],
                   ['--run-name','interrupted','--resume']]:
        subprocess.run(base+suffix,check=True)
    a=torch.load(out/'runs/continuous/last.pt',map_location='cpu',weights_only=False)
    b=torch.load(out/'runs/interrupted/last.pt',map_location='cpu',weights_only=False)
    result['resume_max_parameter_error']=max(float((a['model'][k]-b['model'][k]).abs().max()) for k in a['model'])
    result['resume_max_ema_error']=max(float((a['ema'][k]-b['ema'][k]).abs().max()) for k in a['ema'])
    result['global_steps']=[a['global_step'],b['global_step']]
    result['optimizer_steps']=sorted({int(v['step']) for v in a['optimizer']['state'].values()})
    result['passed']=result['resume_max_parameter_error']<=1e-7 and result['resume_max_ema_error']<=1e-7 and a['global_step']==b['global_step']
    (out/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    assert result['passed'],'Interrupted run differs from continuous run'


if __name__=='__main__': main()
