"""A copied spatial CNN suffix with canonical-input-hash cache joins."""
from pathlib import Path
import copy
import json
import math
import shutil
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from app_eval import canon,classifier_letterbox_input
from input_identity_join import input_identity_join
from partial_cnn_cache import original_cnn
from partial_cnn_healthy import prefix_features,prefix_identity,prefix_cut
from partial_dino_cache import CanonicalInputs,input_rows
from opened_context_data import read
from permissive_encoder import digest
from phase28_roi_confirmation import pixel_key
from population_siglip_tail import freeze


class CopiedCnnBagTail(nn.Module):
    def __init__(self,cnn):
        super().__init__()
        self.blocks=copy.deepcopy(cnn.blocks[prefix_cut(cnn):])
        self.norm=nn.Sequential(copy.deepcopy(cnn.global_pool),copy.deepcopy(cnn.conv_head),
                                copy.deepcopy(cnn.norm_head),copy.deepcopy(cnn.act2),copy.deepcopy(cnn.flatten))
        self.head=copy.deepcopy(cnn.classifier)
        self.requires_grad_(True)

    def train(self,mode=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module,nn.modules.batchnorm._BatchNorm):module.eval()
        return self

    def forward(self,prefix,unused):
        z=self.head(self.norm(self.blocks(prefix)))
        return z[:,0]-z[:,1]


def prepare_tokens(campaign,values,bags,fitting):
    p=Path(campaign);out=p/"phase101_cnn_bag_adaptation/prefix";out.mkdir(parents=True,exist_ok=True)
    donor=p/"phase49_partial_cnn/prefix";complete=read(donor/"complete.json")
    assert complete["protocol_sha256"]==digest(donor/"protocol.json")
    assert complete["features_sha256"]==digest(donor/"features.npy")
    cnn,architecture,normalization=original_cnn(p);cnn=cnn.cuda().eval()
    assert prefix_identity(cnn)==complete["prefix_state_sha256"]
    donor_rows=input_rows(p)
    old_count=len(read(p/"phase29_compact/inputs.json"))
    new_rows=read(p/"phase87_diagnostic_bag_features/new_inputs.json")
    target_rows=donor_rows[:old_count]+new_rows
    active={r["path"] for r in fitting if r["active"]}
    tables=sorted({i for path in active if path in bags for indices in bags[path].values() for i in indices})
    donor_values=np.load(donor/"features.npy",mmap_mode="r")
    assert len(donor_rows)==len(donor_values)==complete["inputs"]
    joined=input_identity_join([target_rows[i] for i in tables],donor_rows,range(len(donor_rows)))
    lookup={table:offset for table,offset in zip(tables,joined) if offset is not None}
    missing=[i for i in tables if i not in lookup];shape=donor_values.shape[1:]
    protocol=dict(source_sha256=digest(__file__),join_source_sha256=digest(Path(__file__).with_name("input_identity_join.py")),
                  donor_complete_sha256=digest(donor/"complete.json"),feature_complete_sha256=digest(p/"phase87_diagnostic_bag_features/complete.json"),
                  selected_input_sha256=[target_rows[i]["input_sha256"] for i in tables],
                  selected_tables=tables,missing_tables=missing,shape=[len(missing),*shape],
                  prefix_state_sha256=prefix_identity(cnn),architecture=architecture,
                  old_inputs_reused=len(lookup),newly_computed_inputs=len(missing),dtype="float32",tf32=False,
                  positional_index_reuse_forbidden=True,ad_independent=False,olid_independent=False)
    freeze(out/"protocol.json",protocol)
    path=out/"missing_prefix.npy";progress=out/"progress.json"
    saved=read(progress) if progress.exists() else dict(completed=0)
    if progress.exists():assert saved["protocol_sha256"]==digest(out/"protocol.json")
    assert path.exists() or shutil.disk_usage(p).free>len(missing)*int(np.prod(shape))*4+(1<<30)
    new=np.lib.format.open_memmap(path,mode="r+" if path.exists() else "w+",dtype=np.float32,shape=(len(missing),*shape))
    pixels=CanonicalInputs(p)
    normalization={name:value.cuda() for name,value in normalization.items()}
    try:
        with torch.inference_mode():
            for start in range(saved["completed"],len(missing),64):
                batch=[]
                for index in missing[start:start+64]:
                    if index<old_count:tensor=pixels(donor_rows[index])
                    else:
                        row=new_rows[index-old_count]
                        with Image.open(row["path"]) as image:rgb=np.asarray(image.convert("RGB"))
                        tensor=(canon.classifier_input if row["view"]=="center" else classifier_letterbox_input)(rgb,224,row["box"])
                    assert pixel_key(tensor)==target_rows[index]["input_sha256"]
                    batch.append(tensor)
                x=torch.from_numpy(np.concatenate(batch)).cuda().permute(0,3,1,2)
                rgb=(x*normalization["canonical_std"]+normalization["canonical_mean"]-normalization["cnn_mean"])/normalization["cnn_std"]
                new[start:start+len(batch)]=prefix_features(cnn,rgb).cpu().numpy();new.flush()
                saved=dict(completed=start+len(batch),protocol_sha256=digest(out/"protocol.json"))
                temp=progress.with_suffix(".tmp");temp.write_text(json.dumps(saved));temp.replace(progress)
    finally:pixels.close()
    missing_lookup={table:i for i,table in enumerate(missing)}
    gpu=torch.empty((len(tables),*shape),dtype=torch.float32,device="cuda")
    maximum=0.
    with torch.inference_mode():
        for start in range(0,len(tables),64):
            part=tables[start:start+64]
            array=np.stack([donor_values[lookup[i]] if i in lookup else new[missing_lookup[i]] for i in part])
            prefix=torch.from_numpy(array).cuda();gpu[start:start+len(part)]=prefix
            restored=cnn.forward_head(cnn.blocks[prefix_cut(cnn):](prefix),pre_logits=True)
            restored=(F.normalize(restored,dim=1)*math.sqrt(1024)).cpu().numpy()
            maximum=max(maximum,float(np.abs(restored-values[part,384:]).max()))
            assert maximum<.002,("CNN prefix/input mismatch",part,maximum)
    report=dict(protocol_sha256=digest(out/"protocol.json"),missing_prefix_sha256=digest(path),
                old_inputs_reused=len(lookup),newly_computed_inputs=len(missing),
                all_selected_inputs_audited=len(tables),all_selected_maximum_error=maximum)
    (out/"complete.json").write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    return cnn,gpu,{table:i for i,table in enumerate(tables)},report
