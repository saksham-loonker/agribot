"""Native-image 448px CNN inputs: no upsampling of cached 224px tensors."""
from pathlib import Path
import hashlib
import json
import shutil
import sqlite3
import numpy as np
from PIL import Image
import torch
from app_eval import canon,classifier_letterbox_input
from cnn_bag_cache import CopiedCnnBagTail
from feature_cache_reuse import preprocessing_contract,runtime_identity
from opened_context_data import read
from partial_cnn_cache import original_cnn
from partial_cnn_healthy import prefix_cut,prefix_identity
from permissive_encoder import digest
from phase28_roi_confirmation import pixel_key
from phase87_diagnostic_bag_features import cached_mobile
from population_siglip_tail import freeze


def high_resolution_key(tensor):
    assert tensor.dtype==np.float32 and tensor.shape==(1,448,448,3)
    return hashlib.sha256(tensor.tobytes()).hexdigest()


def prepare(campaign,data):
    p=Path(campaign);out=p/"phase104_high_resolution_cnn/prefix";out.mkdir(parents=True,exist_ok=True)
    original=p/"phase87_diagnostic_bag_features"
    complete=read(original/"complete.json")
    assert complete["protocol_sha256"]==digest(original/"protocol.json")
    assert complete["bags_sha256"]==digest(original/"bags.json")
    bags224=read(original/"bags.json")
    descriptors224=read(p/"phase29_compact/inputs.json")+read(original/"new_inputs.json")
    fitting=read(p/"phase84_population_head/fitting_rows.json")
    selected=sorted({r["path"] for r in fitting if r["active"] and r["path"] in bags224 and bags224[r["path"]]["center"]})
    sources=read(p/"phase29_compact/source_files.json")
    sources.update({r["path"]:r["file_sha256"] for r in read(p/"phase24_olid_source/sealed_manifest.json")})
    db=sqlite3.connect("file:"+str(p/"phase76_fresh_acceptance/cache/predictions.sqlite3")+"?mode=ro",uri=True)
    detector_sha=read(p/"phase76_fresh_acceptance/frozen_protocol.json")["models"]["detector"]
    work={};pairs={};native_sides=[]
    for path in selected:
        assert digest(path)==sources[path]
        with Image.open(path) as image:rgb=np.asarray(image.convert("RGB"))
        if path in data["photos"]:
            boxes=[leaf["box"] for leaf in data["photos"][path]["leaves"]]
        else:
            tensor,letterbox=canon.detector_input(rgb,320)
            raw=cached_mobile(db,detector_sha,tensor).reshape(1,5,2100)
            contract=data["manifest"]["detector"]
            boxes=[box[:4] for box in canon.decode_detections(raw,letterbox,rgb.shape[1],rgb.shape[0],contract["score_threshold"],contract["nms_iou"],contract["max_detections"])[:6]]
        assert len(boxes)==len(bags224[path]["center"])
        pairs[path]=dict(center=[],full=[])
        for leaf_index,box in enumerate(boxes):
            native_sides.append(min(box[2]-box[0],box[3]-box[1]))
            for view,preprocess in [("center",canon.classifier_input),("full",classifier_letterbox_input)]:
                low=preprocess(rgb,224,box)
                old_key=descriptors224[bags224[path][view][leaf_index]]["input_sha256"]
                assert pixel_key(low)==old_key,"Detector/source geometry changed from verified 224px inputs"
                high=preprocess(rgb,448,box)
                key=high_resolution_key(high)
                work.setdefault(key,dict(path=path,box=list(box),view=view,input_sha256=key,parent_224_input_sha256=old_key,file_sha256=sources[path]))
                pairs[path][view].append(key)
    db.close()
    keys=sorted(work);indices={key:i for i,key in enumerate(keys)}
    bags={path:{view:[indices[key] for key in ids] for view,ids in pair.items()} for path,pair in pairs.items()}
    cnn,architecture,normalization=original_cnn(p);cnn=cnn.cuda().eval()
    donor=p/"phase49_partial_cnn/prefix"
    donor_complete=read(donor/"complete.json")
    assert donor_complete["protocol_sha256"]==digest(donor/"protocol.json")
    assert prefix_identity(cnn)==donor_complete["prefix_state_sha256"]
    old_shape=np.load(donor/"features.npy",mmap_mode="r").shape[1:]
    shape=(old_shape[0],old_shape[1]*2,old_shape[2]*2)
    protocol=dict(source_sha256=digest(__file__),input_size=448,architecture=architecture,
                  parent_224_feature_complete_sha256=digest(original/"complete.json"),
                  original_model_sha256=digest(p/"export_cnn_safe/pytorch_bundle.pt"),
                  prefix_state_sha256=prefix_identity(cnn),shape=[len(keys),*shape],
                  inputs=[work[key] for key in keys],runtime=runtime_identity(),
                  preprocessing=preprocessing_contract(Path(__file__).parents[2],"whole_dino"),
                  scope="One publisher primary-image label per actual ROI bag; no clinical per-leaf labels",
                  representation="Original spatial CNN before final two stages, freshly encoded from canonical 448px crops of original files",
                  old_224_encoder_inputs_reused=0,new_448_encoder_inputs=len(keys),
                  original_nine_logits_unchanged=True,tf32=False,dtype="float32",
                  ad_independent=False,olid_independent=False)
    freeze(out/"protocol.json",protocol)
    freeze(out/"bags.json",bags)
    progress=out/"progress.json";target=out/"prefix.npy";reference_path=out/"original_logits.npy"
    saved=read(progress) if progress.exists() else dict(completed=0)
    if progress.exists():assert saved["protocol_sha256"]==digest(out/"protocol.json")
    assert target.exists() or shutil.disk_usage(p).free>len(keys)*int(np.prod(shape))*4+(1<<30)
    values=np.lib.format.open_memmap(target,mode="r+" if target.exists() else "w+",dtype=np.float32,shape=(len(keys),*shape))
    references=np.lib.format.open_memmap(reference_path,mode="r+" if reference_path.exists() else "w+",dtype=np.float32,shape=(len(keys),2))
    normalization={name:value.cuda() for name,value in normalization.items()}
    captured=[]
    hook=cnn.blocks[prefix_cut(cnn)-1].register_forward_hook(lambda module,args,output:captured.append(output))
    try:
        with torch.inference_mode():
            for start in range(saved["completed"],len(keys),64):
                part=keys[start:start+64];batch=[]
                for key in part:
                    row=work[key]
                    assert digest(row["path"])==row["file_sha256"]
                    with Image.open(row["path"]) as image:rgb=np.asarray(image.convert("RGB"))
                    tensor=(canon.classifier_input if row["view"]=="center" else classifier_letterbox_input)(rgb,448,row["box"])
                    assert high_resolution_key(tensor)==key
                    batch.append(tensor)
                x=torch.from_numpy(np.concatenate(batch)).cuda().permute(0,3,1,2)
                rgb=(x*normalization["canonical_std"]+normalization["canonical_mean"]-normalization["cnn_mean"])/normalization["cnn_std"]
                captured.clear();raw=cnn(rgb)
                assert len(captured)==1 and tuple(captured[0].shape[1:])==shape
                values[start:start+len(part)]=captured[0].cpu().numpy()
                references[start:start+len(part)]=raw.cpu().numpy()
                values.flush();references.flush()
                saved=dict(completed=start+len(part),protocol_sha256=digest(out/"protocol.json"))
                temporary=progress.with_suffix(".tmp");temporary.write_text(json.dumps(saved));temporary.replace(progress)
                print(json.dumps(dict(stage="new_448_cnn_inputs",**saved,total=len(keys))),flush=True)
    finally:hook.remove()
    gpu=torch.empty((len(keys),*shape),dtype=torch.float32,device="cuda");maximum=0.
    with torch.inference_mode():
        for start in range(0,len(keys),64):
            prefix=torch.from_numpy(np.array(values[start:start+64],copy=True)).cuda()
            gpu[start:start+len(prefix)]=prefix
            actual=cnn.forward_head(cnn.blocks[prefix_cut(cnn):](prefix)).cpu().numpy()
            maximum=max(maximum,float(np.abs(actual-references[start:start+len(prefix)]).max()))
            assert maximum<.002,"High-resolution prefix does not restore the original CNN"
    report=dict(protocol_sha256=digest(out/"protocol.json"),prefix_sha256=digest(target),
                original_logits_sha256=digest(reference_path),bags_sha256=digest(out/"bags.json"),
                all_inputs_audited=len(keys),maximum_original_logit_error=maximum,
                new_encoder_inputs=len(keys),old_224_encoder_inputs_reused=0,
                native_roi_short_side_quantiles=np.quantile(native_sides,[0,.25,.5,.75,1]).tolist())
    freeze(out/"complete.json",report)
    print(json.dumps(dict(stage="high_resolution_cache_ready",**report)),flush=True)
    return cnn,gpu,bags,report
