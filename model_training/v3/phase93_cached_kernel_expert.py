"""One bounded nonlinear kernel control, using exact existing features only."""
from pathlib import Path
import argparse
import json
import math
import time
import numpy as np
import torch

from campaign_cnn_fusion import calibrate
from opened_context_data import load,read
from permissive_encoder import digest
from phase81_consumed_olid_features import prepare as whole_features
from phase87_diagnostic_bag_features import prepare as leaf_features
from phase84_population_head import population
from phase85_population_sft import evaluation_data
from population_siglip_tail import freeze
from source_balance import source_class_group_weights


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--campaign",required=True)
    p=Path(parser.parse_args().campaign);out=p/"phase93_cached_kernel_expert";out.mkdir(exist_ok=True)
    started=time.monotonic(); data=load(p)
    leaf,bags,leaf_metadata=leaf_features(p,data)
    whole,whole_metadata=whole_features(p,data)
    rows=read(p/"phase84_population_head/fitting_rows.json")
    active=[i for i,r in enumerate(rows) if r["active"] and r["path"] in bags and bags[r["path"]]["center"]]
    val=np.array([i for i in active if rows[i]["validation"]])
    validation_inputs={key for i in val for indices in bags[rows[i]["path"]].values() for key in indices}
    excluded={rows[i]["group"] for i in active if not rows[i]["validation"] and any(key in validation_inputs for indices in bags[rows[i]["path"]].values() for key in indices)}
    train=np.array([i for i in active if not rows[i]["validation"] and rows[i]["group"] not in excluded])
    assert not ({rows[i]["group"] for i in train}&{rows[i]["group"] for i in val})
    def encode(path):
        pair=bags[path]; views=[]
        for name in ("center","full"):
            value=leaf[pair[name]].mean(0)
            dino=value[:384];cnn=value[384:];sig=whole[path]
            views.append(np.concatenate([v/max(np.linalg.norm(v),1e-8) for v in (dino,cnn,sig)]))
        return views
    x=np.array([encode(rows[i]["path"]) for i in active],np.float64)
    active_lookup={index:i for i,index in enumerate(active)}
    train_ids=np.array([2*active_lookup[i]+view for i in train for view in range(2)])
    val_ids=np.array([2*active_lookup[i]+view for i in val for view in range(2)])
    x=x.reshape(-1,x.shape[-1]);basis=x[train_ids]
    distance=np.maximum(0,(basis*basis).sum(1)[:,None]+(basis*basis).sum(1)[None]-2*basis@basis.T)
    bandwidth=float(np.median(distance[distance>1e-10]));assert bandwidth>0
    weights=np.repeat(source_class_group_weights(rows,train),2)/2
    weights=weights*len(weights)
    root=np.sqrt(weights)
    regularization=1.0
    protocol=dict(source_sha256=digest(__file__),leaf_metadata=leaf_metadata,whole_metadata=whole_metadata,
                  fitting_rows_sha256=digest(p/"phase84_population_head/fitting_rows.json"),
                  train_images=len(train),validation_images=len(val),
                  kernel="exp(-squared_distance/median_positive_training_distance)",bandwidth=bandwidth,
                  regularization=regularization,modalities="Equal unit-normalized DINO/CNN primary-ROI pooled views plus original whole SigLIP",
                  supervision="Primary image labels, actual ROI bags; two views share image weight",
                  sampling="Equal binary class, source, inferred family mass; cost 1, no RL reward",
                  stopping="Exactly one training-only bandwidth and fixed regularization; no grid or encoder inference. Cache score ranking and stop unless cheap recognition gates can pass.",
                  new_encoder_inputs=0,encoder_weights_changed=False,old_disease_weights_changed=False,
                  ad_independent=False,olid_independent=False)
    freeze(out/"protocol.json",protocol)
    if (out/"result.json").exists():
        assert read(out/"result.json")["protocol_sha256"]==digest(out/"protocol.json")
        return
    target=np.repeat([2*rows[i]["y"]-1 for i in train],2)
    kernel=np.exp(-distance/bandwidth)
    coefficients=root*np.linalg.solve(kernel*root[:,None]*root[None,:]+np.eye(len(basis))*regularization,root*target)
    def predict(value):
        d=np.maximum(0,(value*value).sum(1)[:,None]+(basis*basis).sum(1)[None]-2*value@basis.T)
        return np.exp(-d/bandwidth)@coefficients
    validation_logits=predict(x[val_ids])
    calibration=calibrate(validation_logits,np.repeat([rows[i]["y"] for i in val],2),np.repeat(source_class_group_weights(rows,val),2)/2)
    probabilities={}
    for path,pair in bags.items():
        if not pair["center"] or path not in whole: continue
        raw=predict(np.array(encode(path),np.float64))
        probability=1/(1+np.exp(-np.clip(raw*calibration[0]+calibration[1],-80,80)))
        probabilities[path]=dict(positive=float(probability.min()),negative=float(probability.max()))
    np.savez(out/"kernel.npz",basis=basis,coefficients=coefficients,bandwidth=bandwidth,calibration=calibration)
    (out/"probabilities.json").write_text(json.dumps(probabilities))
    result=dict(protocol_sha256=digest(out/"protocol.json"),kernel_sha256=digest(out/"kernel.npz"),
                probabilities_sha256=digest(out/"probabilities.json"),calibration=list(map(float,calibration)),
                paths=len(probabilities),elapsed_seconds=time.monotonic()-started,
                new_encoder_inputs=0,ad_independent=False,olid_independent=False)
    (out/"result.json").write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=="__main__": main()
