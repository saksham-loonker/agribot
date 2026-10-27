"""Screen the retained binary CNN's direction using cached normalized features.

These scores are not the original CNN posterior: its feature norm was discarded
by the feature cache. This is only a cheap representation/ranking control.
"""
from pathlib import Path
import argparse
import json
import math
import numpy as np
import torch
from opened_context_data import read
from partial_cnn_cache import original_cnn
from permissive_encoder import digest
from population_siglip_tail import freeze


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--campaign",required=True)
    p=Path(parser.parse_args().campaign); out=p/"phase92_cached_cnn_direction"; out.mkdir(exist_ok=True)
    old=p/"phase29_compact"; new=p/"phase87_diagnostic_bag_features"
    assert digest(old/"features.npy")==read(old/"complete.json")["features_sha256"]
    assert digest(new/"new_features.npy")==read(new/"complete.json")["new_features_sha256"]
    protocol=dict(source_sha256=digest(__file__),old_complete_sha256=digest(old/"complete.json"),
                  new_complete_sha256=digest(new/"complete.json"),new_encoder_inputs=0,weights_changed=False,
                  class_order=["Healthy","NonHealthy"],
                  score="Original CNN Healthy-minus-NonHealthy linear direction on unit-normalized 1024D pooled embeddings; not calibrated probability",
                  ad_independent=False,olid_independent=False)
    freeze(out/"protocol.json",protocol)
    if (out/"result.json").exists():
        assert read(out/"result.json")["protocol_sha256"]==digest(out/"protocol.json")
        return
    torch.set_num_threads(4)
    cnn,_,_=original_cnn(p)
    weight=cnn.classifier.weight.detach().numpy()
    assert weight.shape==(2,1024)
    bank=np.concatenate([np.load(old/"features.npy",mmap_mode="r"),np.load(new/"new_features.npy",mmap_mode="r")])
    raw=(bank[:,384:] @ (weight[0]-weight[1])) / math.sqrt(1024)
    probabilities={}
    for path,pair in read(new/"bags.json").items():
        if not pair["center"]: continue
        scores=np.array([np.mean(raw[pair[view]]) for view in ("center","full")])
        sigmoid=1/(1+np.exp(-scores))
        probabilities[path]=dict(positive=float(sigmoid.min()),negative=float(sigmoid.max()))
    (out/"probabilities.json").write_text(json.dumps(probabilities))
    (out/"result.json").write_text(json.dumps(dict(protocol_sha256=digest(out/"protocol.json"),
        probabilities_sha256=digest(out/"probabilities.json"),paths=len(probabilities),new_encoder_inputs=0),indent=2))
    print(json.dumps(dict(stage="original_cnn_cached_direction",paths=len(probabilities),new_encoder_inputs=0)),flush=True)


if __name__=="__main__": main()
