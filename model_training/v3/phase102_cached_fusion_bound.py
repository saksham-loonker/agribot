"""Two prespecified complementary-expert blends; optimistic cache-only stop."""
from pathlib import Path
import argparse
import json
import math
import numpy as np
from cached_routing_ceiling import threshold_ceiling
from opened_context_data import load,read
from phase85_population_sft import evaluation_data
from phase84_population_head import population
from phase83_eligible_operating_points import eligible
from permissive_encoder import digest
from population_siglip_tail import freeze


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--campaign",required=True)
    p=Path(parser.parse_args().campaign);out=p/"phase102_cached_fusion_bound";out.mkdir(exist_ok=True)
    names={"cnn":"phase101_cnn_bag_adaptation", "dino":"phase95_corrected_continued_expert", "kernel":"phase93_cached_kernel_expert"}
    protocol=dict(source_sha256=digest(__file__),bound_source_sha256=digest(Path(__file__).with_name("cached_routing_ceiling.py")),
                  parents={name:dict(result_sha256=digest(p/folder/"result.json"),probabilities_sha256=digest(p/folder/"probabilities.json")) for name,folder in names.items()},
                  blends=[["cnn","dino"],["cnn","kernel"]],weights=[.5,.5],
                  aggregation="Equal log-odds blend of each expert's conservative positive/min and negative/max view summaries",
                  role="Optimistic consumed-data impossibility check only, never candidate fitting or independent acceptance. No coefficient or threshold grid training; no new encoder inputs.",
                  stopping="Two blends only; if neither passes even the optimistic 70% Healthy ceiling at unchanged point false-Healthy limits, stop this score family before any restored-model verification or export",
                  ad_independent=False,olid_independent=False,new_encoder_inputs=0)
    freeze(out/"protocol.json",protocol)
    data=load(p);_,records,packets,before,_=evaluation_data(p,data)
    scores={name:read(p/folder/"probabilities.json") for name,folder in names.items()}
    truth=np.array([r["label"]=="Healthy" for r in records]);base=np.array([r["app_replay"]["kind"]=="HEALTHY" for r in before])
    cohorts=[np.ones(len(records),bool)]+[np.array([r["raw_label"]==raw for r in records]) for raw in sorted({r["raw_label"] for r in records if r["label"]!="Healthy"}) if sum(r["raw_label"]==raw and r["label"]!="Healthy" for r in records)>=50]
    allowed=np.array([eligible(r,packets[r["id"]]) and bool(packets[r["id"]]["leaves"]) for r in before])
    controls={}
    for members in protocol["blends"]:
        common=set.intersection(*(set(scores[name]) for name in members))
        blended={}
        for path in common:
            row={}
            for key in ("positive","negative"):
                values=np.clip([scores[name][path][key] for name in members],1e-7,1-1e-7)
                logit=float(np.log(values/(1-values)).mean())
                row[key]=float(1/(1+math.exp(-logit)))
            blended[path]=row
        active=np.array([r["path"] in blended and population(r,packets[r["id"]]) and bool(packets[r["id"]]["leaves"]) for r in before])
        positive=np.array([blended.get(r["path"],{}).get("positive",-1.) for r in before])
        negative=np.array([blended.get(r["path"],{}).get("negative",1.) for r in before])
        result=threshold_ceiling(positive,negative,allowed,active,base,truth,cohorts)
        controls["+".join(members)]=dict(optimistic_best=result,required_healthy=math.ceil(.7*int(truth.sum())),passes_optimistic_ceiling=result["healthy"]>=math.ceil(.7*int(truth.sum())))
    result=dict(protocol_sha256=digest(out/"protocol.json"),controls=controls,passing_optimistic_bound=sum(r["passes_optimistic_ceiling"] for r in controls.values()),new_encoder_inputs=0,ad_independent=False,olid_independent=False,candidate_policy_selected=False)
    (out/"result.json").write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=="__main__":main()
