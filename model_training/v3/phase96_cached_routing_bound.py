"""Stop failed score families using an optimistic exact cached routing bound."""
from pathlib import Path
import argparse
import json
import math
import numpy as np
from cached_routing_ceiling import threshold_ceiling
from opened_context_data import load,read
from permissive_encoder import digest
from phase85_population_sft import evaluation_data
from phase84_population_head import population
from phase83_eligible_operating_points import eligible
from population_siglip_tail import freeze


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--campaign",required=True)
    p=Path(parser.parse_args().campaign);out=p/"phase96_cached_routing_bound";out.mkdir(exist_ok=True)
    data=load(p);_,records,packets,before,_=evaluation_data(p,data)
    controls={"linear":"phase88_diagnostic_bag_head/linear", "mlp":"phase88_diagnostic_bag_head/mlp",
              "corrected_dino_sft":"phase94_corrected_diagnostic_sft",
              "corrected_continued_expert":"phase95_corrected_continued_expert",
              "verified_continuation":"phase97_continued_verified_expert",
              "original_cnn_direction":"phase92_cached_cnn_direction",
              "kernel":"phase93_cached_kernel_expert","logit_reward":"phase99_cached_verdict_reward"}
    protocol=dict(source_sha256=digest(__file__),bound_source_sha256=digest(Path(__file__).with_name("cached_routing_ceiling.py")),
                  controls={name:dict(probabilities_sha256=digest(p/folder/"probabilities.json"),result_sha256=digest(p/folder/"result.json")) for name,folder in controls.items()},
                  source_manifest_sha256=digest(p/"phase24_olid_source/sealed_manifest.json"),
                  invalid_controls=["phase89_diagnostic_bag_sft","phase91_continued_diagnostic_expert"],
                  original_hard_gates_unchanged=True,new_encoder_inputs=0,ad_independent=False,olid_independent=False,
                  role="All consumed labels are used only for optimistic impossibility bounds, never policy fitting or fresh acceptance. Ignore old retention, confidence floor and family bootstrap constraints, making this easier than real acceptance.")
    freeze(out/"protocol.json",protocol)
    truth=np.array([r["label"]=="Healthy" for r in records]);base=np.array([r["app_replay"]["kind"]=="HEALTHY" for r in before])
    allowed=np.array([eligible(r,packets[r["id"]]) and bool(packets[r["id"]]["leaves"]) for r in before])
    cohorts=[np.ones(len(records),bool)]+[np.array([r["raw_label"]==raw for r in records]) for raw in sorted({r["raw_label"] for r in records if r["label"]!="Healthy"}) if sum(r["raw_label"]==raw and r["label"]!="Healthy" for r in records)>=50]
    result={}
    required=math.ceil(.7*int(truth.sum()))
    for name,folder in controls.items():
        scores=read(p/folder/"probabilities.json")
        active=np.array([r["path"] in scores and population(r,packets[r["id"]]) and bool(packets[r["id"]]["leaves"]) for r in before])
        positive=np.array([scores.get(r["path"],{}).get("positive",-1.) for r in before])
        negative=np.array([scores.get(r["path"],{}).get("negative",1.) for r in before])
        best=threshold_ceiling(positive,negative,allowed,active,base,truth,cohorts)
        result[name]=dict(optimistic_best=best,required_healthy=required,passes_optimistic_ceiling=best["healthy"]>=required)
    report=dict(protocol_sha256=digest(out/"protocol.json"),controls=result,
                passing_optimistic_bound=sum(r["passes_optimistic_ceiling"] for r in result.values()),
                ad_independent=False,olid_independent=False,candidate_policy_selected=False,new_encoder_inputs=0)
    (out/"result.json").write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=="__main__":main()
