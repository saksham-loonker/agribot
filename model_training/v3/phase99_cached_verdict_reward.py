"""One cached-kernel logit-reward control with explicit false-Healthy constraint.

Rewards come from existing supervised labels and the app's eligibility rules.
This is differentiable constrained supervised optimization, not clinical RLHF.
"""
from pathlib import Path
import argparse
import json
import math
import time
import numpy as np
import torch
from torch.nn import functional as F

from opened_context_data import load,read
from permissive_encoder import digest
from phase81_consumed_olid_features import prepare as whole_features
from phase87_diagnostic_bag_features import prepare as leaf_features
from phase85_population_sft import evaluation_data
from phase88_diagnostic_bag_head import route_paired
from opened_guard_screen import OpenedGuardScreen
from phase83_eligible_operating_points import safety as cohort_safety
from phase5_scene_ablation import photo_metrics
from phase2_select import retain
from source_balance import source_class_group_weights
from population_siglip_tail import freeze
from training_support import atomic_save


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--campaign",required=True)
    p=Path(parser.parse_args().campaign);out=p/"phase99_cached_verdict_reward";out.mkdir(exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    if (out/"result.json").exists():
        assert read(out/"result.json")["source_sha256"]==digest(__file__)
        return
    started=time.monotonic();data=load(p)
    leaf,bags,lm=leaf_features(p,data);whole,wm=whole_features(p,data)
    entries,records,packets,before,previous=evaluation_data(p,data)
    rows=read(p/"phase84_population_head/fitting_rows.json")
    active=[i for i,r in enumerate(rows) if r["active"] and r["path"] in bags and bags[r["path"]]["center"]]
    val=np.array([i for i in active if rows[i]["validation"]])
    val_inputs={key for i in val for view in bags[rows[i]["path"]].values() for key in view}
    excluded={rows[i]["group"] for i in active if not rows[i]["validation"] and any(key in val_inputs for view in bags[rows[i]["path"]].values() for key in view)}
    train=np.array([i for i in active if not rows[i]["validation"] and rows[i]["group"] not in excluded])
    assert not ({rows[i]["group"] for i in train}&{rows[i]["group"] for i in val})
    kernel_path=p/"phase93_cached_kernel_expert/kernel.npz"
    kernel=np.load(kernel_path)
    basis=torch.from_numpy(kernel["basis"].astype(np.float32)).cuda()
    bandwidth=float(kernel["bandwidth"])
    def encode(path):
        views=[]
        for name in ("center","full"):
            value=leaf[bags[path][name]].mean(0)
            views.append(np.concatenate([v/max(np.linalg.norm(v),1e-8) for v in (value[:384],value[384:],whole[path])]))
        return views
    def project(x):
        x=torch.as_tensor(np.array(x),dtype=torch.float32,device="cuda")
        distance=((x*x).sum(1)[:,None]+(basis*basis).sum(1)[None]-2*x@basis.T).clamp_min(0)
        return (-distance/bandwidth).exp()
    training=project([view for i in train for view in encode(rows[i]["path"])])
    validation=project([view for i in val for view in encode(rows[i]["path"])])
    truth=torch.tensor([rows[i]["y"] for i in train],dtype=torch.float32,device="cuda")
    weights=torch.tensor(source_class_group_weights(rows,train),dtype=torch.float32,device="cuda")
    path_items={r["path"]:(r,*entries[id(r)]) for values0 in list(data["base"].values())+[data["internal"]] for r in values0}
    path_items.update({b["path"]:(b,packets[b["id"]],r) for b,r in zip(before,previous)})
    base_h=torch.tensor([path_items[rows[i]["path"]][2]["app_replay"]["kind"]=="HEALTHY" for i in train],device="cuda")
    # Head can abstain from existing Healthy only at the original .001 gate;
    # new Healthy proposals must exceed the original isolated .95 gate.
    high=math.log(.95/.05);low=math.log(.001/.999)
    target_risk=float((weights*(1-truth)*base_h).sum()/(weights*(1-truth)).sum())
    protocol=dict(source_sha256=digest(__file__),kernel_sha256=digest(kernel_path),leaf_cache=lm,whole_cache=wm,
                  fitting_rows_sha256=digest(p/"phase84_population_head/fitting_rows.json"),
                  train_images=len(train),validation_images=len(val),maximum_steps=1000,evaluate_every=25,
                  minimum_steps=200,stale_validation_patience=200,learning_rate=.01,
                  objective="Balanced supervised BCE + logit expected Healthy-verdict reward with projected dual penalties for training Healthy recovery <.85 and disease-to-Healthy risk above original baseline; no changed labels",
                  original_positive_gate=.95,original_negative_gate=.001,soft_route_temperature=.5,
                  selection="Actual routed validation verdicts: source false-Healthy violations first, OLID Healthy next, old Healthy next; no AUC",
                  training_false_healthy_target=target_risk,new_encoder_inputs=0,encoder_weights_changed=False,
                  old_disease_weights_changed=False,ad_independent=False,olid_independent=False,
                  stopping="Exactly one cached reward control; no reward/temperature search. Stop on 200 stale validation steps after 200, or at 1000; no export unless all unchanged recognition/safety checks pass.")
    freeze(out/"protocol.json",protocol)
    original_calibration=kernel["calibration"]
    alpha=torch.nn.Parameter(torch.from_numpy((kernel["coefficients"]*original_calibration[0]).astype(np.float32)).cuda())
    bias=torch.nn.Parameter(torch.tensor(float(original_calibration[1]),device="cuda"))
    optimizer=torch.optim.Adam([alpha,bias],lr=.01)
    dual_h=1.;dual_f=1.;best=None;last_improvement=0;history=[];start=0
    if (out/"last.pt").exists():
        resumed=torch.load(out/"last.pt",weights_only=True,map_location="cpu")
        assert resumed["protocol_sha256"]==digest(out/"protocol.json")
        with torch.no_grad():alpha.copy_(resumed["alpha"]);bias.copy_(resumed["bias"])
        optimizer.load_state_dict(resumed["optimizer"])
        dual_h,dual_f,best,last_improvement,history,start=resumed["dual_h"],resumed["dual_f"],resumed["best"],resumed["last_improvement"],resumed["history"],resumed["step"]+1
    for step in range(start,1001):
        z=(training@alpha+bias).reshape(-1,2)
        robust=torch.where(truth==1,z.min(1).values,z.max(1).values)
        bce=(F.binary_cross_entropy_with_logits(robust,truth,reduction="none")*weights).sum()
        soft_healthy=torch.where(base_h,torch.sigmoid((z.max(1).values-low)/.5),torch.sigmoid((z.min(1).values-high)/.5))
        recall=(weights*truth*soft_healthy).sum()/(weights*truth).sum()
        risk=(weights*(1-truth)*soft_healthy).sum()/(weights*(1-truth)).sum()
        h_constraint=.85-recall;f_constraint=risk-target_risk
        loss=bce+dual_h*h_constraint+dual_f*f_constraint+.0001*(alpha*alpha).sum()
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_([alpha,bias],10.);optimizer.step()
        if step%25:continue
        dual_h=max(0.,min(100.,dual_h+2*float(h_constraint.detach())))
        dual_f=max(0.,min(100.,dual_f+2*float(f_constraint.detach())))
        with torch.inference_mode():probability=(validation@alpha+bias).sigmoid().reshape(-1,2).cpu().numpy()
        false_by_source={};olid_h=0;old_h=0
        for offset,index in enumerate(val):
            row=rows[index];original,packet,retained=path_items[row["path"]]
            pair=probability[offset]
            revised=route_paired(original,retained,packet,float(pair.min()),float(pair.max()),.95,data["manifest"])
            was_h=retained["app_replay"]["kind"]=="HEALTHY";is_h=revised["app_replay"]["kind"]=="HEALTHY"
            if row["y"]==0:false_by_source[row["source"]]=false_by_source.get(row["source"],0)+int(is_h)-int(was_h)
            elif row["source"]=="olid_tomato":olid_h+=int(is_h)
            else:old_h+=int(is_h)
        decision=(-sum(max(0,value) for value in false_by_source.values()),olid_h,old_h)
        checkpoint=dict(alpha=alpha.detach().cpu(),bias=bias.detach().cpu(),step=step,protocol_sha256=digest(out/"protocol.json"))
        if best is None or decision>tuple(best):
            best=decision;last_improvement=step;atomic_save(checkpoint,out/"best.pt")
        history.append(dict(step=step,routed_selection=list(decision),loss=float(loss.detach()),train_recall=float(recall.detach()),train_false_healthy=float(risk.detach()),dual_h=dual_h,dual_f=dual_f))
        atomic_save(dict(checkpoint,optimizer=optimizer.state_dict(),dual_h=dual_h,dual_f=dual_f,best=best,last_improvement=last_improvement,history=history),out/"last.pt")
        print(json.dumps(history[-1]),flush=True)
        if step>=200 and step-last_improvement>=200:break
    selected=torch.load(out/"best.pt",weights_only=True,map_location="cuda")
    with torch.no_grad():alpha.copy_(selected["alpha"]);bias.copy_(selected["bias"])
    probabilities={}
    paths=[rows[i]["path"] for i in active]
    with torch.inference_mode():
        for start in range(0,len(paths),64):
            part=paths[start:start+64]
            probability=(project([view for path in part for view in encode(path)])@alpha+bias).sigmoid().reshape(-1,2).cpu().numpy()
            for path,pair in zip(part,probability):probabilities[path]=dict(positive=float(pair.min()),negative=float(pair.max()))
    def revise(row):
        packet,retained=entries[id(row)]
        z=probabilities.get(row["path"])
        return retained if z is None else route_paired(row,retained,packet,z["positive"],z["negative"],.95,data["manifest"])
    screen=OpenedGuardScreen(data).evaluate(revise)
    screen["checks"].pop("original_disease_and_healthy_exact")
    screen["checks"]["original_disease_exact"]=all(revise(r)["app_replay"]==r["app_replay"] for values0 in list(data["base"].values())+[data["internal"]] for r in values0 if r["app_replay"]["kind"]=="DISEASE")
    screen["passed"]=all(screen["checks"].values())
    after=[]
    for b,r in zip(before,previous):
        z=probabilities.get(b["path"])
        after.append(r if z is None else route_paired(b,r,packets[b["id"]],z["positive"],z["negative"],.95,data["manifest"]))
    vi=[i for i,r in enumerate(rows[-len(records):]) if r["validation"]]
    vm=photo_metrics([after[i] for i in vi]);metrics=photo_metrics(after);bm=photo_metrics(before)
    for rr,mm in [(after,metrics),(before,bm)]:mm["unsupported_tomato"]=photo_metrics([r for r in rr if r["label"]=="Other"])["all"]
    safe,vs=cohort_safety([records[i] for i in vi],[before[i] for i in vi],[after[i] for i in vi])
    all_safe,alls=cohort_safety(records,before,after)
    checks=dict(screen["checks"],consumed_validation_healthy=vm["healthy"]["correct_rate"]>=.7,consumed_validation_safety=safe,consumed_all_safety=all_safe,consumed_all_retention=retain({"olid":metrics},{"olid":bm}))
    result=dict(passed=all(checks.values()),checks=checks,old_screen=screen,consumed_validation=vm,consumed_all=metrics,
                validation_safety=vs,all_safety=alls,selected_step=selected["step"],history=history,
                source_sha256=digest(__file__),protocol_sha256=digest(out/"protocol.json"),checkpoint_sha256=digest(out/"best.pt"),
                elapsed_seconds=time.monotonic()-started,new_encoder_inputs=0,ad_independent=False,olid_independent=False)
    (out/"probabilities.json").write_text(json.dumps(probabilities));(out/"result.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(passed=result["passed"],checks=checks,validation_healthy=vm["healthy"])),flush=True)


if __name__=="__main__":main()
