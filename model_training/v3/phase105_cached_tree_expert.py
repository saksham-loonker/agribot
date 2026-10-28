"""One source-balanced boosted-tree control over existing image representations.

No encoder inference, threshold search or clinical per-detection labels. All
scores are image-level; unchanged routing protects named disease decisions.
"""
from pathlib import Path
import argparse
import json
import os
import time
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from opened_context_data import load,read
from opened_guard_screen import OpenedGuardScreen
from permissive_encoder import digest
from phase2_select import retain
from phase5_scene_ablation import photo_metrics
from phase22_accept import paired_interval
from phase81_consumed_olid_features import prepare as prepare_whole
from phase83_eligible_operating_points import safety
from phase85_population_sft import evaluation_data
from phase87_diagnostic_bag_features import prepare as prepare_leaf
from phase88_diagnostic_bag_head import route_paired
from population_siglip_tail import freeze
from source_balance import source_class_group_weights


def atomic_model(model,path):
    temporary=path.with_suffix(".tmp")
    joblib.dump(model,temporary)
    os.replace(temporary,path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--campaign",required=True)
    p=Path(parser.parse_args().campaign);out=p/"phase105_cached_tree_expert";out.mkdir(exist_ok=True)
    if (out/"result.json").exists():
        result=read(out/"result.json")
        assert result["source_sha256"]==digest(__file__)
        print(json.dumps(dict(completed=True,passed=result["passed"])),flush=True);return
    started=time.monotonic();data=load(p)
    leaf,bags,leaf_metadata=prepare_leaf(p,data)
    whole,whole_metadata=prepare_whole(p,data)
    entries,records,packets,before,previous=evaluation_data(p,data)
    rows=read(p/"phase84_population_head/fitting_rows.json")
    # Predict all available images, including mixed/unsupported development
    # cases. Ground-truth categories must never decide inference eligibility.
    paths=sorted(path for path,pair in bags.items() if pair["center"] and path in whole)
    path_index={path:i for i,path in enumerate(paths)}
    def encode(path):
        pair=[]
        for name in ("center","full"):
            value=leaf[bags[path][name]].mean(0)
            pair.append(np.concatenate([v/max(float(np.linalg.norm(v)),1e-8) for v in (value[:384],value[384:])]))
        sig=whole[path];sig=sig/max(float(np.linalg.norm(sig)),1e-8)
        return np.concatenate([(pair[0]+pair[1])/2,np.abs(pair[0]-pair[1]),sig,[len(bags[path]["center"])/6]]).astype(np.float32)
    features=np.stack([encode(path) for path in paths])
    active=[i for i,r in enumerate(rows) if r["active"] and r["path"] in path_index]
    val=np.array([i for i in active if rows[i]["validation"]])
    val_inputs={key for i in val for view in bags[rows[i]["path"]].values() for key in view}
    excluded={rows[i]["group"] for i in active if not rows[i]["validation"] and any(key in val_inputs for view in bags[rows[i]["path"]].values() for key in view)}
    excluded.update(read(p/"phase88_diagnostic_bag_head/protocol.json")["overlap_training_groups_quarantined"])
    train=np.array([i for i in active if not rows[i]["validation"] and rows[i]["group"] not in excluded])
    assert not ({rows[i]["group"] for i in train}&{rows[i]["group"] for i in val})
    assert not ({key for i in train for view in bags[rows[i]["path"]].values() for key in view}&val_inputs)
    train_x=features[[path_index[rows[i]["path"]] for i in train]]
    val_x=features[[path_index[rows[i]["path"]] for i in val]]
    y=np.array([r["y"] for r in rows]);weights=source_class_group_weights(rows,train)*len(train)
    protocol=dict(source_sha256=digest(__file__),leaf_metadata=leaf_metadata,whole_metadata=whole_metadata,
                  fitting_rows_sha256=digest(p/"phase84_population_head/fitting_rows.json"),
                  feature_bytes_sha256=__import__("hashlib").sha256(features.tobytes()).hexdigest(),
                  feature_paths=paths,feature_dimension=features.shape[1],sklearn_version=sklearn.__version__,
                  train_images=len(train),validation_images=len(val),seed=2105,
                  model="HistGradientBoostingClassifier",maximum_iterations=300,checkpoint_every=25,
                  minimum_iterations=100,stale_validation_patience=100,learning_rate=.05,
                  maximum_depth=3,maximum_leaves=15,minimum_leaf_images=12,bins=63,l2=1.,
                  supervision="One publisher primary-image label; pooled paired real-ROI DINO/CNN means/differences plus original whole SigLIP; source/class/family weights are not inference features",
                  selection="No newly Healthy diseased validation image; then actual consumed OLID Healthy and old Healthy counts; fixed .95 admission and .001 veto",
                  calibration="Raw supervised tree probability; no threshold or calibration search",
                  new_encoder_inputs=0,old_disease_weights_changed=False,ad_independent=False,olid_independent=False,
                  stopping="Exactly one fixed tree configuration; stop after 100 stale validation iterations after 100, or at 300. No hyperparameter search or export before all original recognition/safety gates pass.")
    freeze(out/"protocol.json",protocol)
    model=HistGradientBoostingClassifier(max_iter=25,learning_rate=.05,max_depth=3,max_leaf_nodes=15,
          min_samples_leaf=12,max_bins=63,l2_regularization=1.,early_stopping=False,warm_start=True,random_state=2105)
    items={r["path"]:(r,*entries[id(r)]) for vv in list(data["base"].values())+[data["internal"]] for r in vv}
    items.update({b["path"]:(b,packets[b["id"]],a) for b,a in zip(before,previous)})
    history=[];best=None;last_improvement=0;start=25
    if (out/"progress.json").exists():
        progress=read(out/"progress.json")
        assert progress["protocol_sha256"]==digest(out/"protocol.json")
        assert progress["model_sha256"]==digest(out/"last.joblib")
        model=joblib.load(out/"last.joblib") # Only our locally produced, SHA-verified checkpoint.
        history,best,last_improvement,start=progress["history"],progress["best"],progress["last_improvement"],progress["iterations"]+25
    for iterations in range(start,301,25):
        model.max_iter=iterations;model.fit(train_x,y[train],sample_weight=weights)
        probability=model.predict_proba(val_x)[:,1]
        new_false=0;olid_h=0;old_h=0
        for offset,index in enumerate(val):
            row=rows[index];original,packet,retained=items[row["path"]]
            score=float(probability[offset])
            revised=route_paired(original,retained,packet,score,score,.95,data["manifest"])
            ish=revised["app_replay"]["kind"]=="HEALTHY";wash=retained["app_replay"]["kind"]=="HEALTHY"
            if row["y"]==0:new_false+=int(ish and not wash)
            elif row["source"]=="olid_tomato":olid_h+=int(ish)
            else:old_h+=int(ish)
        decision=(-new_false,olid_h,old_h)
        if best is None or decision>tuple(best):
            best=decision;last_improvement=iterations;atomic_model(model,out/"best.joblib")
            (out/"best.json").write_text(json.dumps(dict(iterations=iterations,decision=list(decision),sha256=digest(out/"best.joblib"))))
        history.append(dict(iterations=iterations,routed_validation=list(decision)))
        atomic_model(model,out/"last.joblib")
        temporary=out/"progress.tmp"
        temporary.write_text(json.dumps(dict(iterations=iterations,history=history,best=best,last_improvement=last_improvement,
            model_sha256=digest(out/"last.joblib"),protocol_sha256=digest(out/"protocol.json"))))
        temporary.replace(out/"progress.json")
        print(json.dumps(history[-1]),flush=True)
        if iterations>=100 and iterations-last_improvement>=100:break
    selected=read(out/"best.json");assert selected["sha256"]==digest(out/"best.joblib")
    model=joblib.load(out/"best.joblib")
    probabilities=dict(zip(paths,map(float,model.predict_proba(features)[:,1])))
    def revise(row):
        packet,retained=entries[id(row)]
        score=probabilities.get(row["path"])
        return retained if score is None else route_paired(row,retained,packet,score,score,.95,data["manifest"])
    screen=OpenedGuardScreen(data).evaluate(revise)
    screen["checks"].pop("original_disease_and_healthy_exact")
    screen["checks"]["original_disease_exact"]=all(revise(r)["app_replay"]==r["app_replay"] for vv in list(data["base"].values())+[data["internal"]] for r in vv if r["app_replay"]["kind"]=="DISEASE")
    screen["passed"]=all(screen["checks"].values())
    after=[]
    for b,a in zip(before,previous):
        score=probabilities.get(b["path"])
        after.append(a if score is None else route_paired(b,a,packets[b["id"]],score,score,.95,data["manifest"]))
    assert [r["path"] for r in rows[-len(records):]]==[r["path"] for r in records]
    vi=[i for i,r in enumerate(rows[-len(records):]) if r["validation"]]
    vr,vb,va=[[values[i] for i in vi] for values in (records,before,after)]
    vm=photo_metrics(va);metrics=photo_metrics(after);bm=photo_metrics(before)
    for rr,mm in [(after,metrics),(before,bm)]:mm["unsupported_tomato"]=photo_metrics([r for r in rr if r["label"]=="Other"])["all"]
    val_safe,vs=safety(vr,vb,va);all_safe,alls=safety(records,before,after)
    gain=paired_interval(vr,vb,va,lambda r:r["label"]=="Healthy",lambda r:int(r["app_replay"]["kind"]=="HEALTHY"))
    checks=dict(screen["checks"],consumed_validation_healthy=vm["healthy"]["correct_rate"]>=.7,
                consumed_full_healthy=metrics["healthy"]["correct_rate"]>=.7,
                consumed_validation_gain=gain is not None and gain["point"]>=.1 and gain["bootstrap_two_sided_95"][0]>0,
                consumed_validation_safety=val_safe,consumed_full_safety=all_safe,
                consumed_full_retention=retain({"olid":metrics},{"olid":bm}))
    result=dict(passed=all(checks.values()),checks=checks,old_screen=screen,consumed_validation=vm,consumed_all=metrics,
                validation_safety=vs,all_safety=alls,validation_gain=gain,selected=selected,history=history,
                source_sha256=digest(__file__),protocol_sha256=digest(out/"protocol.json"),elapsed_seconds=time.monotonic()-started,
                new_encoder_inputs=0,ad_independent=False,olid_independent=False)
    (out/"probabilities.json").write_text(json.dumps({path:dict(positive=score,negative=score) for path,score in probabilities.items()}))
    (out/"result.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(passed=result["passed"],checks=checks,validation_healthy=vm["healthy"])),flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=4):main()
