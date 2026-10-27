"""Continue the high-performing auxiliary expert; select actual validation verdicts."""

from pathlib import Path
import argparse
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
import timm

from campaign_cnn_fusion import calibrate
from diagnostic_bag_cache_v2 import prepare_tokens
from continued_diagnostic_expert import ContinuedDiagnosticExpert
from partial_dino_healthy import PartialDinoHealthyTail, healthy_margin
from opened_context_data import load, read
from opened_guard_screen import OpenedGuardScreen
from permissive_encoder import digest
from phase2_select import retain
from phase5_scene_ablation import photo_metrics
from phase83_eligible_operating_points import safety as cohort_safety
from phase85_population_sft import evaluation_data
from phase87_diagnostic_bag_features import prepare
from phase88_diagnostic_bag_head import route_paired
from population_siglip_tail import freeze
from source_balance import source_class_group_weights
from training_support import atomic_save, capture_rng, restore_rng


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    p = Path(parser.parse_args().campaign)
    out = p / "phase95_corrected_continued_expert"
    out.mkdir(exist_ok=True)
    if (out/"result.json").exists():
        result = read(out/"result.json")
        assert result["source_sha256"] == digest(__file__)
        print(json.dumps(dict(completed=True, passed=result["passed"])), flush=True)
        return
    assert read(p/"phase88_diagnostic_bag_head/result.json")["passing"] == 0
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    timm.layers.set_fused_attn(False)
    torch.manual_seed(1989)
    started = time.monotonic()
    data = load(p)
    values, bags, metadata = prepare(p, data)
    entries, records, packets, before, previous = evaluation_data(p, data)
    rows = read(p/"phase84_population_head/fitting_rows.json")
    active = [i for i,r in enumerate(rows) if r["active"] and r["path"] in bags and bags[r["path"]]["center"]]
    val = np.array([i for i in active if rows[i]["validation"]])
    val_inputs = {key for i in val for view in bags[rows[i]["path"]].values() for key in view}
    excluded = {rows[i]["group"] for i in active if not rows[i]["validation"] and
                any(key in val_inputs for view in bags[rows[i]["path"]].values() for key in view)}
    train = np.array([i for i in active if not rows[i]["validation"] and rows[i]["group"] not in excluded])
    assert not ({rows[i]["group"] for i in train} & {rows[i]["group"] for i in val})
    assert not ({key for i in train for view in bags[rows[i]["path"]].values() for key in view} & val_inputs)
    backbone, prefix, lookup, prefix_metadata = prepare_tokens(p, values, bags, rows)
    warm = p/"phase55_dino_continuation/head/candidate.pt"
    state = torch.load(warm, weights_only=True, map_location="cpu")
    model = ContinuedDiagnosticExpert(backbone, state).cuda()
    calibration = state["calibration"]
    # Verify the new wrapper against the restored previously qualified expert.
    selected = sorted(lookup)
    cnn = torch.from_numpy(values[selected,384:].copy()).cuda()
    model.eval()
    reference = PartialDinoHealthyTail(backbone).cuda().eval()
    reference.load_state_dict(state["model"],strict=True)
    with torch.inference_mode():
        error = float((model(prefix[:32],cnn[:32])-healthy_margin(reference(prefix[:32]))).abs().max())
    assert error < 0.0001, ("Continued expert restore mismatch",error)
    del reference
    del backbone
    torch.cuda.empty_cache()
    protocol = dict(
        source_sha256=digest(__file__), tail_source_sha256=digest(Path(__file__).with_name("continued_diagnostic_expert.py")),
        feature_cache=metadata, prefix_cache=prefix_metadata, warm_head_sha256=digest(warm),
        fitting_rows_sha256=digest(p/"phase84_population_head/fitting_rows.json"),
        train_images=len(train), validation_images=len(val), seed=1989,
        hypothesis="Continue the previously qualified Healthy expert rather than restart a binary head; balanced supervised loss and actual routed validation verdicts address rejection without changing disease outputs",
        maximum_epochs=12, minimum_epochs=4, patience=4, batch_images=16,
        head_learning_rate=0.0001, suffix_learning_rate=0.00001,
        negative_loss_weight=1.0, weight_decay=0.01,
        supervision="Publisher primary-image bag label; mean real ROI logits separately for center/full views; Healthy minimum and non-Healthy maximum view",
        fixed_family_fold="sha256 phase80:1807:group modulo4; validation unchanged",
        copied_suffix_initial_feature_maximum_error=error,
        original_nine_disease_weights_changed=False, original_field_branch_changed=False,
        original_recognition_and_safety_gates_unchanged=True,
        ad_independent=False, olid_independent=False,
        stopping="One continued expert, at most twelve epochs and four stale routed-validation epochs. Keep the original auxiliary calibration and .95 threshold. Select by false-Healthy constraints then actual Healthy verdicts, not BCE or AUC. Stop before export if any recognition gate fails.",
    )
    freeze(out/"protocol.json", protocol)
    # Compact lookup contains only actual inputs, never padding-as-evidence.
    tables = []
    for row in rows:
        bag = bags.get(row["path"], {})
        tables.append([[lookup[key] for key in bag.get(view,[]) if key in lookup] for view in ("center","full")])
    truth = torch.tensor([r["y"] for r in rows],dtype=torch.float32,device="cuda")
    sampling = torch.from_numpy(source_class_group_weights(rows,train))
    vw = source_class_group_weights(rows,val)
    val_weight = torch.tensor(vw,dtype=torch.float32,device="cuda")

    def forward(ids):
        flat, groups = [], []
        for batch_index, index in enumerate(ids):
            for view, indices in enumerate(tables[index]):
                assert indices
                flat.extend(indices)
                groups.extend([2*batch_index+view]*len(indices))
        unique, inverse = torch.tensor(flat,device="cuda").unique(return_inverse=True)
        raw = model(prefix[unique],cnn[unique])[inverse]
        groups = torch.tensor(groups,device="cuda")
        totals = raw.new_zeros(2*len(ids)).index_add(0,groups,raw)
        counts = torch.bincount(groups,minlength=2*len(ids)).clamp_min(1)
        return (totals/counts).reshape(len(ids),2)

    optimizer = torch.optim.AdamW([
        dict(params=list(model.blocks.parameters())+list(model.norm.parameters()),lr=1e-5),
        dict(params=model.head.parameters(),lr=1e-4)],weight_decay=.01)
    path_items={r["path"]:(r,*entries[id(r)]) for values0 in list(data["base"].values())+[data["internal"]] for r in values0}
    path_items.update({b["path"]:(b,packets[b["id"]],r) for b,r in zip(before,previous)})
    def verdict_score(z, loss):
        probability=1/(1+np.exp(-np.clip(z.cpu().numpy()*calibration[0]+calibration[1],-80,80)))
        false_by_source={}
        olid_healthy, old_healthy = 0,0
        for offset,index in enumerate(val):
            row=rows[index]; original,packet,retained=path_items[row["path"]]
            pair=probability[offset]
            revised=route_paired(original,retained,packet,float(pair.min()),float(pair.max()),.95,data["manifest"])
            was_h=retained["app_replay"]["kind"]=="HEALTHY"
            is_h=revised["app_replay"]["kind"]=="HEALTHY"
            if row["y"]==0:
                false_by_source[row["source"]]=false_by_source.get(row["source"],0)+int(is_h)-int(was_h)
            elif row["source"]=="olid_tomato":
                olid_healthy+=int(is_h)
            else:
                old_healthy+=int(is_h)
        violations=sum(max(0,value) for value in false_by_source.values())
        return (-violations,olid_healthy,old_healthy,-loss)
    history, best, stale, start_epoch = [], None, 0, 0
    if (out/"last.pt").exists():
        resumed = torch.load(out/"last.pt",weights_only=True,map_location="cpu")
        assert resumed["protocol_sha256"] == digest(out/"protocol.json")
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        restore_rng(resumed["rng"])
        history,best,stale,start_epoch = resumed["history"],resumed["best"],resumed["stale"],resumed["epoch"]+1
    for epoch in range(start_epoch,12):
        if epoch>=4 and stale>=4:
            break
        model.train()
        order = torch.multinomial(sampling,len(train),replacement=True,
                                  generator=torch.Generator().manual_seed(198900+epoch))
        for batch in order.split(16):
            ids = train[batch.numpy()]
            z = forward(ids)
            raw = torch.where(truth[ids]==1,z.min(1).values,z.max(1).values)
            loss = (F.binary_cross_entropy_with_logits(raw,truth[ids],reduction="none")*
                    torch.ones_like(truth[ids])).mean()
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            z = torch.cat([forward(ids) for ids in np.array_split(val,max(1,(len(val)+15)//16))])
            raw = torch.where(truth[val]==1,z.min(1).values,z.max(1).values)
            score = float((F.binary_cross_entropy_with_logits(raw,truth[val],reduction="none")*
                           torch.ones_like(truth[val])*val_weight).sum())
        weights = {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
        decision=verdict_score(z,score)
        if best is None or decision > tuple(best):
            best,stale=decision,0
            atomic_save(dict(model=weights,epoch=epoch,protocol_sha256=digest(out/"protocol.json")),out/"best.pt")
        else:
            stale+=1
        history.append(dict(epoch=epoch,validation_loss=score,routed_selection=list(decision)))
        atomic_save(dict(model=weights,optimizer=optimizer.state_dict(),rng=capture_rng(),
                         epoch=epoch,history=history,best=best,stale=stale,
                         protocol_sha256=digest(out/"protocol.json")),out/"last.pt")
        print(json.dumps(dict(stage="continued_diagnostic_expert",**history[-1],stale=stale)),flush=True)
    state = torch.load(out/"best.pt",weights_only=True,map_location="cpu")
    assert state["protocol_sha256"] == digest(out/"protocol.json")
    model.load_state_dict(state["model"]); model.eval()
    with torch.inference_mode():
        z_val = torch.cat([forward(ids) for ids in np.array_split(val,max(1,(len(val)+15)//16))]).cpu().numpy()
        leaf_logits = torch.cat([model(prefix[start:start+64],cnn[start:start+64])
                                 for start in range(0,len(prefix),64)]).cpu().numpy()
    # Preserve the original qualified expert calibration used during selection.
    probabilities={}
    for path,bag in bags.items():
        if not bag["center"] or any(key not in lookup for view in bag.values() for key in view):
            continue
        raw = np.array([leaf_logits[[lookup[key] for key in bag[view]]].mean() for view in ("center","full")])
        probability=1/(1+np.exp(-np.clip(raw*calibration[0]+calibration[1],-80,80)))
        probabilities[path]=dict(positive=float(probability.min()),negative=float(probability.max()))
    base_kind={r["path"]:r["app_replay"]["kind"] for values0 in list(data["base"].values())+[data["internal"]] for r in values0}
    base_kind.update({r["path"]:r["app_replay"]["kind"] for r in before})
    negatives=[probabilities[rows[i]["path"]]["positive"] for i in val if rows[i]["y"]==0 and base_kind[rows[i]["path"]]!="HEALTHY"]
    minimum=.95  # Frozen across all validation epoch comparisons.
    def revise(row):
        packet,retained=entries[id(row)]
        if row["path"] not in probabilities:
            return retained
        z=probabilities[row["path"]]
        return route_paired(row,retained,packet,z["positive"],z["negative"],minimum,data["manifest"])
    screen=OpenedGuardScreen(data).evaluate(revise)
    screen["checks"].pop("original_disease_and_healthy_exact")
    screen["checks"]["original_disease_exact"]=all(revise(r)["app_replay"]==r["app_replay"] for values0 in list(data["base"].values())+[data["internal"]] for r in values0 if r["app_replay"]["kind"]=="DISEASE")
    screen["passed"]=all(screen["checks"].values())
    after=[]
    for b,r in zip(before,previous):
        z=probabilities.get(b["path"])
        after.append(r if z is None else route_paired(b,r,packets[b["id"]],z["positive"],z["negative"],minimum,data["manifest"]))
    vi=[i for i,r in enumerate(rows[-len(records):]) if r["validation"]]
    vm=photo_metrics([after[i] for i in vi]); metrics=photo_metrics(after); bm=photo_metrics(before)
    for rr,mm in [(after,metrics),(before,bm)]:
        mm["unsupported_tomato"]=photo_metrics([r for r in rr if r["label"]=="Other"])["all"]
    safe,vs=cohort_safety([records[i] for i in vi],[before[i] for i in vi],[after[i] for i in vi])
    all_safe,alls=cohort_safety(records,before,after)
    checks=dict(screen["checks"],consumed_validation_healthy=vm["healthy"]["correct_rate"]>=.7,
                consumed_validation_safety=safe,consumed_all_safety=all_safe,
                consumed_all_retention=retain({"olid":metrics},{"olid":bm}))
    result=dict(passed=all(checks.values()),checks=checks,old_screen=screen,consumed_validation=vm,
                consumed_all=metrics,validation_safety=vs,all_safety=alls,
                threshold=minimum,calibration=list(map(float,calibration)),history=history,
                selected_epoch=state["epoch"],checkpoint_sha256=digest(out/"best.pt"),
                protocol_sha256=digest(out/"protocol.json"),source_sha256=digest(__file__),
                selected_input_tables=selected,elapsed_seconds=time.monotonic()-started,
                ad_independent=False,olid_independent=False)
    np.save(out/"leaf_logits.npy",leaf_logits)
    (out/"probabilities.json").write_text(json.dumps(probabilities))
    (out/"result.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(passed=result["passed"],checks=checks,validation_healthy=vm["healthy"])),flush=True)


if __name__=="__main__":
    main()
