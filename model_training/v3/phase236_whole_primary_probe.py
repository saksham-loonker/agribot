"""Measure missing whole-primary inputs using exact deployed-model output caches."""
from pathlib import Path
import argparse,hashlib,json,sqlite3,time
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter
from app_eval import canon,classifier_letterbox_input,posterior
from conditional_recovery_policy import recover_only
from opened_context_data import load,read
from phase85_population_sft import evaluation_data
from permissive_encoder import digest
from population_siglip_tail import freeze

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',required=True);p=Path(ap.parse_args().campaign);out=p/'phase236_whole_primary_probe';out.mkdir(exist_ok=True);started=time.monotonic();rows=read(p/'phase84_population_head/fitting_rows.json');selection=read(p/'phase118_conditional_recovery/protocol.json');ids=selection['validation_indices'];assets=p/'export_cnn_safe';manifest=read(assets/'model_manifest.json');member=manifest['classifier']['members'][0];model_path=assets/'models'/member['file'];assert digest(model_path)==member['sha256']=='d70135804cf750341218e89a055370bef46a9b337951897b9f55f2570844ed39';protocol=dict(source_sha256=digest(__file__),manifest_sha256=digest(assets/'model_manifest.json'),classifier_sha256=member['sha256'],selection_sha256=digest(p/'phase118_conditional_recovery/protocol.json'),validation_indices=ids,hypothesis='Existing fallback9-class scores are absent when ROIs are detected. Measure the original deployed classifier on actual whole-primary center/full inputs, rather than misinterpreting missing fallbacks as failed whole predictions.',scope='One fixed whole-primary recovery score=min of center/full Healthy posterior at.95; preserve old whole/genus/compact/suspicion/evidence guards and exact existing verdicts. No fabricated leaf votes or independent repeated frames.',input_cache_key='Exact deployed model SHA+shape+canonical float32 bytes, identical to phase76 ExactMobileCache',healthy_admission=.95,original_gates_unchanged=True,ad_independent=False,olid_independent=False,acceptance_data_used=False,stopping='One fixed whole-primary baseline proposal. Check exact existing output cache before any new input invocation. >=28/40 and zero new falseHealthy before all original full qualification; no parameter grid.')
    freeze(out/'protocol.json',protocol)
    if (out/'result.json').exists():assert read(out/'result.json')['protocol_sha256']==digest(out/'protocol.json');return
    old=sqlite3.connect('file:'+str(p/'phase76_fresh_acceptance/cache/predictions.sqlite3')+'?mode=ro',uri=True);bank=sqlite3.connect(out/'predictions.sqlite');bank.execute('create table if not exists outputs(identity text primary key,values_blob blob not null)');model=None;new=0;hits=0;data=load(p);entries,_,packets,before,previous=evaluation_data(p,data);items={r['path']:(r,*entries[id(r)]) for group in list(data['base'].values())+[data['internal']] for r in group};items.update({r['path']:(r,packets[r['id']],retained) for r,retained in zip(before,previous)});healthy=selection['fixed_healthy_validation_correct'];errors={};reports=[];input_shas=[]
    def score(value):
        nonlocal model,new,hits
        value=np.ascontiguousarray(value,dtype=np.float32);identity=hashlib.sha256(member['sha256'].encode()+str(value.shape).encode()+value.tobytes()).hexdigest()
        for db in (bank,old):
            saved=db.execute('select values_blob from outputs where identity=?',(identity,)).fetchone()
            if saved:hits+=1;return np.frombuffer(saved[0],dtype=np.float32).copy(),identity
        if model is None:model=Interpreter(model_path=str(model_path),num_threads=4);model.allocate_tensors()
        model.set_tensor(model.get_input_details()[0]['index'],value);model.invoke();raw=model.get_tensor(model.get_output_details()[0]['index']).ravel();assert raw.shape==(9,) and np.isfinite(raw).all();bank.execute('insert into outputs values(?,?)',(identity,raw.astype(np.float32).tobytes()));bank.commit();new+=1;return raw,identity
    for offset,i in enumerate(ids):
        r=rows[i]
        with Image.open(r['path']) as im:rgb=np.asarray(im.convert('RGB'))
        values=[fn(rgb,224,None) for fn in (canon.classifier_input,classifier_letterbox_input)];probabilities=[];keys=[]
        for value in values:
            raw,key=score(value);probabilities.append(posterior([raw],manifest));keys.append(key)
        probability=float(min(q[1] for q in probabilities));original,packet,retained=items[r['path']];now=recover_only(original,retained,packet,probability,data['manifest'])['app_replay']['kind']=='HEALTHY'
        if not r['y']:errors[r['source']]=errors.get(r['source'],0)+int(now)
        elif r['source']=='olid_tomato':healthy+=int(now)
        reports.append(dict(index=i,path=r['path'],source=r['source'],label=r['label'],posteriors=[q.tolist() for q in probabilities],routed_healthy=now,detected_rois=len(packet['leaves'])));input_shas.append(keys)
        if (offset+1)%8==0:print(json.dumps(dict(measured=offset+1,total=len(ids),new_encoder_inputs=new,exact_prediction_cache_hits=hits)),flush=True)
    result=dict(protocol_sha256=digest(out/'protocol.json'),healthy_validation_correct=healthy,new_false_healthy_by_source=errors,cheap_pass=healthy>=28 and sum(errors.values())==0,cases=reports,input_identities=input_shas,new_encoder_inputs=new,exact_prediction_cache_hits=hits,model_promoted=False,acceptance_data_used=False,ad_independent=False,olid_independent=False,elapsed_seconds=time.monotonic()-started);(out/'result.json').write_text(json.dumps(result,indent=2));bank.close();old.close();print(json.dumps({k:v for k,v in result.items() if k not in ('cases','input_identities')}),flush=True)
    if result['cheap_pass']:freeze(out/'candidate.json',dict(protocol_sha256=digest(out/'protocol.json'),full_qualification_required=True,model_promoted=False))

if __name__=='__main__':main()
