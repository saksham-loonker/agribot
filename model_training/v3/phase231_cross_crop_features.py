"""Stream auxiliary canonical inputs; screen provenance before any training."""
from pathlib import Path
import argparse,collections,hashlib,io,json,sqlite3,sys,tarfile,time
import cv2,numpy as np,torch,timm
from compact_healthy_model import primary
from partial_dino_cache import CanonicalInputs
from phase114_new_supervision_features import original_features
from phase106_cross_source_overlap import Cache,verify
from opened_context_data import read
from permissive_encoder import digest
from population_siglip_tail import freeze
from app_eval import canon
from prepared_overlap import prepared_features

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',required=True);ap.add_argument('--manifest',required=True);args=ap.parse_args();p=Path(args.campaign);out=p/'phase231_cross_crop_features';out.mkdir(exist_ok=True);started=time.monotonic();manifest_path=Path(args.manifest);manifest=read(manifest_path);assert manifest['crop']=='Cucumber' and manifest['role']=='auxiliary_training_only' and not manifest['acceptance_data_used'];lookup={r['packet']:r for r in manifest['records']};assert len(lookup)==len(manifest['records'])
    protected=p/'phase109_tcp_cross_source';status=read(protected/'result.json');assert not status['training_used'] and not status['diagnostic_predictions_opened'];gallery=[];known=set()
    for origin,records in [('prior',read(protected/'protocol.json')['gallery']),('protected',read(protected/'provisional_after_geometry.json'))]:
        for r in records:
            row=dict(id=r.get('id',r['file_sha256']),path=r['path'],file_sha256=r['file_sha256'],box=r.get('box') if origin=='prior' else None,mirror=False,origin=origin);key=(row['file_sha256'],None if row['box'] is None else tuple(row['box']))
            if key not in known:gallery.append(row);known.add(key)
    lf_sha=lambda path:hashlib.sha256(Path(path).read_text(encoding='utf-8').encode()).hexdigest()
    cache=Cache(p/'source_overlap_fingerprints.sqlite3');assert manifest['opencv']==cv2.__version__;protocol=dict(source_sha256=digest(__file__),manifest_sha256=digest(manifest_path),source_archive_sha256=manifest['archive_sha256'],source_role=manifest['role'],crop='Cucumber',whole_collection_single_training_fold=True,encoder_bundle_sha256=digest(p/'export_cnn_safe/pytorch_bundle.pt'),app_eval_sha256=lf_sha(Path(__file__).with_name('app_eval.py')),canonical_sha256=lf_sha(canon.__file__),prepared_overlap_sha256=lf_sha(Path(__file__).with_name('prepared_overlap.py')),fingerprint_version=cache.version,verifier_sha256=digest(Path(__file__).with_name('phase106_cross_source_overlap.py')),gallery_identity_sha256=hashlib.sha256(json.dumps(gallery,sort_keys=True).encode()).hexdigest(),protected_manifest_sha256=digest(protected/'provisional_after_geometry.json'),gallery_entries=len(gallery),input='Publisher whole detached-leaf auxiliary photo, canonical center and letterbox224; no tomato detector or invented per-ROI labels',role='Auxiliary Healthy/stress representation only. No foreign crop becomes an eligible tomato app input; all original genus/evidence/recognition checks remain required.',overlap='Exact prior/protected file SHA or verified SIFT crop/augmentation overlap quarantines entire auxiliary collection. Approximate geometry never proves capture independence.',encoder_changed=False,new_source_training_used=False,acceptance_data_used=False,protected_clinical_predictions_opened=False,ad_independent=False,olid_independent=False)
    runtime_binding=read(manifest_path.with_suffix('.runtime.json'));protocol['canonical_runtime_sha256']=lf_sha(Path(__file__).resolve().parents[2]/'tests/model_evaluation/canonical_runtime.py');assert runtime_binding['manifest_sha256']==digest(manifest_path) and runtime_binding['canonical_runtime_sha256']==protocol['canonical_runtime_sha256']
    assert all(manifest[key]==protocol[key] for key in ('app_eval_sha256','canonical_sha256','prepared_overlap_sha256'));freeze(out/'protocol.json',protocol);cv2.setNumThreads(1);cv2.setRNGSeed(2310);parts=[];counts=[]
    for row in gallery:
        feature=cache.get(row);assert feature is not None,'Missing existing gallery identity; do not silently weaken overlap screen';part=feature[1][:64];parts.append(part);counts.append(len(part))
    descriptors=np.concatenate(parts);owners=np.repeat(np.arange(len(gallery),dtype=np.int32),counts);index=cv2.flann_Index(descriptors,dict(algorithm=1,trees=4));old_values,_,old_descriptors,parents=original_features(p);old_lookup={r['input_sha256']:i for i,r in enumerate(old_descriptors)};bank=sqlite3.connect(out/'features.sqlite');bank.execute('create table if not exists features(identity text primary key,payload blob not null,sha256 text not null)');model=None;pending={};new=0;reused=0;reports={};bags={};parity=None
    def saved(key):
        r=bank.execute('select payload,sha256 from features where identity=?',(key,)).fetchone()
        if r is None:return None
        assert hashlib.sha256(r[0]).hexdigest()==r[1];return np.frombuffer(r[0],dtype='<f4').copy()
    def put(key,value):
        value=np.asarray(value,dtype='<f4').tobytes();bank.execute('insert or replace into features values(?,?,?)',(key,value,hashlib.sha256(value).hexdigest()))
    def flush():
        nonlocal model,new,parity
        if not pending:return
        if model is None:
            torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;timm.layers.set_fused_attn(False);_,model=primary(p);model=model.cuda().eval();pixels=CanonicalInputs(p)
            try:
                records=old_descriptors[:4];tensors=[pixels(r if 'kind' in r else dict(r,kind='new',policy=r['view'])) for r in records]
                with torch.inference_mode():actual,_=model.components(torch.from_numpy(np.concatenate(tensors)).cuda())
                parity=float(abs(actual.cpu().numpy()-old_values[:4]).max());assert parity<2e-4,('Feature extractor/cache parity failed',parity)
            finally:pixels.close()
        keys=list(pending)
        with torch.inference_mode():values,_=model.components(torch.from_numpy(np.concatenate([pending[k] for k in keys])).cuda())
        for key,value in zip(keys,values.cpu().numpy()):put(key,value)
        bank.commit();new+=len(keys);pending.clear()
    archive=tarfile.open(fileobj=sys.stdin.buffer,mode='r|');file_shas={r['file_sha256'] for r in gallery}
    for member in archive:
        assert member.isfile() and member.name in lookup and member.size<8<<20 and member.name not in reports;record=lookup[member.name];payload=archive.extractfile(member).read();assert hashlib.sha256(payload).hexdigest()==record['packet_sha256']
        with np.load(io.BytesIO(payload),allow_pickle=False) as packet:
            views=packet['views'];assert views.shape==(2,224,224,3) and views.dtype==np.float32 and np.isfinite(views).all();assert [hashlib.sha256(np.ascontiguousarray(view[None]).tobytes()).hexdigest() for view in views]==record['input_sha256'];features=[prepared_features(packet[name]) for name in ('gray0','gray1')]
        votes=collections.Counter();match=None
        if record['file_sha256'] in file_shas:match=dict(kind='exact_file_sha256')
        else:
            for f in features:
                if not len(f[1]):continue
                neighbors,distances=index.knnSearch(f[1][:64],8,params=dict(checks=64))
                for ids,ds in zip(neighbors,distances):
                    for owner in {int(owners[k]) for k,d in zip(ids,ds) if d<400**2}:votes[owner]+=1
            for j,count in votes.most_common(20):
                if count<6:continue
                old=cache.get(gallery[j]);found=None
                for f in features:
                    found=verify(f,old,2310+len(reports)*len(gallery)+j)
                    if found is not None:break
                if found is not None:match=dict(kind='verified_geometry',gallery_id=gallery[j]['id'],origin=gallery[j]['origin'],**found);break
        reports[member.name]=dict(file_sha256=record['file_sha256'],packet_sha256=record['packet_sha256'],overlap=match);bags[record['id']]=dict(record=record,views=record['input_sha256'])
        if match is None:
            for key,view in zip(record['input_sha256'],views):
                if key in old_lookup:
                    if saved(key) is None:put(key,old_values[old_lookup[key]])
                    reused+=1
                elif saved(key) is None and key not in pending:pending[key]=view[None]
            if len(pending)>=32:flush()
        bank.commit()
        if len(reports)%16==0:print(json.dumps(dict(processed=len(reports),total=len(lookup),new_encoder_inputs=new,overlaps=sum(r['overlap'] is not None for r in reports.values()))),flush=True)
    assert set(reports)==set(lookup);flush();overlap=sum(r['overlap'] is not None for r in reports.values());(out/'overlap_audit.json').write_text(json.dumps(dict(protocol_sha256=digest(out/'protocol.json'),records=reports,overlap_images=overlap,whole_collection_quarantined=overlap>0,protected_clinical_predictions_opened=False,training_used=False,independence_established=False),indent=2))
    if not overlap:
        identities=sorted({k for b in bags.values() for k in b['views']});values=np.stack([saved(k) for k in identities]);assert values.shape==(len(identities),1408);np.save(out/'features.npy',values);(out/'input_sha256.json').write_text(json.dumps(identities));(out/'bags.json').write_text(json.dumps(bags));result=dict(protocol_sha256=digest(out/'protocol.json'),features_sha256=digest(out/'features.npy'),identities_sha256=digest(out/'input_sha256.json'),bags_sha256=digest(out/'bags.json'),source_images=len(bags),new_encoder_inputs=new,existing_encoder_inputs_reused=reused,original_feature_parity_max_error=parity,overlap_images=0,training_used=False,acceptance_data_used=False,ad_independent=False,olid_independent=False,elapsed_seconds=time.monotonic()-started);(out/'complete.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    else:print(json.dumps(dict(blocked=True,overlap_images=overlap,training_used=False)),flush=True)
    bank.close();cache.db.close()

if __name__=='__main__':main()
