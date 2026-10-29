"""Exact-input, resumable spatial colour evidence; no new encoder computation."""
from pathlib import Path
import argparse,hashlib,json,sqlite3,time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from PIL import Image
from app_eval import canon,classifier_letterbox_input
from clinical_color_features import clinical_color_features
from structured_leaf_features import structured_leaf_features,DIMENSION
from opened_context_data import read
from permissive_encoder import digest
from population_siglip_tail import freeze

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True);p=Path(parser.parse_args().campaign);started=time.monotonic();base=Path('/dev/shm/agribot-phase223-structured');base.mkdir(exist_ok=True);out=p/'phase223_structured_leaf_cache'
    if not out.exists():out.symlink_to(base,target_is_directory=True)
    assert out.resolve()==base
    if (out/'complete.json').exists():
        c=read(out/'complete.json');assert c['source_sha256']==digest(__file__) and c['features_sha256']==digest(out/'features.npy');return
    native=p/'phase189_full_field_native_extension';np_=read(native/'protocol.json');old=p/'phase172_native_large_field_features';op=read(old/'protocol.json');color=p/'phase186_color_evidence_kernel/color_cache';cp=read(color/'protocol.json');cc=read(color/'complete.json')
    assert np_['links'][:len(op['links'])]==op['links'] and cc['protocol_sha256']==digest(color/'protocol.json') and cc['features_sha256']==digest(color/'features.npy') and cp['descriptor_sha256']==digest(Path(__file__).with_name('clinical_color_features.py'))
    protocol=dict(source_sha256=digest(__file__),descriptor_sha256=digest(Path(__file__).with_name('structured_leaf_features.py')),color_descriptor_sha256=cp['descriptor_sha256'],native_protocol_sha256=digest(native/'protocol.json'),old_color_complete_sha256=digest(color/'complete.json'),links=np_['links'],dimension=75+DIMENSION,spatial_dimension=DIMENSION,input='Canonical RGB518 exact native SHA; global75 reuse exact original224 parent SHA',clinical_segmentation_claim=False,new_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False,hypothesis='Relative colour around heuristic ridge-like structures and margin/interior may retain deficiency cues suppressed by pooled encoders. No labels are changed and no masks are clinical annotations.',stopping='Compute this single fixed descriptor, then test exactly two C1 kernels. No feature/threshold grid; unchanged recognition gates.')
    freeze(out/'protocol.json',protocol);old_values=np.load(color/'features.npy');global_cache={}
    for key,value in zip(cp['parent_input_sha256'],old_values):
        if key in global_cache:assert np.array_equal(value,global_cache[key])
        global_cache[key]=value
    metadata={r['input_sha256']:r for r in op['inputs']+np_['new_native_inputs']};db=sqlite3.connect(out/'cache.sqlite');db.execute('create table if not exists features (identity text primary key, spatial blob not null, global75 blob not null)');cv2.setNumThreads(1)
    @lru_cache(maxsize=8)
    def image(path,sha):
        assert digest(path)==sha
        with Image.open(path) as im:return np.asarray(im.convert('RGB'))
    def compute(link):
        row=metadata[link['input_sha256']];rgb=image(row['path'],row['file_sha256']);fn=canon.classifier_input if row['view']=='center' else classifier_letterbox_input;native_input=fn(rgb,518,row['box']);assert hashlib.sha256(np.ascontiguousarray(native_input).tobytes()).hexdigest()==link['input_sha256'];spatial=structured_leaf_features(np.clip(native_input[0]*canon.STD+canon.MEAN,0,1));parent=link['parent_224_input_sha256']
        if parent in global_cache:global75=global_cache[parent]
        else:
            small=fn(rgb,224,row['box']);assert hashlib.sha256(np.ascontiguousarray(small).tobytes()).hexdigest()==parent;global75=clinical_color_features(small[0]*canon.STD+canon.MEAN)
        return link['input_sha256'],spatial.astype('<f4').tobytes(),global75.astype('<f4').tobytes()
    links={r['input_sha256']:r for r in np_['links']};existing={r[0] for r in db.execute('select identity from features')};todo=[r for key,r in links.items() if key not in existing];print(json.dumps(dict(unique_inputs=len(links),spatial_cached=len(existing),spatial_new=len(todo),old_global_rows_reused=len(old_values))),flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n,record in enumerate(pool.map(compute,todo),1):
            db.execute('insert into features values (?,?,?)',record)
            if n%32==0:db.commit()
            if n%128==0:print(json.dumps(dict(completed=n,total=len(todo),elapsed=time.monotonic()-started)),flush=True)
    db.commit();cache={key:(np.frombuffer(spatial,dtype='<f4'),np.frombuffer(color_,dtype='<f4')) for key,spatial,color_ in db.execute('select * from features')};values=np.stack([np.concatenate([cache[r['input_sha256']][1],cache[r['input_sha256']][0]]) for r in np_['links']]);assert values.shape==(len(np_['links']),144) and np.isfinite(values).all();assert np.array_equal(values[:len(old_values),:75],old_values);np.save(out/'features.npy',values);db.close();image.cache_clear()
    complete=dict(source_sha256=digest(__file__),protocol_sha256=digest(out/'protocol.json'),features_sha256=digest(out/'features.npy'),shape=list(values.shape),new_encoder_inputs=0,new_spatial_inputs=len(todo),reused_spatial_inputs=len(existing),old_global_rows_reused=len(old_values),old_global_rows_bitwise_equal=True,invalid_foreground_rows=int((values[:,-3]==0).sum()),elapsed_seconds=time.monotonic()-started,acceptance_data_used=False);(out/'complete.json').write_text(json.dumps(complete,indent=2));print(json.dumps(complete),flush=True)

if __name__=='__main__':main()
