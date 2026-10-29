"""New field supervision versus old photos and protected geometry, no scoring."""
from pathlib import Path
import argparse,collections,hashlib,json,shutil,time
import cv2,numpy as np
from phase106_cross_source_overlap import Cache,digest,frozen,verify


def read(path):return json.loads(Path(path).read_text())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);out=p/'phase163_field_source_overlap';out.mkdir(exist_ok=True);started=time.monotonic()
    source=p/'phase162_field_source_profile';status=read(source/'result.json');assert status['manifest_sha256']==digest(source/'manifest.json') and not status['model_predictions_opened'] and not status['training_used']
    old=p/'phase109_tcp_cross_source';protected_status=read(old/'result.json');assert not protected_status['diagnostic_predictions_opened'] and not protected_status['training_used']
    gallery=[];lookup={}
    prior_sources=[('prior',read(old/'protocol.json')['gallery']),('new107',read(p/'phase112_development_source_audit/manifest.json')),('protected',read(old/'provisional_after_geometry.json'))]
    # Only geometric identity fields from protected records enter this audit.
    # Protected clinical labels are neither copied to fitting rows nor scored.
    for origin,rows in prior_sources:
        for row in rows:
            sha=row['file_sha256'];box=row.get('box') if origin=='prior' else None;key=(sha,None if box is None else tuple(box))
            if key in lookup:gallery[lookup[key]]['origins'].append(origin);continue
            identity=row.get('id',origin+':'+hashlib.sha256(json.dumps([sha,box],sort_keys=True).encode()).hexdigest())
            lookup[key]=len(gallery);gallery.append(dict(id=identity,path=row['path'],file_sha256=sha,box=box,group=row.get('group',sha),origins=[origin]))
    manifest=read(source/'manifest.json');queries=[row for row in manifest if not row['quarantined']];cache=Cache(p/'source_overlap_fingerprints.sqlite3')
    protocol=dict(source_sha256=digest(__file__),verifier_sha256=digest(Path(__file__).with_name('phase106_cross_source_overlap.py')),source_profile_sha256=digest(source/'result.json'),
        old_gallery_protocol_sha256=digest(old/'protocol.json'),new107_manifest_sha256=digest(p/'phase112_development_source_audit/manifest.json'),protected_manifest_sha256=digest(old/'provisional_after_geometry.json'),
        gallery_identity_sha256=hashlib.sha256(json.dumps(gallery,sort_keys=True,separators=(',',':')).encode()).hexdigest(),gallery_entries=len(gallery),query_images=len(queries),feature_version=cache.version,
        retrieval_features=64,neighbors=8,maximum_candidates=20,minimum_votes=6,unique_target_correspondences=True,mirrored_queries=True,
        quarantine='Exact prior encoded-file identity or geometrically verified crop/augmentation match removes the complete inferred new-photo family; all old fitting/validation/consumed and provisional protected photos are included',
        source_role='development_only',acceptance_data_used=False,protected_clinical_predictions_opened=False,new_encoder_inputs=0)
    frozen(out/'protocol.json',protocol)
    if (out/'result.json').exists():assert read(out/'result.json')['protocol_sha256']==digest(out/'protocol.json');return
    by_sha=collections.defaultdict(list)
    for row in gallery:by_sha[row['file_sha256']].append(row)
    path=out/'progress.json';progress=read(path) if path.exists() else dict(protocol_sha256=digest(out/'protocol.json'),completed=0,edges=[])
    assert progress['protocol_sha256']==digest(out/'protocol.json')
    for i in range(0,len(queries),32):
        variants=[dict(row,box=None,mirror=mirror) for row in queries[i:i+32] if row['file_sha256'] not in by_sha for mirror in (False,True)]
        missing=sum(cache.get(row) is None for row in variants)
        # Bound each incremental cache write; never delete old useful caches.
        assert shutil.disk_usage(out).free>missing*512*136*4+(1<<30),'Preserve one GiB disk reserve while extending fingerprints'
        cache.ensure(variants,'field93_fingerprints')
    cv2.setNumThreads(1);cv2.setRNGSeed(2163);counts=[]
    for row in gallery:
        value=cache.get(row);assert value is not None,('Required exact-identity old geometry cache missing',row['id']);counts.append(min(64,len(value[1])))
    total=sum(counts);available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    assert available>total*128*4*3+(1<<30),'Preserve RAM for useful existing jobs/caches'
    descriptors=np.empty((total,128),np.float32);owners=np.empty(total,np.int32);cursor=0
    for i,row in enumerate(gallery):
        n=counts[i];descriptors[cursor:cursor+n]=cache.get(row)[1][:n];owners[cursor:cursor+n]=i;cursor+=n
    index=cv2.flann_Index(descriptors,dict(algorithm=1,trees=4))
    for i in range(progress['completed'],len(queries)):
        row=queries[i];found=None
        if row['file_sha256'] in by_sha:found=dict(method='exact_encoded_file',prior=by_sha[row['file_sha256']][0])
        else:
            features=[cache.get(dict(row,box=None,mirror=mirror)) for mirror in (False,True)];votes=collections.Counter()
            for feature in features:
                if not len(feature[1]):continue
                ids,distances=index.knnSearch(feature[1][:64],8,params=dict(checks=64))
                for indices,dist in zip(ids,distances):
                    for owner in {int(owners[key]) for key,d in zip(indices,dist) if d<400**2}:votes[owner]+=1
            for j,count in votes.most_common(20):
                if count<6:continue
                for mirror,feature in enumerate(features):
                    proof=verify(feature,cache.get(gallery[j]),2163+i*len(gallery)+j)
                    if proof is not None:found=dict(method='verified_geometry',prior=gallery[j],mirrored=bool(mirror),**proof);break
                if found is not None:break
        if found is not None:progress['edges'].append(dict(id=row['id'],group=row['group'],**found))
        progress['completed']=i+1
        if i%32==0 or i+1==len(queries):
            temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(progress));temporary.replace(path)
            print(json.dumps(dict(stage='field93_overlap',completed=i+1,total=len(queries),overlap_images=len(progress['edges']))),flush=True)
    bad_groups={row['group'] for row in progress['edges']};retained=[row for row in queries if row['group'] not in bad_groups]
    frozen(out/'retained_development_manifest.json',retained)
    result=dict(protocol_sha256=digest(out/'protocol.json'),retained_manifest_sha256=digest(out/'retained_development_manifest.json'),gallery_entries=len(gallery),query_images=len(queries),
        overlap_images=len(progress['edges']),overlapping_families=len(bad_groups),retained_images=len(retained),retained_label_counts=dict(collections.Counter(row['label'] for row in retained)),
        retained_publisher_class_counts=dict(collections.Counter(row['publisher_class'] for row in retained)),edges=progress['edges'],source_role='development_only',training_used=False,model_predictions_opened=False,
        new_encoder_inputs=0,protected_training_used=False,protected_clinical_predictions_opened=False,independence_established=False,acceptance_ready=False,
        limitations='Approximate geometry and inferred families do not prove specimen/capture independence. The entire publisher collection is development-only; protected data stays diagnostically sealed.',elapsed_seconds=time.monotonic()-started)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='edges'}),flush=True)


if __name__=='__main__':main()
