"""Screen the verified new development collection against cached old sources."""
from pathlib import Path
import argparse,collections,json,time
import cv2,numpy as np
from phase106_cross_source_overlap import Cache,digest,frozen,verify


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);out=p/'phase113_development_overlap';out.mkdir(exist_ok=True);started=time.monotonic()
    audit=p/'phase112_development_source_audit';summary=json.loads((audit/'result.json').read_text());assert summary['archive_verified'] and not summary['training_used'] and not summary['diagnostic_predictions_opened']
    manifest=audit/'manifest.json';rows=json.loads(manifest.read_text())
    previous=p/'phase109_tcp_cross_source';old=json.loads((previous/'result.json').read_text());assert old['protocol_sha256']==digest(previous/'protocol.json')
    gallery=json.loads((previous/'protocol.json').read_text())['gallery'];cache=Cache(p/'source_overlap_fingerprints.sqlite3')
    queries=[dict(row,id='dev107:'+row['member'],group=row['inferred_image_family']) for row in rows if not row['quarantined']]
    protocol=dict(source_sha256=digest(__file__),verifier_sha256=digest(Path(__file__).with_name('phase106_cross_source_overlap.py')),
                  gallery_protocol_sha256=digest(previous/'protocol.json'),query_manifest_sha256=digest(manifest),feature_version=cache.version,
                  source_role='development_only',diagnostic_predictions_opened=False,training_used=False,unique_target_correspondences=True,
                  retrieval_features=64,neighbors=8,maximum_candidates=20,minimum_votes=6,mirrored_queries=True)
    frozen(out/'protocol.json',protocol)
    cv2.setNumThreads(1);cv2.setRNGSeed(2113)
    cache.ensure(gallery,'old_gallery')
    variants=[dict(row,box=None,mirror=mirror) for row in queries for mirror in (False,True)];cache.ensure(variants,'new_development_fingerprints')
    counts=[min(64,len(cache.get(row)[1])) for row in gallery];total=sum(counts)
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    assert available>total*128*4*3+(1<<30),'Preserve existing useful caches'
    descriptors=np.empty((total,128),np.float32);owners=np.empty(total,np.int32);cursor=0
    for i,row in enumerate(gallery):
        count=counts[i];descriptors[cursor:cursor+count]=cache.get(row)[1][:count];owners[cursor:cursor+count]=i;cursor+=count
    index=cv2.flann_Index(descriptors,dict(algorithm=1,trees=4));progress_path=out/'progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else dict(completed=0,edges=[],protocol_sha256=digest(out/'protocol.json'))
    assert progress['protocol_sha256']==digest(out/'protocol.json')
    for i in range(progress['completed'],len(queries)):
        row=queries[i];features=[cache.get(dict(row,box=None,mirror=mirror)) for mirror in (False,True)];votes=collections.Counter()
        for feature in features:
            if not len(feature[1]):continue
            ids,distances=index.knnSearch(feature[1][:64],8,params=dict(checks=64))
            for indices,dist in zip(ids,distances):
                for owner in {int(owners[key]) for key,d in zip(indices,dist) if d<400**2}:votes[owner]+=1
        for j,count in votes.most_common(20):
            if count<6:continue
            found=None
            for mirror,feature in enumerate(features):
                found=verify(feature,cache.get(gallery[j]),2113+i*len(gallery)+j)
                if found is not None:break
            if found is not None:
                progress['edges'].append(dict(id=row['id'],group=row['group'],prior=gallery[j],mirrored=bool(mirror),**found));break
        progress['completed']=i+1
        if i%32==0 or i+1==len(queries):
            temporary=progress_path.with_suffix('.tmp');temporary.write_text(json.dumps(progress));temporary.replace(progress_path)
            print(json.dumps(dict(completed=i+1,total=len(queries),overlap_images=len(progress['edges']))),flush=True)
    bad_groups={row['group'] for row in progress['edges']};retained=[row for row in queries if row['group'] not in bad_groups]
    frozen(out/'retained_development_manifest.json',retained)
    result=dict(protocol_sha256=digest(out/'protocol.json'),gallery_identities=len(gallery),query_images=len(queries),overlap_images=len(progress['edges']),
                overlapping_inferred_families=len(bad_groups),retained_images=len(retained),retained_counts=dict(collections.Counter(r['label'] for r in retained)),
                edges=progress['edges'],source_role='development_only',acceptance_ready=False,independence_established=False,
                diagnostic_predictions_opened=False,training_used=False,new_encoder_inputs=0,elapsed_seconds=time.monotonic()-started,
                limitations='Approximate retrieval and inferred similarity families; no proof of specimen/capture independence or clinical labels. Whole collection must not be randomly split.')
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='edges'}),flush=True)


if __name__=='__main__':main()
