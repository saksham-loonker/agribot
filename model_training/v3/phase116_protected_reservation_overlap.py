"""Check proposed training photos against the still-unpredicted reservation."""
from pathlib import Path
import argparse,collections,json,time
import cv2,numpy as np
from phase106_cross_source_overlap import Cache,digest,frozen,verify


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);out=p/'phase116_protected_reservation_overlap';out.mkdir(exist_ok=True);started=time.monotonic()
    protected=p/'phase109_tcp_cross_source';status=json.loads((protected/'result.json').read_text())
    assert not status['training_used'] and not status['diagnostic_predictions_opened']
    path=protected/'provisional_after_geometry.json';gallery=json.loads(path.read_text())
    prepare=p/'phase114_new_supervision_features';chosen=json.loads((prepare/'protocol.json').read_text())['selected']
    available={row['id']:row for row in json.loads((p/'phase113_development_overlap/retained_development_manifest.json').read_text())}
    queries=[available[row['id']] for row in chosen]
    assert all(a['file_sha256']==b['sha256'] for a,b in zip(queries,chosen))
    cache=Cache(p/'source_overlap_fingerprints.sqlite3')
    frozen(out/'protocol.json',dict(source_sha256=digest(__file__),verifier_sha256=digest(Path(__file__).with_name('phase106_cross_source_overlap.py')),
                                   protected_manifest_sha256=digest(path),proposed_training_protocol_sha256=digest(prepare/'protocol.json'),
                                   feature_version=cache.version,protected_images=len(gallery),proposed_training_photos=len(queries),
                                   purpose='Geometry-only split-protection audit before fitting; no clinical predictions or labels from the protected reservation',clinical_predictions_opened=False,training_used=False))
    cv2.setNumThreads(1);cv2.setRNGSeed(2116);parts=[];counts=[]
    for row in gallery:
        feature=cache.get(dict(row,box=None,mirror=False));assert feature is not None,'Expected exact cached reservation fingerprint missing'
        part=feature[1][:64].copy();parts.append(part);counts.append(len(part))
    descriptors=np.concatenate(parts);owners=np.repeat(np.arange(len(gallery),dtype=np.int32),counts)
    index=cv2.flann_Index(descriptors,dict(algorithm=1,trees=4));edges=[]
    for i,row in enumerate(queries):
        features=[cache.get(dict(row,box=None,mirror=mirror)) for mirror in (False,True)]
        assert all(feature is not None for feature in features),'Expected exact cached new-source fingerprint missing'
        votes=collections.Counter()
        for feature in features:
            if not len(feature[1]):continue
            ids,dist=index.knnSearch(feature[1][:64],8,params=dict(checks=64))
            for indices,distances in zip(ids,dist):
                for owner in {int(owners[key]) for key,d in zip(indices,distances) if d<400**2}:votes[owner]+=1
        for j,count in votes.most_common(20):
            if count<6:continue
            found=None
            for mirror,feature in enumerate(features):
                found=verify(feature,cache.get(dict(gallery[j],box=None,mirror=False)),2116+i*len(gallery)+j)
                if found is not None:break
            if found is not None:edges.append(dict(proposed_training_id=row['id'],protected_id=gallery[j]['id'],protected_group=gallery[j]['group'],**found));break
    result=dict(protocol_sha256=digest(out/'protocol.json'),overlap_images=len(edges),edges=edges,proposed_training_photos=len(queries),protected_images=len(gallery),
                new_fingerprints=0,new_encoder_inputs=0,clinical_predictions_opened=False,protected_training_used=False,acceptance_ready=False,independence_established=False,
                limitations='Approximate geometry screen cannot establish capture independence. The protected reservation stays provisional and diagnostically sealed.',elapsed_seconds=time.monotonic()-started)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
