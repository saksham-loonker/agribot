"""Consumed-source cross-label geometry audit; never relabel or change gates."""
from pathlib import Path
import argparse,json,time
import cv2
from phase106_cross_source_overlap import Cache,extract,verify,digest,frozen
from phase114_new_supervision_features import original_features
from opened_context_data import read


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True);p=Path(parser.parse_args().campaign);out=p/'phase184_nutrient_label_geometry_audit';out.mkdir(exist_ok=True);started=time.monotonic()
    rows=read(p/'phase24_olid_source/sealed_manifest.json');_,bags,descriptors,_=original_features(p)
    queries=[];gallery=[];missing=[]
    for row in rows:
        if row['label'] not in ('Healthy','Nitrogen Deficiency'):continue
        bag=bags.get(row['path'])
        if not bag or not bag['full']:missing.append(row['id']);continue
        for leaf,key in enumerate(bag['full']):
            d=descriptors[key];region=dict(id=row['id']+':leaf'+str(leaf),path=row['path'],file_sha256=row['file_sha256'],box=d['box'],label=row['label'])
            (gallery if row['label']=='Healthy' else queries).append(region)
    cache=Cache(p/'source_overlap_fingerprints.sqlite3');memo={};counts=dict(reused=0,new=0)
    protocol=dict(source_sha256=digest(__file__),verifier_sha256=digest(Path(__file__).with_name('phase106_cross_source_overlap.py')),manifest_sha256=digest(p/'phase24_olid_source/sealed_manifest.json'),feature_version=cache.version,
        query_regions=queries,gallery_regions=gallery,missing_detector_regions=missing,mirrored_queries=True,exhaustive_region_pairs=True,labels_consumed_development=True,diagnostic_only=True,automatic_relabeling=False,automatic_exclusion=False,new_encoder_inputs=0,acceptance_data_used=False)
    frozen(out/'protocol.json',protocol)
    if (out/'result.json').exists():assert read(out/'result.json')['protocol_sha256']==digest(out/'protocol.json');return
    def get(row):
        key=cache.key(row)
        if key not in memo:
            value=cache.get(row)
            if value is None:value=extract(row);counts['new']+=1
            else:counts['reused']+=1
            memo[key]=value
        return memo[key]
    cv2.setNumThreads(1);edges=[];pairs=0
    for i,query in enumerate(queries):
        variants=[get(dict(query,mirror=mirror)) for mirror in (False,True)]
        for j,healthy in enumerate(gallery):
            reference=get(healthy);pairs+=1
            for mirror,value in enumerate(variants):
                proof=verify(value,reference,2184+i*len(gallery)+j)
                if proof is not None:edges.append(dict(nitrogen=query,healthy=healthy,mirrored=bool(mirror),**proof));break
        print(json.dumps(dict(completed_query_regions=i+1,total_query_regions=len(queries),cross_label_matches=len(edges))),flush=True)
    result=dict(protocol_sha256=digest(out/'protocol.json'),query_regions=len(queries),gallery_regions=len(gallery),region_pairs=pairs,geometric_cross_label_matches=len(edges),edges=edges,fingerprints=counts,diagnostic_only=True,
        automatic_relabeling=False,model_promoted=False,new_encoder_inputs=0,acceptance_data_used=False,ad_independent=False,olid_independent=False,elapsed_seconds=time.monotonic()-started,
        limitations='A geometry match suggests shared visual provenance, not confirmed erroneous clinical labels or plant identity. A nonmatch does not establish label validity, specimen independence or separability. No cases are dropped from gates.')
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='edges'}),flush=True)


if __name__=='__main__':main()
