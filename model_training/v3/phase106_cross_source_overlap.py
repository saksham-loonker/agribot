"""Neural-free, resumable cross-source crop/augmentation overlap audit.

This screens image provenance, not plant condition. Passing approximate SIFT
retrieval never establishes capture independence or admits training data.
"""
from pathlib import Path
import argparse,ast,collections,concurrent.futures,functools,hashlib,inspect,io,json,os,sqlite3,time,zlib
import cv2
import numpy as np
from PIL import Image


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def frozen(path,value):
    text=json.dumps(value,sort_keys=True,indent=2)
    if path.exists():assert path.read_text()==text,('Frozen audit identity changed',str(path))
    else:path.write_text(text)


@functools.lru_cache(maxsize=4)
def gray(path,sha):
    value=Path(path).read_bytes()
    assert hashlib.sha256(value).hexdigest()==sha,('Source bytes changed',path)
    image=cv2.imdecode(np.frombuffer(value,np.uint8),cv2.IMREAD_GRAYSCALE|cv2.IMREAD_IGNORE_ORIENTATION)
    assert image is not None,('Image decode failed',path)
    return image


def extract(row):
    image=gray(row['path'],row['file_sha256'])
    box=row.get('box')
    if box:
        left,top,right,bottom=box;dx=(right-left)*.1;dy=(bottom-top)*.1
        image=image[max(0,int(top-dy)):min(len(image),int(np.ceil(bottom+dy))),max(0,int(left-dx)):min(image.shape[1],int(np.ceil(right+dx)))]
    assert image.size>0
    if row.get('mirror'):image=np.fliplr(image)
    factor=600/max(image.shape)
    image=cv2.resize(image,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA if factor<1 else cv2.INTER_LINEAR)
    points,descriptors=cv2.SIFT_create(nfeatures=512).detectAndCompute(image,None)
    if descriptors is None:return np.empty((0,2),np.float32),np.empty((0,128),np.float32),image.shape
    order=sorted(range(len(points)),key=lambda i:(-points[i].response,points[i].pt[0],points[i].pt[1],points[i].angle))[:512]
    return np.array([points[i].pt for i in order],np.float32),descriptors[order],image.shape


class Cache:
    def __init__(self,path):
        self.db=sqlite3.connect(path)
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA cache_size=-65536')
        self.db.execute('CREATE TABLE IF NOT EXISTS features(identity TEXT PRIMARY KEY,payload BLOB NOT NULL,payload_sha TEXT NOT NULL)')
        tree=[ast.dump(ast.parse(inspect.getsource(f)),include_attributes=False) for f in (gray.__wrapped__,extract)]
        self.version=hashlib.sha256(json.dumps([tree,cv2.__version__]).encode()).hexdigest()
    def key(self,row):
        return hashlib.sha256(json.dumps([self.version,row['file_sha256'],row.get('box'),row.get('mirror',False)],sort_keys=True).encode()).hexdigest()
    def get(self,row):
        key=self.key(row);value=self.db.execute('SELECT payload,payload_sha FROM features WHERE identity=?',(key,)).fetchone()
        if value is None:return None
        assert hashlib.sha256(value[0]).hexdigest()==value[1],'Fingerprint cache corrupted'
        with np.load(io.BytesIO(zlib.decompress(value[0])),allow_pickle=False) as data:
            return data['points'],data['descriptors'],tuple(data['shape'])
    def put(self,row,value):
        points,descriptors,shape=value;stream=io.BytesIO()
        np.savez(stream,points=points,descriptors=descriptors,shape=shape)
        blob=zlib.compress(stream.getvalue(),1)
        self.db.execute('INSERT OR REPLACE INTO features VALUES(?,?,?)',(self.key(row),blob,hashlib.sha256(blob).hexdigest()))
    def ensure(self,rows,label):
        with concurrent.futures.ThreadPoolExecutor(6) as pool:
            for start in range(0,len(rows),64):
                part=rows[start:start+64];missing=[row for row in part if self.get(row) is None]
                for row,value in zip(missing,pool.map(extract,missing)):self.put(row,value)
                self.db.commit()
                if start%512==0 or start+64>=len(rows):print(json.dumps(dict(stage=label,completed=min(start+64,len(rows)),total=len(rows),new_fingerprints=len(missing))),flush=True)


def verify(first,second,seed):
    pa,da,sa=first;pb,db,sb=second
    if len(da)<16 or len(db)<16:return None
    good=[m for m,n in cv2.BFMatcher().knnMatch(da,db,k=2) if m.distance<.75*n.distance]
    # Repeated texture can map many query descriptors to one target corner.
    # Those are not independent correspondences and can fit a degenerate H.
    unique={}
    for match in sorted(good,key=lambda m:m.distance):unique.setdefault(match.trainIdx,match)
    good=list(unique.values())
    if len(good)<16:return None
    a=np.array([pa[m.queryIdx] for m in good]);b=np.array([pb[m.trainIdx] for m in good])
    cv2.setRNGSeed(seed)
    transform,mask=cv2.findHomography(a,b,cv2.RANSAC,3.)
    if transform is None or mask is None:return None
    selected=mask.ravel().astype(bool);count=int(selected.sum());fraction=float(selected.mean())
    if count<16 or fraction<.7:return None
    error=float(np.median(np.linalg.norm(cv2.perspectiveTransform(a[:,None,:],transform).reshape(-1,2)[selected]-b[selected],axis=1)))
    coverage=[float(cv2.contourArea(cv2.convexHull(points[selected]))/(shape[0]*shape[1])) for points,shape in [(a,sa),(b,sb)]]
    if error>1.5 or min(coverage)<.04:return None
    return dict(inliers=count,inlier_fraction=fraction,median_error=error,coverage=coverage)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    parser.add_argument('--output-name',default='phase106_tcp_cross_source')
    args=parser.parse_args();assert Path(args.output_name).name==args.output_name and args.output_name not in ('.','..')
    p=Path(args.campaign);out=p/args.output_name;out.mkdir(exist_ok=True)
    started=time.monotonic();audit=json.loads((p/'phase86_tcp_source/membership_audit.json').read_text())
    assert not audit['diagnostic_predictions_opened'] and not audit['training_used']
    sources=json.loads((p/'phase29_compact/source_files.json').read_text())
    fallback={sha:path for path,sha in sources.items() if Path(path).is_file()}
    gallery={};missing=set()
    def add(path,sha,box=None):
        if not path or not sha:return
        if not Path(path).is_file():path=fallback.get(sha)
        if not path:missing.add(sha);return
        for region in ([None,box] if isinstance(box,list) and len(box)==4 else [None]):
            row=dict(path=path,file_sha256=sha,box=region)
            key=json.dumps([sha,region],sort_keys=True)
            gallery.setdefault(key,row)
    for path,sha in sources.items():add(path,sha)
    prior={}
    for relative,expected in audit['prior_manifests'].items():
        file=p/relative;assert digest(file)==expected;prior[relative]=expected
        if relative.endswith('source_files.json'):continue
        values=json.loads(file.read_text())
        if isinstance(values,list):
            for row in values:add(row.get('path'),row.get('file_sha256'),row.get('box'))
    gallery=[gallery[key] for key in sorted(gallery)]
    query_path=p/'phase86_tcp_source/provisional_sealed_manifest.json'
    queries=json.loads(query_path.read_text())
    fingerprint=p/'source_overlap_fingerprints.sqlite3';cache=Cache(fingerprint)
    protocol=dict(source_sha256=digest(__file__),feature_version=cache.version,prior_manifests=prior,
                  gallery=gallery,query_manifest_sha256=digest(query_path),queries=len(queries),
                  retrieval_features_per_image=64,neighbors=8,minimum_votes=6,maximum_candidates=20,
                  verification_features=512,minimum_inliers=16,minimum_inlier_fraction=.7,maximum_median_error=1.5,minimum_coverage=.04,
                  unique_target_correspondences=True,
                  mirrored_queries=True,missing_prior_identities=sorted(missing),
                  role='Geometry-only overlap audit; no diagnostic model, labels used only to summarize final retained counts',
                  training_used=False,diagnostic_predictions_opened=False)
    frozen(out/'protocol.json',protocol)
    assert __import__('shutil').disk_usage(p).free>len(gallery)*110000+(2<<30),'Preserve existing caches; insufficient bounded fingerprint storage'
    cv2.setNumThreads(1);cv2.setRNGSeed(2106)
    cache.ensure(gallery,'gallery_fingerprints')
    query_rows=[dict(row,box=None,mirror=mirror) for row in queries for mirror in (False,True)]
    cache.ensure(query_rows,'reserved_source_fingerprints')
    counts=[min(64,len(cache.get(row)[1])) for row in gallery];total=sum(counts)
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    assert available>total*128*4*3+(1<<30),'Preserve useful caches; insufficient memory for bounded retrieval'
    descriptors=np.empty((total,128),np.float32);owners=np.empty(total,np.int32);cursor=0
    for i,row in enumerate(gallery):
        n=counts[i];descriptors[cursor:cursor+n]=cache.get(row)[1][:n];owners[cursor:cursor+n]=i;cursor+=n
    index=cv2.flann_Index(descriptors,dict(algorithm=1,trees=4))
    progress_path=out/'progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else dict(completed=0,edges=[])
    if progress_path.exists():assert progress['protocol_sha256']==digest(out/'protocol.json')
    for i in range(progress['completed'],len(queries)):
        row=queries[i];variants=[dict(row,box=None,mirror=mirror) for mirror in (False,True)]
        votes=collections.Counter();features=[cache.get(v) for v in variants]
        for feature in features:
            part=feature[1][:64]
            if not len(part):continue
            ids,distances=index.knnSearch(part,8,params=dict(checks=64))
            for ids0,dist0 in zip(ids,distances):
                for owner in {int(owners[k]) for k,d in zip(ids0,dist0) if d<400**2}:votes[owner]+=1
        candidates=[j for j,count in votes.most_common(20) if count>=6]
        for j in candidates:
            old=cache.get(gallery[j]);found=None
            for variant,feature in enumerate(features):
                found=verify(feature,old,2106+i*len(gallery)+j)
                if found is not None:break
            if found is not None:
                progress['edges'].append(dict(query_id=row['id'],query_group=row['group'],prior=gallery[j],mirrored=bool(variant),votes=votes[j],**found));break
        progress['completed']=i+1
        if i%32==0 or i+1==len(queries):
            progress['protocol_sha256']=digest(out/'protocol.json')
            temporary=progress_path.with_suffix('.tmp');temporary.write_text(json.dumps(progress));temporary.replace(progress_path)
            print(json.dumps(dict(stage='cross_source_overlap',completed=i+1,total=len(queries),overlap_images=len(progress['edges']))),flush=True)
    overlapping={r['query_group'] for r in progress['edges']}
    retained=[row for row in queries if row['group'] not in overlapping]
    frozen(out/'provisional_after_geometry.json',retained)
    result=dict(protocol_sha256=digest(out/'protocol.json'),gallery_identities=len(gallery),missing_prior_identities=len(missing),
                overlap_images=len(progress['edges']),overlapping_families=len(overlapping),retained_images=len(retained),
                retained_counts=dict(collections.Counter(row['label'] for row in retained)),
                edges=progress['edges'],training_used=False,diagnostic_predictions_opened=False,
                acceptance_ready=False,independence_established=False,elapsed_seconds=time.monotonic()-started,
                limitations='Approximate 64-feature retrieval and 20-candidate cap; full 512-feature homography verification. Source mixes internet/prior datasets, and successful overlap screening does not establish independent capture or clinical labels.')
    (out/'result.json').write_text(json.dumps(result,indent=2));cache.db.execute('PRAGMA wal_checkpoint(TRUNCATE)');cache.db.close()
    print(json.dumps({key:value for key,value in result.items() if key!='edges'}),flush=True)


if __name__=='__main__':main()
