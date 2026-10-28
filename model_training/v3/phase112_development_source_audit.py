"""Profile a publisher-verified development source without model predictions.

Photo similarity components are inferred image families, never plant IDs.
Only leaf folders are extracted. Publisher Infected means broad disease,
not one of the application's named diseases and not unsupported Other.
"""
from pathlib import Path
import argparse,collections,concurrent.futures,hashlib,io,json,shutil,zipfile,zlib
import cv2,numpy as np
from PIL import Image


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def profile(row):
    path=Path(row['path'])
    encoded=path.read_bytes();assert zlib.crc32(encoded)==row['crc32'],('Publisher member CRC mismatch',str(path))
    with Image.open(io.BytesIO(encoded)) as image:
        exif=image.getexif();rgb=np.asarray(image.convert('RGB'))
        h,w=rgb.shape[:2]
        small=cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA)
        gray=cv2.cvtColor(small,cv2.COLOR_RGB2GRAY)
        hashes=[]
        for mirror in (False,True):
            value=np.fliplr(gray) if mirror else gray
            for rotation in range(4):
                tiny=cv2.resize(np.rot90(value,rotation),(32,32),interpolation=cv2.INTER_AREA).astype(np.float32)
                low=cv2.dct(tiny)[:8,:8];bits=low>np.median(low.ravel()[1:]);bits[0,0]=False
                hashes.append(int.from_bytes(np.packbits(bits.ravel()).tobytes(),'big'))
        return dict(row,file_sha256=hashlib.sha256(encoded).hexdigest(),decoded_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),width=w,height=h,
                    orientation='portrait' if h>w else 'landscape' if w>h else 'square',exif_orientation=int(exif.get(274,1)),
                    camera=str(exif.get(272,'')),capture_time=str(exif.get(306,'')),rotation_mirror_phash=f'{min(hashes):016x}')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);source=p/'phase107_new_leaf_development';out=p/'phase112_development_source_audit';out.mkdir(exist_ok=True)
    archive=source/'Tomato.zip';verified=json.loads((source/'archive_verified.json').read_text());inventory=json.loads((source/'inventory.json').read_text())['files'][0]
    assert verified['verified'] and verified['sha256']==inventory['content_details']['sha256_hash']==digest(archive)
    protocol=dict(source_sha256=digest(__file__),archive_sha256=verified['sha256'],source_doi='10.17632/xc94xbg239.1',
                  source_role='development_only',labels={'Healthy_Leaf':'Healthy','Infected_Leaf':'Diseased_unspecified'},
                  photo_is_unit=True,leaf_clinical_labels_inferred=False,training_used=False,diagnostic_predictions_opened=False,
                  family_method='Connected components of canonical rotation/mirror pHash Hamming <=6; approximate image similarity only, never specimen/capture independence',
                  split_policy='Treat the entire source as one collection; no random train/validation split of these photographs')
    contract=out/'protocol.json';text=json.dumps(protocol,sort_keys=True,indent=2)
    if contract.exists():assert contract.read_text()==text
    else:contract.write_text(text)
    records=[]
    with zipfile.ZipFile(archive) as z:
        selected=[m for m in z.infolist() if m.filename.split('/')[0] in protocol['labels'] and not m.is_dir()]
        assert len(selected)==2063
        assert shutil.disk_usage(out).free>sum(m.file_size for m in selected)+(2<<30),'Preserve caches and headroom'
        for member in selected:
            parts=Path(member.filename).parts;assert len(parts)==2 and parts[1] not in ('.','..')
            assert member.file_size<20<<20
            path=out/'images'/parts[0]/parts[1];path.parent.mkdir(exist_ok=True,parents=True)
            if not path.exists():
                value=z.read(member);temporary=path.with_suffix('.tmp');temporary.write_bytes(value);temporary.replace(path)
            assert path.stat().st_size==member.file_size
            records.append(dict(path=str(path),member=member.filename,label=protocol['labels'][parts[0]],crc32=member.CRC))
    cv2.setNumThreads(1)
    with concurrent.futures.ThreadPoolExecutor(4) as pool:rows=list(pool.map(profile,records))
    labels=collections.defaultdict(set)
    for row in rows:labels[row['decoded_rgb_sha256']].add(row['label'])
    contradictory={sha for sha,value in labels.items() if len(value)>1}
    parents=list(range(len(rows)))
    def root(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    codes=[int(row['rotation_mirror_phash'],16) for row in rows];edges=0
    for i in range(len(rows)):
        for j in range(i):
            if (codes[i]^codes[j]).bit_count()<=6:parents[root(i)]=root(j);edges+=1
    groups=collections.defaultdict(list)
    for i,row in enumerate(rows):groups[root(i)].append(row)
    for family in groups.values():
        key=hashlib.sha256('\n'.join(sorted({row['file_sha256'] for row in family})).encode()).hexdigest()[:20]
        for row in family:row['inferred_image_family']='new107:'+key;row['collection_group']='new107:entire_publisher_collection';row['quarantined']=row['decoded_rgb_sha256'] in contradictory
    summary=dict(protocol_sha256=digest(contract),source_sha256=digest(__file__),archive_verified=True,images=len(rows),
                 label_counts=dict(collections.Counter(row['label'] for row in rows)),file_unique=len({row['file_sha256'] for row in rows}),decoded_unique=len(labels),
                 exact_cross_label_pixel_identities=len(contradictory),inferred_image_families=len(groups),similarity_edges=edges,
                 largest_inferred_family=max(map(len,groups.values())),
                 by_label={label:dict(images=len(part),orientation=dict(collections.Counter(r['orientation'] for r in part)),cameras=dict(collections.Counter(r['camera'] for r in part)),
                                     missing_capture_time=sum(not r['capture_time'] for r in part),inferred_families=len({r['inferred_image_family'] for r in part}))
                           for label in protocol['labels'].values() for part in [[r for r in rows if r['label']==label]]},
                 training_used=False,diagnostic_predictions_opened=False,independence_established=False,acceptance_ready=False,training_ready=False,
                 remaining='Whole-source prior-overlap check, clinical label-scope review and mitigation of class-correlated orientation/camera/background; similarity components are not verified plant identities.')
    (out/'manifest.json').write_text(json.dumps(rows,indent=2));(out/'result.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
