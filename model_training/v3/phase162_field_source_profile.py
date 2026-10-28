"""Decode/profile verified new field photos; no detector or classifier calls."""
from pathlib import Path
import argparse,collections,concurrent.futures,hashlib,json,time,zlib
import cv2
from phase112_development_source_audit import profile
from phase106_cross_source_overlap import digest,frozen


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);source=p/'phase161_field_supervision_source';out=p/'phase162_field_source_profile';out.mkdir(exist_ok=True);started=time.monotonic()
    complete=json.loads((source/'download_complete.json').read_text());sp=json.loads((source/'protocol.json').read_text());assert complete['all_publisher_hashes_verified'] and complete['protocol_sha256']==digest(source/'protocol.json')
    protocol=dict(source_sha256=digest(__file__),profile_source_sha256=digest(Path(__file__).with_name('phase112_development_source_audit.py')),source_protocol_sha256=digest(source/'protocol.json'),
        download_complete_sha256=digest(source/'download_complete.json'),opencv_version=cv2.__version__,source_role='development_only',unit='One original publisher photograph, not detector ROI or augmented rotation',
        family='Rotation/mirror canonical pHash Hamming <=6 connected components; approximate photo similarity only, never verified specimen identity',
        quarantine='Entire inferred family when an exact decoded pixel identity has conflicting Healthy/disease labels or distinct named clinical targets. Coarse disease is compatible with any named non-Healthy target.',
        publisher_listing_scope='Ten explicitly labelled class folders; no claim that these exhaust unlabelled files or the publisher description Other category',training_used=False,model_predictions_opened=False,acceptance_data_used=False)
    frozen(out/'protocol.json',protocol)
    if (out/'result.json').exists():assert json.loads((out/'result.json').read_text())['protocol_sha256']==digest(out/'protocol.json');return
    records=[]
    for member in sp['selection']:
        path=source/member['relative_path'];value=path.read_bytes();assert len(value)==member['bytes'] and hashlib.sha256(value).hexdigest()==member['sha256']
        records.append(dict(id='field93:'+member['id'],path=str(path),member=member['relative_path'],publisher_filename=member['filename'],publisher_class=member['publisher_class'],publisher_sha256=member['sha256'],label=member['label'],crc32=zlib.crc32(value)))
    cv2.setNumThreads(1)
    with concurrent.futures.ThreadPoolExecutor(4) as pool:rows=list(pool.map(profile,records))
    assert all(row['file_sha256']==row['publisher_sha256'] for row in rows)
    by_pixels=collections.defaultdict(list)
    for row in rows:by_pixels[(row['width'],row['height'],row['decoded_rgb_sha256'])].append(row)
    conflicts=set()
    for identity,group in by_pixels.items():
        labels={r['label'] for r in group};specific=labels-{'Diseased_unspecified'}
        if ('Healthy' in labels and len(labels)>1) or len(specific)>1:conflicts.add(identity)
    parents=list(range(len(rows)))
    def root(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    codes=[int(r['rotation_mirror_phash'],16) for r in rows];edges=0
    for i in range(len(rows)):
        for j in range(i):
            if (codes[i]^codes[j]).bit_count()<=6:parents[root(i)]=root(j);edges+=1
    groups=collections.defaultdict(list)
    for i,row in enumerate(rows):groups[root(i)].append(row)
    for group in groups.values():
        key=hashlib.sha256('\n'.join(sorted({r['file_sha256'] for r in group})).encode()).hexdigest()[:20]
        quarantine=any((r['width'],r['height'],r['decoded_rgb_sha256']) in conflicts for r in group)
        for row in group:row.update(group='field93:'+key,collection_group='field93:entire_publisher_collection',source='field93',source_version='10.17632/93h9p62kg4.1',licence='CC BY 4.0',quarantined=quarantine)
    (out/'manifest.json').write_text(json.dumps(rows,indent=2))
    result=dict(protocol_sha256=digest(out/'protocol.json'),manifest_sha256=digest(out/'manifest.json'),images=len(rows),label_counts=dict(collections.Counter(r['label'] for r in rows)),
        publisher_class_counts=dict(collections.Counter(r['publisher_class'] for r in rows)),file_unique=len({r['file_sha256'] for r in rows}),decoded_unique=len(by_pixels),exact_label_conflict_pixel_identities=len(conflicts),
        inferred_photo_families=len(groups),similarity_edges=edges,largest_inferred_family=max(map(len,groups.values())),quarantined_images=sum(r['quarantined'] for r in rows),
        by_publisher_class={label:dict(images=len(part),orientation=dict(collections.Counter(r['orientation'] for r in part)),camera=dict(collections.Counter(r['camera'] for r in part)),missing_capture_time=sum(not r['capture_time'] for r in part),exif_orientations=dict(collections.Counter(r['exif_orientation'] for r in part)),inferred_families=len({r['group'] for r in part})) for label in sorted({r['publisher_class'] for r in rows}) for part in [[r for r in rows if r['publisher_class']==label]]},
        source_role='development_only',training_ready=False,training_used=False,model_predictions_opened=False,independence_established=False,acceptance_ready=False,
        remaining='Cross-source and protected-reservation exact/geometric overlap audit before any fitting; no random source split; no disease target inferred for individual leaves.',elapsed_seconds=time.monotonic()-started)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
