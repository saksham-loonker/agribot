"""Publisher-verified, bounded new field supervision; no model predictions.

The entire collection is designated development-only before downloading images.
No remainder, author split or near-copy family is independent acceptance.
"""
from pathlib import Path
import argparse,collections,concurrent.futures,hashlib,json,shutil,time
import requests

DOI='10.17632/93h9p62kg4.1'
SOURCE='https://data.mendeley.com/datasets/93h9p62kg4/1'
BASE='https://data.mendeley.com/public-api/datasets/93h9p62kg4'
LABELS={'Healthy':'Healthy','Early Blight':'Early_blight','Late Blight':'Late_blight','Leaf Miner':'Leaf Miner',
    'Bacterial Spot':'Diseased_unspecified','Cercospora leaf mold':'Diseased_unspecified','Insect Damage':'Diseased_unspecified','Leaf Mold':'Diseased_unspecified','Spider Mites':'Diseased_unspecified','Tomato Leaf Curl Virus':'Diseased_unspecified'}


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def freeze(path,value):
    text=json.dumps(value,sort_keys=True,indent=2)
    if path.exists():assert path.read_text()==text,('Source contract changed',str(path))
    else:path.write_text(text)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-root',required=True);parser.add_argument('--inventory-only',action='store_true')
    args=parser.parse_args();out=Path(args.output_root);out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    freeze(out/'purpose.json',dict(source_doi=DOI,source_url=SOURCE,licence='CC BY 4.0',role='Entire publisher collection is development-only; never independent acceptance',
        no_random_source_holdout=True,labels=LABELS,coarse_disease_not_other=True,clinical_ROI_labels_inferred=False,publisher_labels_unverified_clinically=True,
        requirements='Verify publisher file hashes, decode integrity, class/family confounds and exact/near-copy overlap against all old sources and protected reservations before training. Preserve existing gates; no sealed acceptance predictions.',
        diagnostic_predictions_opened=False,training_used=False))
    inventory_path=out/'inventory.json'
    if inventory_path.exists():inventory=json.loads(inventory_path.read_text())
    else:
        response=requests.get(BASE+'/folders/1',timeout=(15,45));response.raise_for_status();folders=response.json();assert isinstance(folders,list)
        roots={row['id']:row for row in folders};assert len(roots)==len(folders)
        def folder_files(folder):
            values=[];start=0
            while True:
                response=requests.get(BASE+'/files',params={'folder_id':folder['id'],'version':1,'$start':start,'$limit':1000},headers={'Accept':'application/vnd.mendeley-public-dataset.1+json'},timeout=(15,45));response.raise_for_status();part=response.json();assert isinstance(part,list)
                values.extend(dict(item,publisher_class=folder['name']) for item in part)
                if len(part)<1000:return values
                start+=len(part);assert start<10000,'Unexpected source size'
        selected_folders=[folder for folder in folders if folder['name'] in LABELS]
        assert {folder['name'] for folder in selected_folders}==set(LABELS)
        with concurrent.futures.ThreadPoolExecutor(4) as pool:files=[item for part in pool.map(folder_files,selected_folders) for item in part]
        assert files and len({item['id'] for item in files})==len(files)
        inventory=dict(source_doi=DOI,folders=folders,files=files);inventory_path.write_text(json.dumps(inventory,indent=2))
    candidates=[]
    for item in inventory['files']:
        extension=Path(item['filename']).suffix.lower()
        if extension not in ('.jpg','.jpeg','.png','.bmp'):continue
        details=item['content_details'];size=int(details['size']);sha=details['sha256_hash']
        assert 0<size<20<<20 and len(sha)==64 and all(c in '0123456789abcdef' for c in sha)
        assert item['publisher_class'] in LABELS and item['folder_id'] in {row['id'] for row in inventory['folders'] if row['name']==item['publisher_class']}
        assert len(item['id'])==36 and all(c in '0123456789abcdef-' for c in item['id'])
        candidates.append(item)
    selection=[]
    for category in LABELS:
        group=sorted([r for r in candidates if r['publisher_class']==category],key=lambda r:hashlib.sha256(('phase161:'+r['id']).encode()).hexdigest())
        assert group,('Missing intended class',category)
        selection.extend(group[:256 if category=='Healthy' else 32])
    total=sum(int(row['content_details']['size']) for row in selection);assert total<=256<<20,'Keep source pilot/storage bounded'
    protocol=dict(source_sha256=digest(__file__),purpose_sha256=digest(out/'purpose.json'),inventory_sha256=digest(inventory_path),
        source_doi=DOI,licence='CC BY 4.0',selection_seed='phase161',healthy_cap=256,disease_cap_per_publisher_class=32,
        selection=[dict(id=r['id'],filename=r['filename'],publisher_class=r['publisher_class'],label=LABELS[r['publisher_class']],sha256=r['content_details']['sha256_hash'],bytes=int(r['content_details']['size']),relative_path='images/'+r['id']+Path(r['filename']).suffix.lower()) for r in selection],
        total_selected_bytes=total,source_role='development_only',ad_independent=False,olid_independent=False,acceptance_data_used=False,model_predictions_opened=False,training_used=False)
    freeze(out/'protocol.json',protocol)
    counts=dict(collections.Counter(row['publisher_class'] for row in candidates));chosen=dict(collections.Counter(row['publisher_class'] for row in selection))
    print(json.dumps(dict(stage='field_source_inventory',publisher_classes=counts,selected_classes=chosen,selected_images=len(selection),selected_bytes=total,source_role='development_only')),flush=True)
    if args.inventory_only:return
    if (out/'download_complete.json').exists():
        complete=json.loads((out/'download_complete.json').read_text());assert complete['protocol_sha256']==digest(out/'protocol.json')
        for row in protocol['selection']:assert digest(out/row['relative_path'])==row['sha256']
        return
    assert shutil.disk_usage(out).free>total+(1<<30),'Preserve existing cache/storage reserve'
    (out/'images').mkdir(exist_ok=True);by_id={r['id']:r for r in selection}
    def download(row):
        target=out/row['relative_path']
        if target.exists():assert target.stat().st_size==row['bytes'] and digest(target)==row['sha256'];return row['id']
        partial=target.with_suffix('.partial')
        for attempt in range(3):
            try:
                with requests.get(by_id[row['id']]['content_details']['download_url'],stream=True,timeout=(15,45)) as response:
                    response.raise_for_status();size=0
                    with partial.open('wb') as stream:
                        for chunk in response.iter_content(1<<20):
                            size+=len(chunk);assert size<=row['bytes'],'Oversized publisher image';stream.write(chunk)
                assert size==row['bytes'] and digest(partial)==row['sha256'],'Publisher image digest mismatch'
                partial.replace(target);return row['id']
            except (requests.RequestException,OSError):
                if attempt==2:raise
                time.sleep(1)
    completed=0
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        for _ in pool.map(download,protocol['selection']):
            completed+=1
            if completed%32==0 or completed==len(selection):print(json.dumps(dict(stage='verified_field_download',completed=completed,total=len(selection))),flush=True)
    result=dict(protocol_sha256=digest(out/'protocol.json'),images=completed,all_publisher_hashes_verified=True,publisher_image_counts=counts,selected_image_counts=chosen,
        source_role='development_only',training_ready=False,model_predictions_opened=False,training_used=False,independence_established=False,acceptance_ready=False,elapsed_seconds=time.monotonic()-started)
    (out/'download_complete.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
