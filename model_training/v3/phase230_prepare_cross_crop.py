"""Prepare verified auxiliary publisher pixels on the laptop, no model inference."""
from pathlib import Path
import argparse,hashlib,io,json,platform,time,zipfile
import cv2,numpy as np
from PIL import Image
from app_eval import canon,classifier_letterbox_input
from prepared_overlap import prepare_gray
from permissive_encoder import digest
from population_siglip_tail import freeze

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);out=Path(ap.parse_args().source);archive=out/'cucumber.zip';audit=json.loads((out/'archive_audit.json').read_text());assert archive.stat().st_size==audit['bytes'] and digest(archive)==audit['sha256'];lf=lambda path:hashlib.sha256(Path(path).read_text(encoding='utf-8').encode()).hexdigest();mapping={'cucumber__healthy':'Healthy','cucumber__N':'N','cucumber__K':'K','cucumber__N_K':'N_K'};protocol=dict(source_sha256=lf(__file__),archive_sha256=audit['sha256'],archive_publisher_md5=audit['md5'],label_mapping=mapping,role='auxiliary_training_only',crop='Cucumber',whole_collection_single_training_fold=True,app_eval_sha256=lf(Path(__file__).with_name('app_eval.py')),canonical_sha256=lf(canon.__file__),prepared_overlap_sha256=lf(Path(__file__).with_name('prepared_overlap.py')),opencv=cv2.__version__,numpy=np.__version__,python=platform.python_version(),input='Original publisher whole-leaf pixels, center/letterbox224 ImageNet0-1 normalization and exact existing600px original/mirror grayscale geometry preparation.',acceptance_data_used=False,ad_independent=False,olid_independent=False,training_used=False)
    freeze(out/'preprocessing_protocol.json',protocol)
    if (out/'input_manifest.json').exists():
        m=json.loads((out/'input_manifest.json').read_text());assert m['preprocessing_protocol_sha256']==digest(out/'preprocessing_protocol.json')
        for r in m['records']:assert digest(out/r['packet'])==r['packet_sha256']
        print(json.dumps(dict(already_complete=True,records=len(m['records']))));return
    packets=out/'packets';packets.mkdir(exist_ok=True);records=[];seen={};started=time.monotonic();cv2.setNumThreads(1)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.is_dir():continue
            assert info.filename in {r['path'] for r in audit['catalog']};folder=info.filename.split('/')[0];assert folder in mapping and info.file_size<20<<20;payload=z.read(info);sha=hashlib.sha256(payload).hexdigest();label=mapping[folder]
            if sha in seen:
                assert seen[sha]['label']==label,'Contradictory exact auxiliary image labels';seen[sha].setdefault('aliases',[]).append(info.filename);continue
            with Image.open(io.BytesIO(payload)) as im:rgb=np.asarray(im.convert('RGB'))
            views=np.concatenate([canon.classifier_input(rgb,224,None),classifier_letterbox_input(rgb,224,None)]).astype(np.float32);gray0=prepare_gray(payload);gray1=prepare_gray(payload,True);name='packets/'+sha+'.npz';np.savez_compressed(out/name,views=views,gray0=gray0,gray1=gray1);row=dict(id='olid_cucumber_aux:'+sha,member=info.filename,file_sha256=sha,bytes=len(payload),label=label,crop='Cucumber',y=int(label=='Healthy'),source='olid_cucumber_aux',group='OLID_CUCUMBER_ENTIRE_COLLECTION',path=str(archive)+'::'+info.filename,packet=name,packet_sha256=digest(out/name),input_sha256=[hashlib.sha256(np.ascontiguousarray(view[None]).tobytes()).hexdigest() for view in views]);records.append(row);seen[sha]=row
            if len(records)%32==0:print(json.dumps(dict(preprocessed=len(records),elapsed_seconds=time.monotonic()-started)),flush=True)
    result=dict(**{k:protocol[k] for k in ('archive_sha256','role','crop','app_eval_sha256','canonical_sha256','prepared_overlap_sha256','opencv','acceptance_data_used')},preprocessing_protocol_sha256=digest(out/'preprocessing_protocol.json'),records=records,training_used=False)
    (out/'input_manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(records=len(records),packet_bytes=sum((out/r['packet']).stat().st_size for r in records),elapsed_seconds=time.monotonic()-started)),flush=True)

if __name__=='__main__':main()
