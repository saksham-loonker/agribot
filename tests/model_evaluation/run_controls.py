"""Synthetic non-plant controls, reported separately from downloaded-image accuracy."""
import hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image
from evaluate_public import Models, OUT, ROOT, image_record

models=Models(1)
directory=OUT/'synthetic_controls'; directory.mkdir(exist_ok=True)
cases={'black':np.zeros((480,640,3),np.uint8),'white':np.full((480,640,3),255,np.uint8),
       'gray':np.full((480,640,3),128,np.uint8),'random_noise':np.random.default_rng(20261015).integers(0,256,(480,640,3),dtype=np.uint8)}
results=[]
for name,pixels in cases.items():
    path=directory/(name+'.png'); Image.fromarray(pixels).save(path)
    record=dict(id='synthetic_'+name,source='synthetic_control',source_path=name,source_url=None,
                label='Nonplant',split='synthetic_negative_control',path=str(path.relative_to(ROOT)),objects=[],
                width=640,height=480,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
    row=image_record(models,record); results.append(row)
    print(name,row['app_replay']['kind'],'detected:',len(row['detections']),flush=True)
(OUT/'synthetic_control_results.json').write_text(json.dumps(results,indent=2))
