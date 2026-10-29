"""Exact whole-SigLIP prefix coverage including legitimate no-ROI supervision."""
from pathlib import Path
import argparse,json,shutil,time
import numpy as np,torch,timm
from PIL import Image
from app_eval import classifier_letterbox_input
from opened_context_data import read
from partial_primary_siglip import prefix_features,prefix_identity,PartialPrimarySiglip
from permissive_encoder import CanonicalPermissiveEncoder,digest
from phase28_roi_confirmation import pixel_key
from population_siglip_tail import freeze


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',required=True);p=Path(ap.parse_args().campaign);out=p/'phase246_whole_prefix_coverage';out.mkdir(exist_ok=True);started=time.monotonic()
    parent=p/'phase243_all_photo_context_supervision';cp=read(parent/'protocol.json');rows=read(p/'phase84_population_head/fitting_rows.json');descriptors={};whole={}
    for name,sha in cp['parents'].items():
        root=p/name;assert digest(root/'complete.json')==sha;cc=read(root/'complete.json');assert digest(root/'features.npy')==cc['features_sha256'] and digest(root/'inputs.json')==cc['inputs_sha256'];v=np.load(root/'features.npy',mmap_mode='r')
        for r,value in zip(read(root/'inputs.json'),v):descriptors[r['path']]=r;whole[r['path']]=value
    donor=p/'phase67_primary_siglip/prefix';dc=read(donor/'complete.json');dp=read(donor/'protocol.json');assert dc['protocol_sha256']==digest(donor/'protocol.json') and dc['inputs_sha256']==digest(donor/'inputs.json');lookup={};banks={}
    for shard in dc['shards']:
        path=donor/shard['file'];assert digest(path)==shard['sha256'];banks[str(path)]=np.load(path,mmap_mode='r')
    for j,r in enumerate(read(donor/'inputs.json')):
        shard=next(s for s in dc['shards'] if s['start']<=j<s['end']);lookup[r['input_sha256']]=dict(file=str(donor/shard['file']),index=j-shard['start'])
    extra=p/'phase85_population_sft/prefix';ep=read(extra/'protocol.json');ec=read(extra/'complete.json');assert ec['protocol_sha256']==digest(extra/'protocol.json') and ep['prefix_state_sha256']==dp['prefix_state_sha256']
    extra_keys=sorted({r['input_sha256'] for r in ep['inputs'].values()}-set(lookup));assert len(extra_keys)==ep['shape'][0]
    path=extra/'missing_tokens.npy'
    if not path.exists():path=extra/'tokens.npy'
    assert digest(path)==ec['missing_tokens_sha256'];banks[str(path)]=np.load(path,mmap_mode='r')
    lookup.update({key:dict(file=str(path),index=j) for j,key in enumerate(extra_keys)})
    # Keep all target-source training photos; bound old replay to existing exact prefixes.
    train=[i for i in cp['train_indices'] if rows[i]['source']=='olid_tomato' or descriptors[rows[i]['path']]['input_sha256'] in lookup];val=cp['validation_indices']
    keys=sorted({descriptors[rows[i]['path']]['input_sha256'] for i in train+val});missing=sorted(set(keys)-set(lookup));by_key={descriptors[rows[i]['path']]['input_sha256']:rows[i]['path'] for i in train+val}
    storage=Path('/run/user/1000/agribot-whole-prefix246');storage.mkdir(exist_ok=True);target=storage/'tokens.npy'
    protocol=dict(source_sha256=digest(__file__),parent_protocol_sha256=digest(parent/'protocol.json'),donor_complete_sha256=digest(donor/'complete.json'),extra_complete_sha256=digest(extra/'complete.json'),encoder_sha256=ep['encoder_sha256'],prefix_state_sha256=ep['prefix_state_sha256'],train_indices=train,validation_indices=val,keys=keys,missing=missing,storage=str(target),shape=[len(missing),256,768],
        cohort='All quarantined OLID training photos, including no-ROI photos. Old replay bounded to available exact whole-prefix identities, not full original replay.',
        new_prefix_inputs=len(missing),reused_prefix_inputs=len(keys)-len(missing),original_gates_unchanged=True,ad_independent=False,olid_independent=False,acceptance_data_used=False)
    freeze(out/'protocol.json',protocol)
    if (out/'complete.json').exists():assert read(out/'complete.json')['tokens_sha256']==digest(target);return
    assert shutil.disk_usage(storage).free>len(missing)*256*768*4+(200<<20)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;timm.layers.set_fused_attn(False)
    source=p/'phase23_siglip/siglip2_feature_heads/encoder.pt';assert digest(source)==ep['encoder_sha256'];state=torch.load(source,weights_only=True,map_location='cpu');encoder=CanonicalPermissiveEncoder(state['spec']);encoder.load_state_dict(state['model']);assert prefix_identity(encoder)==ep['prefix_state_sha256'];encoder=encoder.cuda().eval();tail=PartialPrimarySiglip(encoder).cuda().eval();del state
    maximum=0.;start=0;progress=out/'progress.json'
    if progress.exists():saved=read(progress);assert saved['protocol_sha256']==digest(out/'protocol.json');start=saved['completed'];maximum=saved['maximum_feature_error'];values=np.load(target,mmap_mode='r+')
    else:values=np.lib.format.open_memmap(target,mode='w+',dtype=np.float32,shape=tuple(protocol['shape']))
    for begin in range(start,len(missing),8):
        part=missing[begin:begin+8];pixels=[]
        for key in part:
            image_path=by_key[key];assert digest(image_path)==descriptors[image_path]['file_sha256']
            with Image.open(image_path) as im:tensor=classifier_letterbox_input(np.asarray(im.convert('RGB')),224)
            assert pixel_key(tensor)==key;pixels.append(tensor)
        with torch.inference_mode():tokens=prefix_features(encoder,torch.from_numpy(np.concatenate(pixels)).cuda());actual=tail.features(tokens).cpu().numpy()
        expected=np.stack([whole[by_key[key]] for key in part]);maximum=max(maximum,float(np.max(np.abs(actual-expected))));assert maximum<.0002
        values[begin:begin+len(part)]=tokens.cpu().numpy();values.flush();progress.write_text(json.dumps(dict(protocol_sha256=digest(out/'protocol.json'),completed=begin+len(part),maximum_feature_error=maximum)))
        print(json.dumps(dict(completed=begin+len(part),total=len(missing),maximum_feature_error=maximum)),flush=True)
    banks[str(target)]=values;lookup.update({key:dict(file=str(target),index=j) for j,key in enumerate(missing)})
    # Verify every reused and new representation before any adaptation.
    for begin in range(0,len(keys),16):
        part=keys[begin:begin+16];tensor=torch.from_numpy(np.stack([banks[lookup[k]['file']][lookup[k]['index']] for k in part])).cuda()
        with torch.inference_mode():actual=tail.features(tensor).cpu().numpy()
        maximum=max(maximum,float(np.max(np.abs(actual-np.stack([whole[by_key[k]] for k in part])))));assert maximum<.0002
    locations=[lookup[k] for k in keys];(out/'locations.json').write_text(json.dumps(locations));path_indices={rows[i]['path']:keys.index(descriptors[rows[i]['path']]['input_sha256']) for i in train+val};(out/'path_indices.json').write_text(json.dumps(path_indices))
    result=dict(protocol_sha256=digest(out/'protocol.json'),tokens_sha256=digest(target),locations_sha256=digest(out/'locations.json'),path_indices_sha256=digest(out/'path_indices.json'),maximum_feature_error=maximum,new_prefix_inputs=len(missing),reused_prefix_inputs=len(keys)-len(missing),elapsed_seconds=time.monotonic()-started,model_promoted=False,acceptance_data_used=False);(out/'complete.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
