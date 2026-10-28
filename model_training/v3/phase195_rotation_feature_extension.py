"""Cache only missing right-angle rotations of frozen full-replay ROI inputs."""
from pathlib import Path
from functools import lru_cache
import argparse,hashlib,json,shutil,time
import numpy as np
from PIL import Image
import torch,timm
from app_eval import canon,classifier_letterbox_input
from compact_healthy_model import primary
from exhaustive_replay_data import exhaustive_replay_data
from feature_cache_reuse import runtime_identity
from opened_context_data import read
from partial_dino_cache import CanonicalInputs
from permissive_encoder import digest
from population_siglip_tail import freeze
from safe_feature_memmap import feature_memmap


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True);p=Path(parser.parse_args().campaign);out=p/'phase195_rotation_feature_extension';out.mkdir(exist_ok=True);started=time.monotonic()
    if (out/'complete.json').exists():assert read(out/'complete.json')['source_sha256']==digest(__file__);return
    values,identities,rows,tables,train,val,meta=exhaustive_replay_data(p);descriptors=meta['original_descriptors'];base_count=len(values);donor={}
    for i,sha in enumerate(identities):
        if sha in donor:assert float(np.abs(values[donor[sha]]-values[i]).max())<.002
        else:donor[sha]=i
    source=read(p/'phase172_native_large_field_features/protocol.json');native={r['input_sha256']:r for r in source['inputs']};field_metadata={r['parent_224_input_sha256']:native[r['input_sha256']] for r in source['links']}
    pixels=CanonicalInputs(p)
    @lru_cache(maxsize=4)
    def image(path,sha):
        assert digest(path)==sha
        with Image.open(path) as im:return np.asarray(im.convert('RGB'))
    @lru_cache(maxsize=4)
    def raw(key):
        if key<len(descriptors):
            row=descriptors[key];value=pixels(row if 'kind' in row else dict(row,kind='new',policy=row['view']))
        else:
            assert key>=meta['field_offset'],'No continued source107 replay in this control'
            row=field_metadata[identities[key]];value=(canon.classifier_input if row['view']=='center' else classifier_letterbox_input)(image(row['path'],row['file_sha256']),224,row['box'])
        value=np.ascontiguousarray(value);assert hashlib.sha256(value.tobytes()).hexdigest()==identities[key]
        return value
    selected=sorted({key for i in train for view in tables[i][0] for key in view});work={};rotated={};new_keys=[];new_index={}
    for key in selected:
        rotated[key,0]=key
        for rotation in (1,2,3):
            value=np.ascontiguousarray(np.rot90(raw(key),rotation,axes=(1,2)));sha=hashlib.sha256(value.tobytes()).hexdigest()
            if sha in donor:index=donor[sha]
            else:
                if sha not in work:
                    work[sha]=dict(parent_index=key,parent_input_sha256=identities[key],rotation_quarters=rotation,input_sha256=sha);new_index[sha]=base_count+len(new_keys);new_keys.append(sha)
                index=new_index[sha]
            rotated[key,rotation]=index
    augmented={str(i):[[[rotated[key,rotation] for key in view] for view in tables[i][0]] for rotation in (0,1,2,3)] for i in train}
    augmented.update({str(i):tables[i] for i in val});freeze(out/'augmented_bags.json',augmented)
    protocol=dict(source_sha256=digest(__file__),data_source_sha256=digest(Path(__file__).with_name('exhaustive_replay_data.py')),full_replay_protocol_sha256=meta['full_replay_protocol_sha256'],original_parents=meta['original_parents'],base_count=base_count,
        original_weight_sha256=digest(p/'export_cnn_safe/pytorch_bundle.pt'),runtime=runtime_identity(),preprocessing_source_sha256=digest(canon.__file__),train_indices=train,validation_indices=val,selected_base_inputs=selected,new_inputs=[work[sha] for sha in new_keys],shape=[len(new_keys),1408],
        augmented_bags_sha256=digest(out/'augmented_bags.json'),augmentation='Exact 90/180/270-degree rotation of canonical square ROI pixels, no interpolation or color changes; one image label, no patch labels',validation_augmented=False,new_detector_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False)
    freeze(out/'protocol.json',protocol);assert shutil.disk_usage(out).free>len(new_keys)*1408*4+(32<<20)+(1<<30),'Preserve prefix/checkpoint caches and disk reserve'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;timm.layers.set_fused_attn(False);_,model=primary(p);model=model.cuda().eval()
    with torch.inference_mode():
        audit=selected[:8]+selected[-8:];actual=model.components(torch.from_numpy(np.concatenate([raw(key) for key in audit])).cuda())[0].cpu().numpy();error=float(np.abs(actual-values[audit]).max());assert error<.002,('Original mobile encoder restore differs before rotation',error)
    features=feature_memmap(out/'features.npy',[len(new_keys),1408]);progress=read(out/'progress.json') if (out/'progress.json').exists() else dict(completed=0,protocol_sha256=digest(out/'protocol.json'));assert progress['protocol_sha256']==digest(out/'protocol.json')
    with torch.inference_mode():
        for start in range(progress['completed'],len(new_keys),32):
            batch=[]
            for sha in new_keys[start:start+32]:
                row=work[sha];value=np.ascontiguousarray(np.rot90(raw(row['parent_index']),row['rotation_quarters'],axes=(1,2)));assert hashlib.sha256(value.tobytes()).hexdigest()==sha;batch.append(value)
            actual=model.components(torch.from_numpy(np.concatenate(batch)).cuda())[0].cpu().numpy();features[start:start+len(batch)]=actual;features.flush()
            temporary=out/'progress.tmp';temporary.write_text(json.dumps(dict(completed=start+len(batch),protocol_sha256=digest(out/'protocol.json'))));temporary.replace(out/'progress.json')
            if start%512==0 or start+len(batch)==len(new_keys):print(json.dumps(dict(stage='new_rotation_features',completed=start+len(batch),total=len(new_keys))),flush=True)
    pixels.close();features.flush();complete=dict(source_sha256=digest(__file__),protocol_sha256=digest(out/'protocol.json'),features_sha256=digest(out/'features.npy'),augmented_bags_sha256=digest(out/'augmented_bags.json'),new_rotated_encoder_inputs=len(new_keys),base_encoder_inputs_reused=len(selected),maximum_original_feature_error=error,new_detector_inputs=0,validation_augmented=False,model_promoted=False,acceptance_data_used=False,elapsed_seconds=time.monotonic()-started)
    (out/'complete.json').write_text(json.dumps(complete,indent=2));print(json.dumps(complete),flush=True)


if __name__=='__main__':main()
