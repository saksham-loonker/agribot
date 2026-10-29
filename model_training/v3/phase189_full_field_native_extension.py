"""Extend native/texture evidence only for omitted quarantined field supervision."""
from pathlib import Path
from functools import lru_cache
import argparse,hashlib,json,shutil,time
import numpy as np
from PIL import Image
import torch,timm
from torch.nn import functional as F
from torchvision.models import convnext_tiny
from app_eval import canon,classifier_letterbox_input
from dino_large_encoder import SOURCE_URL,SOURCE_SHA256
from feature_cache_reuse import runtime_identity
from native_joint_features import join_native_cnn
from opened_context_data import read
from phase114_new_supervision_features import original_features
from permissive_encoder import digest
from population_siglip_tail import freeze


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True);p=Path(parser.parse_args().campaign);out=p/'phase189_full_field_native_extension';out.mkdir(exist_ok=True);started=time.monotonic()
    if (out/'complete.json').exists():assert read(out/'complete.json')['source_sha256']==digest(__file__);return
    old=p/'phase172_native_large_field_features';op=read(old/'protocol.json');oc=read(old/'complete.json');assert oc['protocol_sha256']==digest(old/'protocol.json') and oc['features_sha256']==digest(old/'features.npy')
    rows=read(p/'phase168_early_cnn_prefix/rows.json');assert digest(p/'phase168_early_cnn_prefix/rows.json')==op['rows_sha256']
    omitted=read(p/'phase178_guard_coverage_audit/result.json')['training_coverage']['omitted_indices'];assert len(omitted)==71 and all(not rows[i]['validation'] for i in omitted)
    population=read(p/'phase168_early_cnn_prefix/protocol.json');target=population['target_indices'];anchors=population['anchor_indices']+omitted;field_indices=population['field_indices'];train=target+anchors+field_indices;val=op['validation_indices']
    assert len(set(train))==len(train) and not {rows[i]['group'] for i in train}&{rows[i]['group'] for i in val}
    conv=p/'phase154_convnext_features';cp=read(conv/'protocol.json');cc=read(conv/'complete.json');assert cc['features_sha256']==digest(conv/'features.npy') and digest(conv/'complete.json')==op['convnext_complete_sha256']
    texture={sha:value for sha,value in zip(cp['input_sha256'],np.load(conv/'features.npy'))};field=p/'phase164_new_field_features';fc=read(field/'complete.json');assert digest(field/'complete.json')==op['field_complete_sha256'] and fc['features_sha256']==digest(field/'features.npy') and fc['input_sha256_sha256']==digest(field/'input_sha256.json')
    texture.update({sha:value[2432:] for sha,value in zip(read(field/'input_sha256.json'),np.load(field/'features.npy'))})
    native={row['input_sha256']:value for row,value in zip(op['inputs'],np.load(old/'features.npy'))};old_joint=join_native_cnn(np.load(old/'features.npy'),[r['input_sha256'] for r in op['inputs']],op['links'],texture)
    _,bags,descriptors,_=original_features(p);file_sha={r['path']:r['file_sha256'] for r in read(p/'phase24_olid_source/sealed_manifest.json')};pairs=dict(op['bags']);links=list(op['links']);link_lookup={(r['input_sha256'],r['parent_224_input_sha256']):i for i,r in enumerate(links)};metadata={r['input_sha256']:dict(r) for r in op['inputs']};low_metadata={};new_native={};new_texture={}
    @lru_cache(maxsize=4)
    def image(path,sha):
        assert digest(path)==sha
        with Image.open(path) as im:return np.asarray(im.convert('RGB'))
    def pixels(row,size):return (canon.classifier_input if row['view']=='center' else classifier_letterbox_input)(image(row['path'],row['file_sha256']),size,row['box'])
    for i in omitted:
        row=rows[i];path=row['path'];assert path not in pairs;pairs[path]=[[],[]]
        for view,name in enumerate(('center','full')):
            for key in bags[path][name]:
                descriptor=descriptors[key];entry=dict(path=path,file_sha256=file_sha[path],box=descriptor['box'],view=name)
                low=pixels(entry,224);parent=hashlib.sha256(np.ascontiguousarray(low).tobytes()).hexdigest();assert parent==descriptor['input_sha256']
                value=pixels(entry,518);sha=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest();entry['input_sha256']=sha;metadata.setdefault(sha,entry);low_metadata.setdefault(parent,entry)
                if sha not in native:new_native.setdefault(sha,entry)
                if parent not in texture:new_texture.setdefault(parent,entry)
                link=(sha,parent)
                if link not in link_lookup:link_lookup[link]=len(links);links.append(dict(input_sha256=sha,parent_224_input_sha256=parent))
                pairs[path][view].append(link_lookup[link])
    identities=lambda ids:{links[key]['input_sha256'] for i in ids for view in pairs[rows[i]['path']] for key in view}
    assert not identities(train)&identities(val)
    protocol=dict(source_sha256=digest(__file__),original_native_complete_sha256=digest(old/'complete.json'),convnext_complete_sha256=digest(conv/'complete.json'),field_complete_sha256=digest(field/'complete.json'),rows_sha256=op['rows_sha256'],coverage_audit_sha256=digest(p/'phase178_guard_coverage_audit/result.json'),
        train_indices=train,validation_indices=val,target_indices=target,anchor_indices=anchors,field_indices=field_indices,added_training_indices=omitted,links=links,bags=pairs,new_native_inputs=list(new_native.values()),new_texture_parent_inputs=[dict(parent_input_sha256=sha,**row) for sha,row in new_texture.items()],shape=[len(links),1792],runtime=runtime_identity(),original_gates_unchanged=True,new_detector_inputs=0,acceptance_data_used=False,ad_independent=False,olid_independent=False)
    assert protocol['runtime']==op['runtime'];freeze(out/'protocol.json',protocol)
    assert shutil.disk_usage(out).free>len(links)*1792*4+(1<<30),'Preserve existing runs and one GiB disk reserve'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;timm.layers.set_fused_attn(False)
    official=read(p/'phase23_dino/encoder_source.json');assert official['url']==SOURCE_URL and digest(official['local_file'])==SOURCE_SHA256
    dino=timm.create_model(official['model_name'],pretrained=True,img_size=518,dynamic_img_size=True,num_classes=0,pretrained_cfg_overlay=dict(file=official['local_file'],hf_hub_id=None,license='apache-2.0')).cuda().eval().requires_grad_(False)
    def encode_native(rows_):
        x=torch.from_numpy(np.concatenate([pixels(r,518) for r in rows_])).cuda().permute(0,3,1,2).contiguous(memory_format=torch.channels_last)
        return (F.normalize(dino(x),dim=1)*(1024**.5)).cpu().numpy()
    with torch.inference_mode():
        audit=op['inputs'][:8];actual=encode_native(audit);error=float(np.abs(actual-np.stack([native[r['input_sha256']] for r in audit])).max());assert error<.002
        keys=list(new_native)
        for start in range(0,len(keys),2):
            part=keys[start:start+2];actual=encode_native([new_native[key] for key in part]);native.update(zip(part,actual))
    del dino;torch.cuda.empty_cache()
    weights=Path(torch.hub.get_dir())/'checkpoints'/cp['pretrained_url'].rsplit('/',1)[1];assert digest(weights)==cp['pretrained_sha256']
    cnn=convnext_tiny(weights=None).cuda().eval().requires_grad_(False);cnn.load_state_dict(torch.load(weights,weights_only=True,map_location='cpu'),strict=True)
    def encode_texture(rows_):
        x=torch.from_numpy(np.concatenate([pixels(r,224) for r in rows_])).cuda().permute(0,3,1,2).contiguous();prefix=cnn.features[0](x).contiguous(memory_format=torch.channels_last)
        return cnn.avgpool(cnn.features[1:](prefix)).flatten(1).cpu().numpy()
    with torch.inference_mode():
        audit=op['links'][:8];entries=[metadata[r['input_sha256']] for r in audit];actual=encode_texture(entries);conv_error=float(np.abs(actual-np.stack([texture[r['parent_224_input_sha256']] for r in audit])).max());assert conv_error<.002
        keys=list(new_texture)
        for start in range(0,len(keys),8):
            part=keys[start:start+8];actual=encode_texture([new_texture[key] for key in part]);texture.update(zip(part,actual))
    values=np.stack([np.concatenate((native[r['input_sha256']],texture[r['parent_224_input_sha256']])) for r in links]);assert np.array_equal(values[:len(old_joint)],old_joint)
    np.save(out/'features.npy',values)
    result=dict(source_sha256=digest(__file__),protocol_sha256=digest(out/'protocol.json'),features_sha256=digest(out/'features.npy'),new_native_encoder_inputs=len(new_native),new_convnext_encoder_inputs=len(new_texture),old_joint_features_reused=len(old_joint),added_training_images=71,added_training_healthy=66,maximum_native_restore_error=error,maximum_convnext_restore_error=conv_error,new_detector_inputs=0,model_promoted=False,acceptance_data_used=False,elapsed_seconds=time.monotonic()-started)
    (out/'complete.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
