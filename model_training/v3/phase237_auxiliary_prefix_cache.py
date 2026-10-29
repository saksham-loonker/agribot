"""Exact bounded prefix bank for auxiliary-aware copied DINO suffix adaptation."""
from pathlib import Path
from functools import lru_cache
import argparse,hashlib,json,math,os,shutil,time
import numpy as np,torch,timm
from torch.nn import functional as F
from PIL import Image
from app_eval import canon,classifier_letterbox_input
from exhaustive_replay_data import exhaustive_replay_data
from partial_dino_cache import original_dino,input_rows,CanonicalInputs
from partial_dino_healthy import prefix_identity,prefix_features,prefix_cut
from lossless_input_transport import unpack_views
from opened_context_data import read
from permissive_encoder import digest
from population_siglip_tail import freeze

def embedding_identity(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        if name in ('cls_token','reg_token','pos_embed') or name.startswith(('patch_embed.','norm_pre.')):
            a=value.detach().cpu().contiguous().numpy();h.update(str((name,str(a.dtype),a.shape)).encode());h.update(a.tobytes())
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',required=True);p=Path(ap.parse_args().campaign);out=p/'phase237_auxiliary_prefix_cache';out.mkdir(exist_ok=True);started=time.monotonic();base=Path('/run/user')/str(os.getuid())/'agribot-auxiliary-prefix237';assert base.resolve()==base and base.is_dir();transport=base/'prefix_transport.npz';tc=read(base/'prefix_transport.json');assert digest(transport)==tc['sha256'];aux_root=p/'phase231_cross_crop_features';aux_complete=read(aux_root/'complete.json');assert aux_complete['features_sha256']==digest(aux_root/'features.npy') and aux_complete['bags_sha256']==digest(aux_root/'bags.json') and aux_complete['identities_sha256']==digest(aux_root/'input_sha256.json') and aux_complete['overlap_images']==0 and not read(aux_root/'overlap_audit.json')['whole_collection_quarantined'];aux_protocol=read(aux_root/'protocol.json');assert tc['input_manifest_sha256']==aux_protocol['manifest_sha256']
    values,identities,original_rows,tables,full_train,val,meta=exhaustive_replay_data(p);native=p/'phase189_full_field_native_extension';np_=read(native/'protocol.json');native_rows_path=p/'phase168_early_cnn_prefix/rows.json';assert digest(native_rows_path)==np_['rows_sha256'];native_rows=read(native_rows_path);by_path={original_rows[i]['path']:i for i in full_train};selected=[by_path[native_rows[i]['path']] for i in np_['train_indices']]
    for ni,mi in zip(np_['train_indices'],selected):assert tuple(native_rows[ni][k] for k in ('y','label','group'))==tuple(original_rows[mi][k] for k in ('y','label','group'))
    prior_path=p/'phase215_shrinkage_density/native/protocol.json';prior=read(prior_path);assert prior['train_indices']==np_['train_indices'] and prior['validation_indices']==val;rows=[];bags={};old_to_new={};roles=[]
    for old in selected+val:
        new=len(rows);old_to_new[old]=new;rows.append(dict(original_rows[old],original_mobile_index=old));bags[new]=[[identities[k] for k in view] for view in tables[old][0]]
    train=list(range(len(selected)));validation=list(range(len(selected),len(rows)));native_position={i:j for j,i in enumerate(np_['train_indices'])};roles=[[native_position[i] for i in np_[name]] for name in ('target_indices','anchor_indices','field_indices')];assert sum(roles,[])==train
    all_values={key:values[i] for i,key in enumerate(identities)};aux_values=np.load(aux_root/'features.npy');aux_keys=read(aux_root/'input_sha256.json');assert not set(aux_keys)&set(all_values);all_values.update({key:aux_values[i] for i,key in enumerate(aux_keys)});aux_indices=[];family_labels={}
    for name,bag in sorted(read(aux_root/'bags.json').items()):
        record=bag['record'];family=hashlib.sha256(json.dumps(bag['views']).encode()).hexdigest();family_labels.setdefault(family,set()).add(record['y']);i=len(rows);aux_indices.append(i);rows.append(dict(path=record['path'],y=record['y'],label='Healthy' if record['y'] else 'Auxiliary_'+record['label'],source='olid_cucumber_aux',group='aux_exact_inputs:'+family,crop='Cucumber',collection='OLID_CUCUMBER_ENTIRE_COLLECTION',validation=False));bags[i]=[[key] for key in bag['views']]
    assert all(len(s)==1 for s in family_labels.values());keys=sorted({key for views in bags.values() for view in views for key in view});positions={key:i for i,key in enumerate(keys)};old_cache=p/'phase51_dino_adaptation/prefix';old_complete=read(old_cache/'complete.json');old_protocol=read(old_cache/'protocol.json');old_rows=input_rows(p);old_ids=np.load(old_cache/'tables.npy');old_lookup={old_rows[int(i)]['input_sha256']:j for j,i in enumerate(old_ids)};old_tokens=np.load(old_cache/'features.npy',mmap_mode='r');zero=p/'phase133_anchor_input_cache';zc=read(zero/'complete.json');zp=read(zero/'protocol.json');assert zc['protocol_sha256']==digest(zero/'protocol.json') and zc['tokens_sha256']==digest(zero/'tokens.npy');zero_tokens=np.load(zero/'tokens.npy',mmap_mode='r');zero_lookup={key:i for i,key in enumerate(zp['input_sha256'])};torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;timm.layers.set_fused_attn(False);model=original_dino(p).cuda().eval().requires_grad_(False);state=prefix_identity(model);assert old_complete['prefix_state_sha256']==state and embedding_identity(model)==zp['input_prefix_state_sha256'];borrowed=sorted(set(keys)&set(old_lookup));new_keys=sorted(set(keys)-set(old_lookup));required=len(new_keys)*257*384*4;memory=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024;assert memory>required+(3<<30);assert (base/'tokens.npy').exists() or shutil.disk_usage(base).free>required+(512<<20)
    protocol=dict(source_sha256=digest(__file__),original_bundle_sha256=digest(p/'export_cnn_safe/pytorch_bundle.pt'),full_replay_protocol_sha256=meta['full_replay_protocol_sha256'],bounded_split_protocol_sha256=digest(prior_path),auxiliary_complete_sha256=digest(aux_root/'complete.json'),transport_sha256=tc['sha256'],transport_codec_sha256=digest(Path(__file__).with_name('lossless_input_transport.py')),prefix_state_sha256=state,prefix0_source_complete_sha256=digest(zero/'complete.json'),prefix10_source_complete_sha256=digest(old_cache/'complete.json'),train_indices=train,validation_indices=validation,validation_original_indices=val,auxiliary_indices=aux_indices,role_indices=roles,cv_groups=prior['cv_groups'],calibration_folds=prior['calibration_folds'],input_sha256=keys,new_prefix_input_sha256=new_keys,borrowed_prefix_input_sha256=borrowed,new_prefix_shape=[len(new_keys),257,384],new_prefix_bytes=required,representation='FP32 original DINO-small tokens before final2blocks. Exact matching prefix10 rows reused; matching prefix0 input embeddings reused; only absent prefix representations computed. Existing finalCLS features cannot reconstruct patch tokens.',storage=str(base/'tokens.npy'),storage_volatile=True,scope='Bounded original977-photo cohort plus entire auxiliary collection; no claim of full original encoder replay. Original frozen feature bank retained for reference and unchanged disease path.',auxiliary_training_role='Entire auxiliary collection stays on training side of each original OOF fit; its labels never calibrate app probabilities. No internal auxiliary train/validation split or independence claim.',original_gates_unchanged=True,acceptance_data_used=False,ad_independent=False,olid_independent=False)
    freeze(out/'protocol.json',protocol)
    if (out/'complete.json').exists():
        r=read(out/'complete.json');assert r['protocol_sha256']==digest(out/'protocol.json') and r['tokens_sha256']==digest(base/'tokens.npy');print(json.dumps(dict(already_complete=True)));return
    with np.load(transport,allow_pickle=False) as packed:aux_rgb=unpack_views(packed['planes'],packed['shape']);transport_ids=packed['identities'].tolist()
    assert transport_ids==aux_keys
    for key,value in zip(transport_ids,aux_rgb):assert hashlib.sha256(value[None].tobytes()).hexdigest()==key
    aux_lookup={key:i for i,key in enumerate(transport_ids)};descriptors={r['input_sha256']:r for r in meta['original_descriptors']};native_meta={r['input_sha256']:r for r in read(p/'phase172_native_large_field_features/protocol.json')['inputs']+np_['new_native_inputs']};field_meta={r['parent_224_input_sha256']:native_meta[r['input_sha256']] for r in np_['links']};pixels=CanonicalInputs(p)
    @lru_cache(maxsize=8)
    def image(path,sha):
        assert digest(path)==sha
        with Image.open(path) as im:return np.asarray(im.convert('RGB'))
    def pixel(key):
        if key in aux_lookup:return aux_rgb[aux_lookup[key]:aux_lookup[key]+1]
        if key in descriptors:
            r=descriptors[key];return pixels(r if 'kind' in r else dict(r,kind='new',policy=r['view']))
        r=field_meta[key];rgb=image(r['path'],r['file_sha256']);value=(canon.classifier_input if r['view']=='center' else classifier_letterbox_input)(rgb,224,r['box']);assert hashlib.sha256(value.tobytes()).hexdigest()==key;return value
    target=base/'tokens.npy';array=np.lib.format.open_memmap(target,mode='r+' if target.exists() else 'w+',dtype=np.float32,shape=(len(new_keys),257,384));progress=read(out/'progress.json') if (out/'progress.json').exists() else dict(completed=0,reused_input_embeddings=0,new_input_embeddings=0,protocol_sha256=digest(out/'protocol.json'));assert progress['protocol_sha256']==digest(out/'protocol.json')
    try:
        with torch.inference_mode():
            for start in range(progress['completed'],len(new_keys),32):
                part=new_keys[start:start+32];stem=np.empty((len(part),257,384),np.float32);missing=[];rgb=[]
                for j,key in enumerate(part):
                    if key in zero_lookup:stem[j]=zero_tokens[zero_lookup[key]];progress['reused_input_embeddings']+=1
                    else:missing.append(j);rgb.append(pixel(key));progress['new_input_embeddings']+=1
                if missing:
                    x=torch.from_numpy(np.concatenate(rgb)).cuda().permute(0,3,1,2);stem[missing]=model.norm_pre(model.patch_drop(model._pos_embed(model.patch_embed(x)))).cpu().numpy()
                array[start:start+len(part)]=model.blocks[:prefix_cut(model)](torch.from_numpy(stem).cuda()).cpu().numpy();array.flush();progress['completed']=start+len(part);(out/'progress.json').write_text(json.dumps(progress,indent=2))
                if start%256==0:print(json.dumps(dict(stage='new_prefix_rows',**progress)),flush=True)
            maximum=0.;reference_prefix_error=0.
            for key in borrowed:
                if key in zero_lookup:expected=model.blocks[:prefix_cut(model)](torch.from_numpy(np.array(zero_tokens[zero_lookup[key]:zero_lookup[key]+1],copy=True)).cuda())
                else:expected=prefix_features(model,torch.from_numpy(pixel(key)).cuda().permute(0,3,1,2))
                error=float(abs(expected.cpu().numpy()-old_tokens[old_lookup[key]:old_lookup[key]+1]).max());reference_prefix_error=max(reference_prefix_error,error);assert error<.002
            new_position={key:i for i,key in enumerate(new_keys)}
            for start in range(0,len(keys),32):
                part=keys[start:start+32];a=np.stack([old_tokens[old_lookup[k]] if k in old_lookup else array[new_position[k]] for k in part]);raw=model.norm(model.blocks[prefix_cut(model):](torch.from_numpy(a).cuda()));cls=model.forward_head(raw,pre_logits=True);actual=(F.normalize(cls,dim=1)*math.sqrt(384)).cpu().numpy();expected=np.stack([all_values[k][:384] for k in part]);maximum=max(maximum,float(abs(actual-expected).max()));assert maximum<.002,('Cached prefix does not restore original features',maximum)
    finally:pixels.close();image.cache_clear()
    array.flush();table=np.zeros((len(rows),2,6),np.int64);mask=np.zeros(table.shape,np.float32)
    for i,views in bags.items():
        for view,ids in enumerate(views):assert 0<len(ids)<=6;table[i,view,:len(ids)]=[positions[k] for k in ids];mask[i,view,:len(ids)]=1.
    np.savez(out/'tables.npz',indices=table,mask=mask);np.save(out/'reference_features.npy',np.stack([all_values[k] for k in keys]));(out/'rows.json').write_text(json.dumps(rows,indent=2));(out/'prefix_locations.json').write_text(json.dumps([dict(bank='original',index=old_lookup[k]) if k in old_lookup else dict(bank='new',index=new_position[k]) for k in keys]));result=dict(protocol_sha256=digest(out/'protocol.json'),tokens_sha256=digest(target),tables_sha256=digest(out/'tables.npz'),reference_features_sha256=digest(out/'reference_features.npy'),rows_sha256=digest(out/'rows.json'),locations_sha256=digest(out/'prefix_locations.json'),old_prefix_rows_reused=len(borrowed),new_prefix_rows=len(new_keys),new_prefix_bytes=required,maximum_original_cls_error=maximum,reference_prefix_error=reference_prefix_error,reference_prefix_rows_verified=len(borrowed),original_cls_rows_verified=len(keys),**{k:v for k,v in progress.items() if k!='protocol_sha256'},elapsed_seconds=time.monotonic()-started,acceptance_data_used=False,model_promoted=False,cache_volatile=True);(out/'complete.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
