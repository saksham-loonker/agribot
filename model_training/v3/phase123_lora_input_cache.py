"""Cache only the unchanged patch-embedding prefix before LoRA transformer blocks."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
import timm,torch
from partial_dino_cache import CanonicalInputs,original_dino
from phase114_new_supervision_features import original_features
from opened_context_data import read
from permissive_encoder import digest
from population_siglip_tail import freeze
from safe_feature_memmap import feature_memmap


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);out=p/'phase123_lora_input_cache';out.mkdir(exist_ok=True);started=time.monotonic()
    values,bags,descriptors,parent=original_features(p);local=p/'phase120_local_patch_cache';local_protocol=read(local/'protocol.json');local_complete=read(local/'complete.json')
    assert local_complete['protocol_sha256']==digest(local/'protocol.json') and local_complete['tokens_sha256']==digest(local/'tokens.npy')
    selected=local_protocol['selected_tables'];assert [descriptors[key]['input_sha256'] for key in selected]==local_protocol['input_sha256']
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;timm.layers.set_fused_attn(False)
    backbone=original_dino(p).cuda().eval();state=hashlib.sha256()
    for name,value in sorted(backbone.state_dict().items()):
        if name in ('cls_token','reg_token','pos_embed') or name.startswith(('patch_embed.','norm_pre.')):
            array=value.detach().cpu().contiguous().numpy();state.update(str((name,str(array.dtype),array.shape)).encode());state.update(array.tobytes())
    protocol=dict(source_sha256=digest(__file__),cache_helper_sha256=digest(Path(__file__).with_name('safe_feature_memmap.py')),parents=parent,
                  selected_tables=selected,input_sha256=local_protocol['input_sha256'],input_prefix_state_sha256=state.hexdigest(),representation='Original patch embedding, position, class token and norm_pre; before transformer block zero; FP32',
                  shape=[len(selected),257,384],torch_version=torch.__version__,timm_version=timm.__version__,all_transformer_blocks_excluded_from_cached_prefix=True,
                  ad_independent=False,olid_independent=False,acceptance_data_used=False)
    freeze(out/'protocol.json',protocol)
    if (out/'complete.json').exists():
        saved=read(out/'complete.json');assert saved['protocol_sha256']==digest(out/'protocol.json') and saved['tokens_sha256']==digest(out/'tokens.npy');return
    tokens=feature_memmap(out/'tokens.npy',protocol['shape']);progress=read(out/'progress.json') if (out/'progress.json').exists() else dict(completed=0)
    if (out/'progress.json').exists():assert progress['protocol_sha256']==digest(out/'protocol.json')
    pixels=CanonicalInputs(p)
    with torch.inference_mode():
        for start in range(progress['completed'],len(selected),16):
            part=selected[start:start+16];images=[]
            for key in part:
                row=descriptors[key];images.append(pixels(row if 'kind' in row else dict(row,kind='new',policy=row['view'])))
            x=torch.from_numpy(np.concatenate(images)).cuda().permute(0,3,1,2)
            prefix=backbone.norm_pre(backbone.patch_drop(backbone._pos_embed(backbone.patch_embed(x))))
            tokens[start:start+len(part)]=prefix.cpu().numpy();tokens.flush();progress.update(completed=start+len(part),protocol_sha256=digest(out/'protocol.json'))
            temporary=out/'progress.tmp';temporary.write_text(json.dumps(progress));temporary.replace(out/'progress.json')
        audit=np.linspace(0,len(selected)-1,16,dtype=int);final=backbone.norm(backbone.blocks(torch.from_numpy(np.array(tokens[audit],copy=True)).cuda())).cpu().numpy()
        previous=np.load(local/'tokens.npy',mmap_mode='r');error=float(np.abs(final-previous[audit]).max());assert error<.002,('Cached input-embedding prefix does not reconstruct verified original tokens',error)
    pixels.close();result=dict(protocol_sha256=digest(out/'protocol.json'),tokens_sha256=digest(out/'tokens.npy'),inputs=len(selected),frozen_input_embedding_passes=len(selected),
                              original_transformer_audit_inputs=16,maximum_final_token_audit_error=error,elapsed_seconds=time.monotonic()-started,ad_independent=False,olid_independent=False,acceptance_data_used=False)
    (out/'complete.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
