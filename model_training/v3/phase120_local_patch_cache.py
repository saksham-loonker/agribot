"""Recover missing local patch representations from verified cached DINO prefixes."""
from pathlib import Path
import argparse,hashlib,json,math,shutil,time
import numpy as np
import torch,timm
from torch.nn import functional as F
from diagnostic_bag_cache_v2 import prepare_tokens
from opened_context_data import read
from permissive_encoder import digest
from phase114_new_supervision_features import original_features
from population_siglip_tail import freeze
from local_token_cache_identity import token_calculation_identity


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);out=p/'phase120_local_patch_cache';out.mkdir(exist_ok=True);started=time.monotonic()
    values,bags,descriptors,identity=original_features(p);rows=read(p/'phase84_population_head/fitting_rows.json');conditional=read(p/'phase118_conditional_recovery/protocol.json')
    indices=conditional['train_indices']+conditional['validation_indices']
    selected=sorted({key for i in indices for view in bags[rows[i]['path']].values() for key in view})
    protocol=dict(source_sha256=digest(__file__),parents=identity,conditional_protocol_sha256=digest(p/'phase118_conditional_recovery/protocol.json'),
                  prefix_cache_source_sha256=digest(Path(__file__).with_name('diagnostic_bag_cache_v2.py')),
                  selected_tables=selected,input_sha256=[descriptors[key]['input_sha256'] for key in selected],representation='Original final normalized CLS and all 256 patch tokens, FP32; no adapted encoder weights',
                  shape=[len(selected),257,384],encoder_prefix_inputs_recomputed=0,old_cls_features_not_substituted_for_patch_tokens=True,
                  original_model_bundle_sha256=digest(p/'export_cnn_safe/pytorch_bundle.pt'),ad_independent=False,olid_independent=False,acceptance_data_used=False)
    if (out/'complete.json').exists():
        saved=read(out/'complete.json');stored=read(out/'protocol.json')
        assert saved['protocol_sha256']==digest(out/'protocol.json') and saved['tokens_sha256']==digest(out/'tokens.npy')
        assert all(stored[key]==value for key,value in protocol.items() if key!='source_sha256'),'Cached input/model/representation identity changed'
        if stored['source_sha256']!=protocol['source_sha256']:
            # This specific repaired cache has a preserved pre-repair source,
            # unchanged payload proof and the exact same extraction AST.
            repair=read(out/'header_repair.json');original=read(out/'protocol.before_header_fix.json')
            assert stored['source_sha256']==repair['new_source_sha256']
            assert digest(out/'source_before_header_fix.py')==original['source_sha256']
            assert token_calculation_identity(Path(__file__).read_text())==token_calculation_identity((out/'source_before_header_fix.py').read_text()),'Frozen token computation changed'
        tokens=np.load(out/'tokens.npy',mmap_mode='r');assert tokens.shape==tuple(protocol['shape']) and tokens.dtype==np.float32
        return
    freeze(out/'protocol.json',protocol)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;timm.layers.set_fused_attn(False)
    backbone,prefix,lookup,parent=prepare_tokens(p,values,bags,rows)
    assert shutil.disk_usage(out).free>len(selected)*257*384*4+(1<<30),'Preserve existing caches'
    target=out/'tokens.npy';tokens=np.lib.format.open_memmap(target,mode='r+' if target.exists() else 'w+',dtype=np.float32,shape=tuple(protocol['shape']))
    progress=read(out/'progress.json') if (out/'progress.json').exists() else dict(completed=0,maximum_cls_error=0.)
    if (out/'progress.json').exists():assert progress['protocol_sha256']==digest(out/'protocol.json')
    with torch.inference_mode():
        for start in range(progress['completed'],len(selected),32):
            part=selected[start:start+32];final=backbone.norm(backbone.blocks[-2:](prefix[[lookup[key] for key in part]]))
            cls=(F.normalize(final[:,0],dim=1)*math.sqrt(384)).cpu().numpy();error=float(np.abs(cls-values[part,:384]).max())
            progress['maximum_cls_error']=max(progress['maximum_cls_error'],error);assert error<.002,('Local patch/cache input mismatch',error)
            tokens[start:start+len(part)]=final.cpu().numpy();tokens.flush();progress.update(completed=start+len(part),protocol_sha256=digest(out/'protocol.json'))
            temporary=out/'progress.tmp';temporary.write_text(json.dumps(progress));temporary.replace(out/'progress.json')
    result=dict(protocol_sha256=digest(out/'protocol.json'),tokens_sha256=digest(target),inputs=len(selected),encoder_prefix_inputs_recomputed=0,new_final_token_tail_inputs=len(selected),
                maximum_cls_error=progress['maximum_cls_error'],prefix_parent=parent,elapsed_seconds=time.monotonic()-started,ad_independent=False,olid_independent=False,acceptance_data_used=False)
    (out/'complete.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
