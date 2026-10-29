"""Clinical text priors for the already cached SigLIP2 image embeddings.

Fetch only text layers, used vocabulary rows, and a vision-weight proof.
The unused 256k-token embedding matrix is never downloaded. No image inference.
"""
from pathlib import Path
import argparse,copy,hashlib,json,time
import numpy as np
import torch
from clinical_semantic_prompts import PROMPTS
from hf_range_tensors import RangeTensors,digest
from opened_context_data import read
from population_siglip_tail import freeze

REPOSITORY='google/siglip2-base-patch16-256'
REVISION='3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab'


def vision_proof(cache,state):
    """Compare every original image-tower tensor, including packed attention."""
    compared=set();official=set();maximum=0.
    def equal(target,names,transform=None):
        nonlocal maximum
        arrays=[torch.from_numpy(cache.array(name)) for name in names];value=arrays[0] if len(arrays)==1 else torch.cat(arrays,0)
        if transform:value=transform(value)
        expected=state['network.'+target];assert expected.shape==value.shape,(target,expected.shape,value.shape)
        error=float((expected-value).abs().max());maximum=max(maximum,error);assert error==0.,('Text/image checkpoint coordinate mismatch',target,error)
        compared.add('network.'+target);official.update(names)
    equal('patch_embed.proj.weight',['vision_model.embeddings.patch_embedding.weight']);equal('patch_embed.proj.bias',['vision_model.embeddings.patch_embedding.bias'])
    equal('pos_embed',['vision_model.embeddings.position_embedding.weight'],lambda x:x[None])
    for i in range(12):
        a=f'vision_model.encoder.layers.{i}.';b=f'blocks.{i}.'
        for parameter in ('weight','bias'):
            equal(b+'norm1.'+parameter,[a+'layer_norm1.'+parameter]);equal(b+'norm2.'+parameter,[a+'layer_norm2.'+parameter])
            equal(b+'attn.qkv.'+parameter,[a+'self_attn.'+name+'_proj.'+parameter for name in ('q','k','v')])
            equal(b+'attn.proj.'+parameter,[a+'self_attn.out_proj.'+parameter])
            for layer in ('fc1','fc2'):equal(b+'mlp.'+layer+'.'+parameter,[a+'mlp.'+layer+'.'+parameter])
    equal('attn_pool.latent',['vision_model.head.probe'])
    for parameter in ('weight','bias'):
        equal('norm.'+parameter,['vision_model.post_layernorm.'+parameter])
        equal('attn_pool.q.'+parameter,['vision_model.head.attention.in_proj_'+parameter],lambda x:x[:768])
        equal('attn_pool.kv.'+parameter,['vision_model.head.attention.in_proj_'+parameter],lambda x:x[768:])
        equal('attn_pool.proj.'+parameter,['vision_model.head.attention.out_proj.'+parameter])
        equal('attn_pool.norm.'+parameter,['vision_model.head.layernorm.'+parameter])
        for layer in ('fc1','fc2'):equal('attn_pool.mlp.'+layer+'.'+parameter,['vision_model.head.mlp.'+layer+'.'+parameter])
    assert compared=={name for name in state if name.startswith('network.')},'Unverified cached-image model tensors remain'
    assert official=={name for name in cache.header if name.startswith('vision_model.')},'Unverified official vision tensors remain'
    return dict(timm_tensors=len(compared),official_vision_tensors=len(official),maximum_weight_error=maximum,all_vision_weights_exact=True)


def pruning_check(TextModel,TextConfig):
    torch.manual_seed(2149);config=TextConfig(vocab_size=31,hidden_size=8,intermediate_size=16,num_hidden_layers=1,num_attention_heads=2,max_position_embeddings=4)
    config._attn_implementation='eager';original=TextModel(config).eval();ids=torch.tensor([[1,7,3,0],[6,1,0,0]])
    used=sorted(set(ids.ravel().tolist()));cfg=copy.deepcopy(config);cfg.vocab_size=len(used);pruned=TextModel(cfg).eval();state=original.state_dict();key=next(k for k in state if k.endswith('token_embedding.weight'));state[key]=state[key][used];pruned.load_state_dict(state,strict=True)
    mapping={token:i for i,token in enumerate(used)};remapped=torch.tensor([[mapping[token] for token in row] for row in ids.tolist()])
    with torch.inference_mode():error=float((original(input_ids=ids).pooler_output-pruned(input_ids=remapped).pooler_output).abs().max())
    assert error<1e-6,('Fixed-prompt vocabulary pruning changes text encoding',error)
    return error


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    p=Path(parser.parse_args().campaign);out=p/'phase149_semantic_text_features';out.mkdir(exist_ok=True);started=time.monotonic()
    image=p/'phase23_siglip/siglip2_feature_heads/encoder.pt'
    prompt_sha=hashlib.sha256(json.dumps(PROMPTS).encode()).hexdigest()
    protocol=dict(source_sha256=digest(__file__),range_source_sha256=digest(Path(__file__).with_name('hf_range_tensors.py')),prompt_source_sha256=digest(Path(__file__).with_name('clinical_semantic_prompts.py')),
        repository=REPOSITORY,revision=REVISION,prompts=[list(row) for row in PROMPTS],prompt_sha256=prompt_sha,image_encoder_checkpoint_sha256=digest(image),
        role='Development-only semantic prior; text strings are hypotheses, not new clinical labels or acceptance',new_image_encoder_inputs=0,ad_independent=False,olid_independent=False,acceptance_data_used=False,
        vocabulary='Only prompt-used original embedding rows; exact ID remapping, unchanged transformer/position/head weights and last-position pooling',
        image_alignment='Every official vision tensor must match the existing image tower after explicit QKV/attention-pool packing conversion')
    freeze(out/'protocol.json',protocol)
    if (out/'complete.json').exists():
        result=read(out/'complete.json');assert result['protocol_sha256']==digest(out/'protocol.json') and result['text_features_sha256']==digest(out/'text_features.npy');return
    from transformers import AutoTokenizer,SiglipConfig,SiglipTextModel,SiglipTextConfig
    import transformers
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cache=RangeTensors(out/'ranges',REPOSITORY,REVISION)
    vision=[name for name in cache.header if name.startswith('vision_model.')];cache.prefetch(vision)
    state=torch.load(image,weights_only=True,map_location='cpu');proof=vision_proof(cache,state['model']);del state
    print(json.dumps(dict(stage='vision_weight_alignment',**proof)),flush=True)
    tokenizer=AutoTokenizer.from_pretrained(REPOSITORY,revision=REVISION,trust_remote_code=False)
    inputs=tokenizer([text for _,text in PROMPTS],padding='max_length',max_length=64,truncation=True,return_tensors='pt')
    assert not any(key not in ('input_ids','attention_mask') for key in inputs)
    ids=inputs['input_ids'];used=sorted(set(ids.ravel().tolist()));embedding='text_model.embeddings.token_embedding.weight';text_names=[name for name in cache.header if name.startswith('text_model.') and name!=embedding]
    cache.prefetch(text_names);embedding_rows=cache.rows(embedding,used)
    # Config/tokenizer downloads are pinned to the same immutable model commit.
    config=SiglipConfig.from_pretrained(REPOSITORY,revision=REVISION).text_config;config.vocab_size=len(used);config._attn_implementation='eager'
    model=SiglipTextModel(config).eval().requires_grad_(False);expected=set(model.state_dict());weights={}
    for name in text_names:
        key=name if name in expected else name.removeprefix('text_model.');assert key in expected,name
        weights[key]=torch.from_numpy(cache.array(name))
    key=embedding if embedding in expected else embedding.removeprefix('text_model.');weights[key]=torch.from_numpy(embedding_rows)
    assert set(weights)==expected;model.load_state_dict(weights,strict=True);del weights
    error=pruning_check(SiglipTextModel,SiglipTextConfig);mapping={token:i for i,token in enumerate(used)}
    inputs['input_ids']=torch.tensor([[mapping[token] for token in row] for row in ids.tolist()])
    assert not any(key not in ('input_ids','attention_mask') for key in inputs)
    with torch.inference_mode():features=torch.nn.functional.normalize(model(**inputs).pooler_output,dim=1).cpu().numpy()
    assert features.shape==(len(PROMPTS),768) and np.isfinite(features).all();np.save(out/'text_features.npy',features)
    (out/'used_token_ids.json').write_text(json.dumps(used));(out/'ranges_manifest.json').write_text(json.dumps(cache.manifest(),indent=2))
    result=dict(protocol_sha256=digest(out/'protocol.json'),text_features_sha256=digest(out/'text_features.npy'),range_manifest_sha256=digest(out/'ranges_manifest.json'),used_token_ids_sha256=digest(out/'used_token_ids.json'),
        vision_weight_proof=proof,transformers_version=transformers.__version__,torch_version=torch.__version__,vocabulary_rows_used=len(used),original_vocabulary_rows=cache.header[embedding]['shape'][0],
        pruning_runtime_maximum_error=error,new_image_encoder_inputs=0,range_bytes=sum(row['bytes'] for row in cache.manifest()),elapsed_seconds=time.monotonic()-started,
        full_parent_safetensors_downloaded=False,full_parent_file_hash_verified=False,transport='HTTPS ranges at immutable commit; local range hashes and every vision tensor verified',model_promoted=False,ad_independent=False,olid_independent=False,acceptance_data_used=False)
    (out/'complete.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
