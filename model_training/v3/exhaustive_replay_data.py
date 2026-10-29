"""Reconstruct the frozen full-replay namespace without opening acceptance data."""
from pathlib import Path
import numpy as np
from opened_context_data import read
from phase114_new_supervision_features import original_features
from permissive_encoder import digest


def exhaustive_replay_data(campaign):
    p=Path(campaign);values,bags,descriptors,parents=original_features(p);rows=read(p/'phase84_population_head/fitting_rows.json');selection=read(p/'phase118_conditional_recovery/protocol.json');anchors=read(p/'phase133_anchor_input_cache/protocol.json');full=read(p/'phase127_clinical_anchors/protocol.json')
    extra=p/'phase137_full_source_features';complete=read(extra/'complete.json');assert complete['protocol_sha256']==digest(extra/'protocol.json')
    for name,field in [('new_features.npy','new_features_sha256'),('bags.json','bags_sha256'),('new_input_sha256.json','new_keys_sha256')]:assert digest(extra/name)==complete[field]
    identities=[r['input_sha256'] for r in descriptors]+read(extra/'new_input_sha256.json');values=np.concatenate([values,np.load(extra/'new_features.npy')]);assert len(values)==len(identities)
    field=p/'phase164_new_field_features';fc=read(field/'complete.json');assert fc['protocol_sha256']==digest(field/'protocol.json')
    for filename,key in [('features.npy','features_sha256'),('input_sha256.json','input_sha256_sha256'),('bags.json','bags_sha256')]:assert digest(field/filename)==fc[key]
    field_values=np.load(field/'features.npy');field_keys=read(field/'input_sha256.json');assert field_values.shape==(len(field_keys),3200)
    field_offset=len(identities);identities+=field_keys;values=np.concatenate([values,field_values[:,:1408]])
    target=anchors['target_indices'];replay=[i for i in full['train_indices'] if i not in set(target)];val=selection['validation_indices']
    tables={i:[[bags[rows[i]['path']][view] for view in ('center','full')]] for i in target+replay+val}
    for path,bag in read(extra/'bags.json').items():
        if not bag['actual_leaves']:continue
        i=len(rows);tables[i]=bag['views'];record=bag['source_record'];rows.append(dict(path=path,y=int(record['label']=='Healthy'),source='new107_development',group='new107:entire_publisher_collection',validation=False,label=record['label']))
    field_indices=[]
    for path,bag in read(field/'bags.json').items():
        if not bag['actual_leaves']:continue
        i=len(rows);field_indices.append(i);tables[i]=[[[field_offset+key for key in view] for view in pair] for pair in bag['views']];record=bag['source_record'];rows.append(dict(path=path,y=int(record['label']=='Healthy'),source='field93',group=record['group'],validation=False,label=record['label']))
    train=target+replay+field_indices;reference=read(p/'phase193_exhaustive_mobile_replay/protocol.json');assert train==reference['train_indices'] and val==reference['validation_indices'] and field_indices==reference['field_training_indices']
    keys=lambda ids:{identities[key] for i in ids for pair in tables[i] for view in pair for key in view}
    assert not keys(train)&keys(val) and not {rows[i]['group'] for i in train}&{rows[i]['group'] for i in val}
    return values,identities,rows,tables,train,val,dict(original_descriptors=descriptors,field_offset=field_offset,target=target,replay=replay,field_indices=field_indices,original_parents=parents,full_replay_protocol_sha256=digest(p/'phase193_exhaustive_mobile_replay/protocol.json'))
