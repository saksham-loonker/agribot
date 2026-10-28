"""Reserve half the training mass for target errors and half for protective replay."""
import numpy as np
from source_balance import source_class_group_weights


def target_replay_weights(rows,indices,target_source):
    indices=np.asarray(indices,dtype=np.int64)
    mask=np.array([rows[i]['source']==target_source for i in indices])
    if not mask.any() or mask.all():raise ValueError('Target replay requires both populations')
    weights=np.zeros(len(indices),np.float64)
    for selected in (mask,~mask):
        if {rows[i]['y'] for i in indices[selected]}!={0,1}:raise ValueError('Each replay population needs both classes')
        weights[selected]=.5*source_class_group_weights(rows,indices[selected])
    assert abs(weights.sum()-1.)<1e-12
    return weights
