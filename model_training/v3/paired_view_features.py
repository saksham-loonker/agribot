"""Fixed two-view conditional evidence, without fitting image statistics."""
import numpy as np

def paired_view_features(views):
    v=np.asarray(views,dtype=np.float64)
    if v.ndim!=3 or v.shape[1]!=2 or v.shape[2]==0 or not np.isfinite(v).all():raise ValueError('Finite photo x2 views xfeature required')
    other=v[:,::-1];delta=abs(v-other);product=v*other
    def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)
    return np.concatenate([unit(v),unit(other),unit(delta),unit(product)],axis=-1)/2
