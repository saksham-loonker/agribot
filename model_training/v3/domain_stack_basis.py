"""Conditional two-view evidence with optional learned source posterior."""
import numpy as np

def domain_stack_basis(x,center,scale,domain=None):
    x=np.asarray(x,dtype=np.float64);center=np.asarray(center);scale=np.asarray(scale)
    if x.ndim!=3 or x.shape[1]!=2 or center.shape!=(x.shape[2],) or scale.shape!=center.shape or not np.isfinite(x).all() or not np.isfinite(center).all() or not np.isfinite(scale).all() or (scale<=0).any():raise ValueError('Finite photo x2 views xevidence with positive aligned scales required')
    z=(x-center)/scale;other=z[:,::-1];parts=[z,other,abs(z-other),z*other]
    if domain is not None:
        d=np.asarray(domain,dtype=np.float64)
        if d.shape!=(len(x),) or not np.isfinite(d).all() or (d<0).any() or (d>1).any():raise ValueError('Aligned source probabilities required')
        d=np.broadcast_to((2*d-1)[:,None,None],(len(x),2,1));parts.extend([d,z*d,other*d])
    return np.concatenate(parts,axis=2)
