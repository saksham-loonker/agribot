"""Explicit stable linear/quadratic basis for cross-fitted density evidence."""
import numpy as np

def stack_basis(x,center,scale,degree):
    x=np.asarray(x,dtype=np.float64);center=np.asarray(center,dtype=np.float64);scale=np.asarray(scale,dtype=np.float64)
    if x.ndim!=2 or center.shape!=(x.shape[1],) or scale.shape!=center.shape or degree not in (1,2) or not np.isfinite(x).all() or not np.isfinite(center).all() or not np.isfinite(scale).all() or (scale<=0).any():
        raise ValueError('Finite aligned evidence, positive scales and degree 1/2 required')
    z=(x-center)/scale
    if degree==1:return z
    return np.column_stack([z]+[z[:,i]*z[:,j] for i in range(z.shape[1]) for j in range(i,z.shape[1])])
