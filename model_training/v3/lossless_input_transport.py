"""Lossless byte-plane transport for normalized FP32 image inputs."""
import numpy as np

def pack_views(views):
    x=np.ascontiguousarray(views,dtype=np.float32)
    if x.ndim!=4 or x.shape[-1]!=3 or not np.isfinite(x).all():raise ValueError('Finite NHWC RGB input required')
    bits=x.view(np.uint32);delta=bits.copy();delta[:,:,1:,:]=bits[:,:,1:,:]^bits[:,:,:-1,:]
    return np.ascontiguousarray(delta.view(np.uint8).reshape(-1,4).T),np.array(x.shape,dtype=np.int64)

def unpack_views(planes,shape):
    shape=tuple(map(int,shape));planes=np.asarray(planes)
    if len(shape)!=4 or shape[-1]!=3 or min(shape)<=0 or planes.dtype!=np.uint8 or planes.shape!=(4,int(np.prod(shape))):raise ValueError('Aligned uint8 byte planes and NHWC RGB shape required')
    bits=np.ascontiguousarray(planes.T).view(np.uint32).reshape(shape);np.bitwise_xor.accumulate(bits,axis=2,out=bits);x=bits.view(np.float32)
    if not np.isfinite(x).all():raise ValueError('Non-finite decoded input')
    return x
