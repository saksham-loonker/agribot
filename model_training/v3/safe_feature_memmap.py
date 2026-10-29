"""Create NumPy feature files whose headers remain readable after a restart."""
from pathlib import Path
import numpy as np


def feature_memmap(path,shape,dtype=np.float32):
    path=Path(path);shape=tuple(int(value) for value in shape);dtype=np.dtype(dtype)
    if dtype.hasobject:raise ValueError('Feature caches cannot contain Python objects')
    value=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=dtype,shape=shape)
    if value.shape!=shape or value.dtype!=dtype:raise ValueError('Existing feature cache shape/dtype changed')
    return value
