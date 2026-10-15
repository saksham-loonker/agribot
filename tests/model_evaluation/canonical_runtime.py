"""Sparse execution of the exact canonical resize weights used by Android.

The original NumPy reference multiplies dense mostly-zero matrices. This keeps
the same weights, order, channels, and float32 arithmetic, avoiding billions of
zero multiplications on high-resolution source photos.
"""
from functools import lru_cache
import numpy as np
from scipy.sparse import csr_matrix


def install(canonical):
    original_axis = canonical.axis_weights

    @lru_cache(maxsize=32)
    def weights(n_in, n_out):
        return csr_matrix(original_axis(n_in,n_out))

    def resize(image, out_w, out_h):
        h,w=image.shape[:2]
        x=image.astype(np.float32)
        vertical=weights(h,out_h).dot(x.reshape(h,w*3)).reshape(out_h,w,3)
        horizontal=weights(w,out_w).dot(vertical.transpose(1,0,2).reshape(w,out_h*3))
        return np.ascontiguousarray(horizontal.reshape(out_w,out_h,3).transpose(1,0,2))

    canonical.resize=resize
