"""Training-only bandwidth and explicit-array binary RBF inference."""
import numpy as np
from sklearn.metrics import pairwise_distances

def training_gamma(photo_views):
    x=np.asarray(photo_views).mean(1);d=pairwise_distances(x,metric='sqeuclidean');median=float(np.median(d[np.triu_indices(len(x),1)]))
    if not np.isfinite(median) or median<=0:raise ValueError('Positive finite training distance required')
    return 1./median

def restored_margin(x,state):
    assert np.array_equal(state['classes'],[0,1])
    d=pairwise_distances(np.asarray(x),state['support_vectors'],metric='sqeuclidean')
    return np.exp(-float(state['gamma'])*d)@state['dual_coef'].reshape(-1)+float(state['intercept'].item())
