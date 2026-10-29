"""Weighted train-only PCA and regularized clinical Gaussian mixtures."""
import numpy as np
from scipy.linalg import eigh
from scipy.special import logsumexp

def fit_projection(x,weights,dimensions=32):
    x=np.asarray(x,dtype=np.float64);w=np.asarray(weights,dtype=np.float64)
    if x.ndim!=2 or w.shape!=(len(x),) or len(x)<2 or not np.isfinite(x).all() or not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:
        raise ValueError('Finite aligned training vectors and nonnegative mass required')
    w=w/w.sum();center=w@x;z=x-center;cov=(z*w[:,None]).T@z
    k=min(int(dimensions),x.shape[1],int(np.count_nonzero(w))-1)
    if k<1:raise ValueError('Positive projection dimension required')
    values,axes=eigh(cov,subset_by_index=[x.shape[1]-k,x.shape[1]-1])
    floor=max(float(np.trace(cov))/x.shape[1]*1e-6,1e-12)
    axes=axes[:,::-1]/np.sqrt(np.maximum(values[::-1],floor))
    return dict(center=center,axes=axes)

def project(x,state):return (np.asarray(x,dtype=np.float64)-state['center'])@state['axes']

def fit_density(z,labels,weights,mode):
    z=np.asarray(z,dtype=np.float64);labels=np.asarray(labels,dtype=str);w=np.asarray(weights,dtype=np.float64)
    if mode not in ('shared','class_shrinkage') or z.ndim!=2 or labels.shape!=(len(z),) or w.shape!=(len(z),) or not np.isfinite(z).all() or not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:
        raise ValueError('Invalid density training contract')
    w=w/w.sum();names=np.array(sorted(set(labels[w>0])));healthy=names=='Healthy'
    if not healthy.any() or healthy.all():raise ValueError('Healthy and negative clinical components required')
    means=[];covs=[];mass=[]
    for name in names:
        mask=labels==name;local=w[mask];total=local.sum();local=local/total;mean=local@z[mask];delta=z[mask]-mean
        means.append(mean);covs.append((delta*local[:,None]).T@delta);mass.append(total)
    covs=np.stack(covs);mass=np.array(mass);pooled=np.einsum('k,kij->ij',mass,covs);d=z.shape[1]
    scale=max(float(np.trace(pooled))/d,1e-8)
    base=.5*pooled+.5*scale*np.eye(d)
    matrices=np.repeat(base[None],len(names),axis=0) if mode=='shared' else .5*covs+.5*base[None]
    matrices+=.001*scale*np.eye(d)[None]
    sign,logdet=np.linalg.slogdet(matrices)
    if not np.all(sign>0):raise ValueError('Covariance must be positive definite')
    prior=mass.copy();prior[healthy]=.5*mass[healthy]/mass[healthy].sum();prior[~healthy]=.5*mass[~healthy]/mass[~healthy].sum()
    return dict(component_names=names,means=np.stack(means),precision=np.linalg.inv(matrices),logdet=logdet,logprior=np.log(prior),healthy_components=healthy)

def density_margin(z,state):
    delta=np.asarray(z,dtype=np.float64)[:,None,:]-state['means'][None]
    score=-.5*(np.einsum('nki,kij,nkj->nk',delta,state['precision'],delta)+state['logdet'][None])+state['logprior'][None]
    healthy=state['healthy_components']
    return logsumexp(score[:,healthy],axis=1)-logsumexp(score[:,~healthy],axis=1)
