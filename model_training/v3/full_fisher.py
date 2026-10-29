"""Weighted full-feature discriminants with family-effective OAS-style shrinkage."""
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.special import logsumexp

def fit_fisher(x,labels,weights,families,mode):
    x=np.asarray(x,dtype=np.float64);labels=np.asarray(labels,dtype=str);w=np.asarray(weights,dtype=np.float64);families=np.asarray(families)
    if x.ndim!=2 or labels.shape!=(len(x),) or w.shape!=(len(x),) or families.shape!=(len(x),) or mode not in ('binary','clinical_mixture') or not np.isfinite(x).all() or not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:
        raise ValueError('Aligned finite training data, family IDs and nonnegative mass required')
    keep=w>0;x=x[keep];labels=labels[keep];families=families[keep];w=w[keep];w=w/w.sum()
    if mode=='binary':labels=np.where(labels=='Healthy','Healthy','Diseased')
    names=np.array(sorted(set(labels)));healthy=names=='Healthy'
    if not healthy.any() or healthy.all():raise ValueError('Both Healthy and negative labels required')
    means=[];mass=[];residual=x.copy()
    for name in names:
        mask=labels==name;total=w[mask].sum();mean=(w[mask]/total)@x[mask];means.append(mean);mass.append(total);residual[mask]-=mean
    means=np.stack(means);mass=np.array(mass);cov=(residual*w[:,None]).T@residual;cov=.5*(cov+cov.T);d=x.shape[1];mu=max(float(np.trace(cov))/d,1e-12)
    _,group=np.unique(families,return_inverse=True);family_mass=np.bincount(group,weights=w);effective_n=float(1/np.square(family_mass).sum())
    alpha=float(np.square(cov).mean());numerator=alpha+mu*mu;denominator=(effective_n+1)*(alpha-mu*mu/d);shrinkage=1. if denominator<=0 else min(1.,numerator/denominator)
    regularized=(1-shrinkage)*cov;regularized.flat[::d+1]+=shrinkage*mu+1e-6*mu
    coef=cho_solve(cho_factor(regularized,lower=True,check_finite=False),means.T,check_finite=False).T
    prior=mass.copy();prior[healthy]=.5*mass[healthy]/mass[healthy].sum();prior[~healthy]=.5*mass[~healthy]/mass[~healthy].sum();intercept=-.5*np.einsum('ij,ij->i',means,coef)+np.log(prior)
    return dict(coef=coef,intercept=intercept,component_names=names,healthy_components=healthy,shrinkage=np.array(shrinkage),effective_families=np.array(effective_n))

def fisher_margin(x,state):
    score=np.asarray(x,dtype=np.float64)@state['coef'].T+state['intercept'];healthy=state['healthy_components']
    return logsumexp(score[:,healthy],axis=1)-logsumexp(score[:,~healthy],axis=1)
