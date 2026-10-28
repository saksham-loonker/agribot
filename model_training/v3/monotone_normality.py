"""Calibrate normal-patch distances without rewarding greater anomaly scores."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def fit_normality(x,y,weights,regularization=.01):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64);weights=np.asarray(weights,dtype=np.float64)
    if x.ndim!=2 or y.shape!=(len(x),) or weights.shape!=y.shape or set(y)!={0.,1.} or not np.isfinite(x).all() or not (weights>0).all():raise ValueError('Invalid normality calibration cohort')
    weights=weights/weights.sum();mean=(x*weights[:,None]).sum(0);scale=np.sqrt(((x-mean)**2*weights[:,None]).sum(0));scale=np.where(scale>1e-8,scale,1.);z=(x-mean)/scale
    def objective(theta):
        margin=z@theta[:-1]+theta[-1];risk=(weights*(np.logaddexp(0,margin)-y*margin)).sum()+regularization*(theta[:-1]@theta[:-1])/2
        residual=weights*(expit(margin)-y);gradient=np.r_[z.T@residual+regularization*theta[:-1],residual.sum()]
        return risk,gradient
    prior=float(weights@y);initial=np.r_[np.zeros(x.shape[1]),np.log(prior/(1-prior))]
    result=minimize(objective,initial,jac=True,method='L-BFGS-B',bounds=[(None,0.)]*x.shape[1]+[(None,None)],options=dict(maxiter=1000,ftol=1e-12))
    if not result.success:raise RuntimeError('Normality calibration did not converge: '+result.message)
    assert (result.x[:-1]<=0).all()
    return dict(mean=mean,scale=scale,coefficient=result.x[:-1],intercept=result.x[-1])


def predict_normality(model,x):return expit((np.asarray(x)-model['mean'])/model['scale']@model['coefficient']+model['intercept'])
