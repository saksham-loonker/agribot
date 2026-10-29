"""Conservative outer relaxation for a consumed-data committee diagnostic.

Allow negative scores above, and positive scores below, the real boundary by a
small margin. Thus this counts a superset of actual safe committees. It must
never supply deployed weights or be called independent validation.
"""
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint


def relaxed_committee_upper_bound(positive,negative,boundary,relaxation=1e-6,seconds=30.):
    positive=np.asarray(positive,dtype=np.float64);negative=np.asarray(negative,dtype=np.float64)
    if positive.ndim!=2 or negative.ndim!=2 or positive.shape[1]!=negative.shape[1] or not min(*positive.shape,*negative.shape) or not np.isfinite(positive).all() or not np.isfinite(negative).all() or not np.isfinite(boundary) or relaxation<=0:raise ValueError('Invalid finite committee matrices or relaxation')
    h,m=positive.shape;n=len(negative);floor=min(float(positive.min()),float(negative.min()),boundary)-relaxation;big_m=boundary-relaxation-floor
    a=np.zeros((1+n+h,m+h));a[0,:m]=1.;a[1:1+n,:m]=negative;a[1+n:,:m]=positive;a[1+n:,m:]=-big_m*np.eye(h)
    result=milp(np.r_[np.zeros(m),-np.ones(h)],integrality=np.r_[np.zeros(m),np.ones(h)],bounds=Bounds(np.zeros(m+h),np.ones(m+h)),constraints=LinearConstraint(a,np.r_[1.,np.full(n,-np.inf),np.full(h,floor)],np.r_[1.,np.full(n,boundary+relaxation),np.full(h,np.inf)]),options=dict(time_limit=seconds,mip_rel_gap=0.,presolve=True))
    bound=getattr(result,'mip_dual_bound',None);upper=int(np.floor(-bound+1e-5)) if bound is not None and np.isfinite(bound) else (0 if result.status==2 else None)
    return dict(status=int(result.status),message=str(result.message),optimal=result.status==0,relaxed_healthy_upper_bound=upper,relaxed_objective=None if result.fun is None else int(round(-result.fun)),
        numerical_relaxation=relaxation,diagnostic_only=True,weights_not_a_candidate=True,weights=None if result.x is None else result.x[:m].tolist())
