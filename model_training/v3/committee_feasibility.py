"""Optimistic consumed-label separation diagnostic, never a policy selector."""
import time
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint


def optimistic_committee(positive,negative,boundary,seconds=30.,epsilon=1e-5):
    positive=np.asarray(positive,dtype=np.float64);negative=np.asarray(negative,dtype=np.float64)
    if positive.ndim!=2 or negative.ndim!=2 or positive.shape[1]!=negative.shape[1] or not min(*positive.shape,*negative.shape) or not np.isfinite(positive).all() or not np.isfinite(negative).all():raise ValueError('Finite, nonempty positive and negative expert matrices required')
    if not np.isfinite(boundary) or epsilon<=0 or seconds<=0:raise ValueError('Invalid committee diagnostic bounds')
    started=time.monotonic();h,m=positive.shape;n=len(negative);floor=min(float(positive.min()),float(negative.min()),float(boundary))-epsilon
    big_m=boundary+epsilon-floor
    objective=np.r_[np.zeros(m),-np.ones(h)]
    # Nonnegative weights sum to one. All negatives must remain below the
    # unchanged boundary. Positive binary variables count admitted Healthy.
    a=np.zeros((1+n+h,m+h));a[0,:m]=1.;a[1:1+n,:m]=negative;a[1+n:,:m]=positive;a[1+n:,m:]=-big_m*np.eye(h)
    lower=np.r_[1.,np.full(n,-np.inf),np.full(h,floor)];upper=np.r_[1.,np.full(n,boundary-epsilon),np.full(h,np.inf)]
    result=milp(objective,integrality=np.r_[np.zeros(m),np.ones(h)],bounds=Bounds(np.zeros(m+h),np.ones(m+h)),constraints=LinearConstraint(a,lower,upper),options=dict(time_limit=seconds,mip_rel_gap=0.,presolve=True))
    report=dict(status=int(result.status),message=str(result.message),optimal=result.status==0,elapsed_seconds=time.monotonic()-started,diagnostic_only=True)
    if result.x is None:return dict(report,feasible=False,admitted_healthy=None,optimistic_healthy_upper_bound=0 if result.status==2 else None)
    weights=result.x[:m];assert weights.min()>=-1e-7 and abs(weights.sum()-1)<1e-6
    assert float((negative@weights).max())<=boundary-epsilon/2,'Solver violated fixed-boundary negative safety'
    admitted=int((positive@weights>=boundary+epsilon/2).sum());count=int(round(-result.fun));assert admitted>=count
    bound=getattr(result,'mip_dual_bound',None)
    upper_bound=int(np.floor(-bound+1e-5)) if bound is not None and np.isfinite(bound) else None
    return dict(report,feasible=True,weights=weights.tolist(),admitted_healthy=admitted,objective_healthy=count,optimistic_healthy_upper_bound=upper_bound,
        maximum_negative=float((negative@weights).max()),boundary=float(boundary),numerical_safety_margin=epsilon,mip_gap=float(result.mip_gap))
