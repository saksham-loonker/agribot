"""Optimistic two-threshold routing bound; never a candidate-selection method."""
import numpy as np


def threshold_ceiling(positive,negative,eligible,active,baseline,truth,cohorts):
    positive,negative=np.asarray(positive),np.asarray(negative)
    eligible,active,baseline,truth=[np.asarray(value,dtype=bool) for value in (eligible,active,baseline,truth)]
    if not all(value.shape==truth.shape for value in (positive,negative,eligible,active,baseline)):
        raise ValueError("Inputs must have identical aligned shapes")
    if not np.isfinite(positive).all() or not np.isfinite(negative).all():
        raise ValueError("Nonfinite cached score")
    budgets=[int((baseline&~truth&cohort).sum()) for cohort in cohorts]
    best=dict(healthy=int((baseline&truth).sum()),false_healthy=int((baseline&~truth).sum()),
              positive_cut=1.,negative_cut=0.,expert_enabled=False)
    for cut in sorted(set(positive[eligible&active])):
        proposed=baseline|(eligible&active&(positive>=cut))
        low=0.
        impossible=False
        for cohort,budget in zip(cohorts,budgets):
            mask=proposed&~truth&cohort
            fixed=int((mask&~active).sum())
            removable=np.sort(negative[mask&active])
            remain=budget-fixed
            if remain<0:
                impossible=True;break
            if len(removable)>remain:
                low=max(low,float(removable[-remain-1]),float(np.nextafter(0.,1.)))
        if impossible:continue
        predicted=proposed&(~active|(negative>low)) if low>0 else proposed
        assert all(int((predicted&~truth&cohort).sum())<=budget for cohort,budget in zip(cohorts,budgets))
        healthy=int((predicted&truth).sum())
        if healthy>best["healthy"]:
            best=dict(healthy=healthy,false_healthy=int((predicted&~truth).sum()),
                      positive_cut=float(cut),negative_cut=low,expert_enabled=True)
    return best
