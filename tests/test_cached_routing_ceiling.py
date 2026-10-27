"""The cheap stopping bound must agree with an exhaustive two-threshold search."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"model_training/v3"))
from cached_routing_ceiling import threshold_ceiling


class RoutingCeilingTests(unittest.TestCase):
    def test_matches_brute_force_with_ties_zeros_and_protected_cases(self):
        rng=np.random.default_rng(96)
        for _ in range(100):
            p,n=rng.choice([0.,.2,.5,.9,1.],size=(2,8))
            eligible,active,base,truth=rng.integers(0,2,size=(4,8)).astype(bool)
            cohorts=[np.ones(8,bool),np.arange(8)%2==0]
            result=threshold_ceiling(p,n,eligible,active,base,truth,cohorts)
            best=int((base&truth).sum())
            for high in [*p,2.]:
                for low in [0.,np.nextafter(0.,1.),*n,2.]:
                    predicted=base|(eligible&active&(p>=high))
                    if low>0:predicted&=(~active|(n>low))
                    if all((predicted&~truth&c).sum()<=(base&~truth&c).sum() for c in cohorts):
                        best=max(best,int((predicted&truth).sum()))
            self.assertEqual(result["healthy"],best)

    def test_misaligned_or_nonfinite_inputs_fail(self):
        with self.assertRaises(ValueError):
            threshold_ceiling([.5],[.5,.2],[True],[True],[True],[True],[])
        with self.assertRaises(ValueError):
            threshold_ceiling([float("nan")],[.5],[True],[True],[True],[True],[])


if __name__=="__main__":unittest.main()
