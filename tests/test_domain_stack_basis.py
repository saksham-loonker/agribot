import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
try:
    import numpy as np
    from domain_stack_basis import domain_stack_basis
except ImportError:np=None

@unittest.skipIf(np is None,'NumPy unavailable')
class DomainBasisTests(unittest.TestCase):
    def test_view_swap_keeps_minimum_and_domain_changes_only_conditional_terms(self):
        x=np.array([[[1.,2.],[3.,4.]]]);c=np.zeros(2);s=np.ones(2);a=domain_stack_basis(x,c,s,[0.]);b=domain_stack_basis(x[:,::-1],c,s,[0.]);np.testing.assert_array_equal(a[:,::-1],b);q=domain_stack_basis(x,c,s,[1.]);np.testing.assert_array_equal(a[:,:,:8],q[:,:,:8]);np.testing.assert_array_equal(a[:,:,8:],-q[:,:,8:])
    def test_invalid_domain_or_scale_is_rejected(self):
        for d in ([1.5],[float('nan')]):
            with self.assertRaises(ValueError):domain_stack_basis(np.ones((1,2,2)),np.zeros(2),np.ones(2),d)
        with self.assertRaises(ValueError):domain_stack_basis(np.ones((1,2,2)),np.zeros(2),[1,0])
