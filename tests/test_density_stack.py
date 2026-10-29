import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
try:
    import numpy as np
    from density_stack import stack_basis
except ImportError:np=None

@unittest.skipIf(np is None,'NumPy unavailable')
class StackTests(unittest.TestCase):
    def test_quadratic_basis_and_batch_order(self):
        x=np.array([[2.,3.],[-2.,1.]]);center=np.zeros(2);scale=np.ones(2)
        actual=stack_basis(x,center,scale,2)
        np.testing.assert_array_equal(actual,[[2,3,4,6,9],[-2,1,4,-2,1]])
        np.testing.assert_array_equal(actual,np.concatenate([stack_basis(row[None],center,scale,2) for row in x]))
        np.testing.assert_array_equal(x,[[2,3],[-2,1]])

    def test_invalid_normalization_rejected(self):
        with self.assertRaises(ValueError):stack_basis([[1.,2.]],[0,0],[1,0],1)
        with self.assertRaises(ValueError):stack_basis([[1.,float('nan')]],[0,0],[1,1],2)
