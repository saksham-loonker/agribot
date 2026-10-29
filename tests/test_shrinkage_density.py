import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
try:
    import numpy as np
    from shrinkage_density import fit_projection,project,fit_density,density_margin
except ImportError:np=None

@unittest.skipIf(np is None,'NumPy unavailable')
class DensityTests(unittest.TestCase):
    def test_zero_mass_outlier_cannot_change_training_projection(self):
        x=np.array([[-2.,0.],[0.,1.],[2.,0.]])
        a=fit_projection(x,np.ones(3),1);b=fit_projection(np.vstack([x,[900.,800.]]),np.array([1.,1.,1.,0.]),1)
        np.testing.assert_allclose(a['center'],b['center']);np.testing.assert_allclose(abs(a['axes']),abs(b['axes']))

    def test_both_covariances_order_clusters_and_restore_exactly(self):
        x=np.array([[-3.,0.],[-2.,0.],[2.,0.],[3.,0.]])
        for mode in ('shared','class_shrinkage'):
            state=fit_density(x,['Other','Other','Healthy','Healthy'],np.ones(4),mode)
            query=np.array([[-2.5,0.],[2.5,0.]]);margin=density_margin(query,state)
            self.assertLess(margin[0],0);self.assertGreater(margin[1],0)
            clone={k:v.copy() for k,v in state.items()};np.testing.assert_array_equal(margin,density_margin(query,clone))
            density_margin(np.array([[100.,100.]]),state)
            for k in state:np.testing.assert_array_equal(state[k],clone[k])

    def test_invalid_weights_and_single_binary_class_rejected(self):
        with self.assertRaises(ValueError):fit_projection(np.ones((3,2)),[1,-1,1])
        with self.assertRaises(ValueError):fit_density(np.ones((3,2)),['Healthy']*3,np.ones(3),'shared')

    def test_zero_mass_rows_do_not_increase_available_rank(self):
        x=np.arange(15,dtype=float).reshape(3,5)
        a=fit_projection(x,np.ones(3),32);b=fit_projection(np.vstack([x,np.ones(5)*99]),[1,1,1,0],32)
        self.assertEqual(a['axes'].shape,(5,2));self.assertEqual(b['axes'].shape,a['axes'].shape)
