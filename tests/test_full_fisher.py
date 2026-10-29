import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
try:
    import numpy as np
    from full_fisher import fit_fisher,fisher_margin
except ImportError:np=None

@unittest.skipIf(np is None,'NumPy unavailable')
class FisherTests(unittest.TestCase):
    def test_retains_low_variance_supervised_direction(self):
        x=np.array([[-20,-.02],[20,-.01],[-20,.01],[20,.02]])
        for mode in ('binary','clinical_mixture'):
            state=fit_fisher(x,['Other','Other','Healthy','Healthy'],np.ones(4),np.arange(4),mode)
            scores=fisher_margin([[0,-.015],[0,.015]],state)
            self.assertLess(scores[0],0);self.assertGreater(scores[1],0)
            self.assertEqual(state['coef'].shape[1],2)

    def test_duplicate_views_do_not_inflate_effective_family_count(self):
        x=np.array([[-2.,0],[-1.,1],[1.,-1],[2.,0]]);labels=np.array(['Other','Other','Healthy','Healthy']);w=np.ones(4);families=np.arange(4)
        a=fit_fisher(x,labels,w,families,'binary');b=fit_fisher(np.repeat(x,2,axis=0),np.repeat(labels,2),np.repeat(w/2,2),np.repeat(families,2),'binary')
        self.assertAlmostEqual(float(a['effective_families']),4.)
        self.assertAlmostEqual(float(a['effective_families']),float(b['effective_families']))
        np.testing.assert_allclose(fisher_margin(x,a),fisher_margin(x,b),atol=1e-12)

    def test_invalid_binary_population_rejected(self):
        with self.assertRaises(ValueError):fit_fisher(np.ones((3,2)),['Healthy']*3,np.ones(3),np.arange(3),'binary')
