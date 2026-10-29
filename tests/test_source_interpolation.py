import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
try:
    import numpy as np
    from source_interpolation import partner_indices
except ImportError:np=None

@unittest.skipIf(np is None,'NumPy unavailable')
class InterpolationTests(unittest.TestCase):
    def test_only_fit_same_clinical_label_other_source_and_family(self):
        rows=[dict(label=l,source=s,group=g) for l,s,g in [('Healthy','A','1'),('Healthy','B','2'),('Nitrogen','B','3'),('Healthy','C','4')]];p,valid=partner_indices(rows,[0,1,2],[0,1,2],np.random.default_rng(1));self.assertEqual(p.tolist(),[1,0,2]);self.assertEqual(valid.tolist(),[True,True,False]);self.assertNotIn(3,p)
    def test_same_family_is_not_a_cross_source_partner(self):
        rows=[dict(label='Healthy',source=s,group='same') for s in ('A','B')];p,valid=partner_indices(rows,[0,1],[0],np.random.default_rng(1));self.assertFalse(valid[0]);self.assertEqual(p[0],0)
