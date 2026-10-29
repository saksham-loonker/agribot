from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
from target_replay_sampling import target_replay_weights


class TargetReplayTests(unittest.TestCase):
    def test_replay_mass_and_classes_stay_balanced_despite_unequal_volumes(self):
        rows=[dict(source='target',group=str(i),y=int(i<90)) for i in range(100)]
        rows += [dict(source='old',group=str(i),y=i%2) for i in range(12)]
        weights=target_replay_weights(rows,np.arange(len(rows)),'target')
        self.assertAlmostEqual(float(weights[:100].sum()),.5)
        self.assertAlmostEqual(float(weights[[r['y']==1 for r in rows]].sum()),.5)
        self.assertAlmostEqual(float(weights[:90].sum()),.25)

    def test_indices_never_include_held_out_rows(self):
        rows=[dict(source=s,group=str(i),y=y) for i,(s,y) in enumerate([('target',0),('target',1),('old',0),('old',1),('heldout',1)])]
        weights=target_replay_weights(rows,[0,1,2,3],'target')
        self.assertEqual(len(weights),4)
        np.testing.assert_allclose(weights,np.full(4,.25))

    def test_missing_class_fails_instead_of_silently_changing_objective(self):
        rows=[dict(source='target',group='a',y=1),dict(source='old',group='b',y=0)]
        with self.assertRaisesRegex(ValueError,'both classes'):target_replay_weights(rows,[0,1],'target')


if __name__=='__main__':unittest.main()
