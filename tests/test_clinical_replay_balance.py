from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
from clinical_replay_balance import clinical_replay_weights


class ClinicalReplayBalanceTests(unittest.TestCase):
    def test_rare_named_condition_is_not_swamped_by_other(self):
        rows=[dict(y=0,source='field',label='Other',group=str(i)) for i in range(100)]
        rows += [dict(y=0,source='field',label='Nitrogen Deficiency',group='n'),dict(y=1,source='field',label='Healthy',group='h')]
        weights=clinical_replay_weights(rows,range(len(rows)))
        self.assertAlmostEqual(weights[:100].sum(),.25);self.assertAlmostEqual(weights[100],.25);self.assertAlmostEqual(weights[101],.5)

    def test_source_mass_and_duplicate_family_mass_are_preserved(self):
        rows=[dict(y=y,source=s,label=('Healthy' if y else 'Other'),group=g) for y,s,g in [(1,'a','h'),(1,'b','j'),(0,'a','d'),(0,'a','d'),(0,'b','e')]]
        w=clinical_replay_weights(rows,range(5))
        self.assertAlmostEqual(w[0],.25);self.assertAlmostEqual(w[1],.25);self.assertAlmostEqual(w[2]+w[3],.25);self.assertAlmostEqual(w[4],.25)

    def test_inconsistent_supervision_is_rejected(self):
        with self.assertRaises(ValueError):clinical_replay_weights([dict(y=1,source='x',label='Other',group='g')],[0])
        with self.assertRaises(ValueError):clinical_replay_weights([],[])
