from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
from routed_validation_cohort import healthy_cohort


class RoutedCohortTests(unittest.TestCase):
    def test_inactive_cases_keep_denominator_and_fixed_correct_verdicts(self):
        rows=[dict(path=str(i),source='heldout',validation=True,y=1) for i in range(40)]
        retained={str(i):dict(app_replay=dict(kind='HEALTHY' if i>=35 else 'UNSURE')) for i in range(40)}
        self.assertEqual(healthy_cohort(rows,{str(i) for i in range(31)},retained,'heldout'),(40,5))

    def test_missing_verdict_cannot_silently_shrink_cohort(self):
        with self.assertRaisesRegex(ValueError,'Missing'):
            healthy_cohort([dict(path='missing',source='s',validation=True,y=1)],set(),{},'s')

    def test_duplicate_photo_cannot_inflate_cohort(self):
        row=dict(path='same',source='s',validation=True,y=1)
        with self.assertRaisesRegex(ValueError,'duplicate'):
            healthy_cohort([row,row],set(),{'same':dict(app_replay=dict(kind='HEALTHY'))},'s')


if __name__=='__main__':unittest.main()
