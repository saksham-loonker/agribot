from pathlib import Path
import copy,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
from clinical_retention_proposal import evaluate_clinical_retention


class ClinicalRetentionProposalTests(unittest.TestCase):
    def case(self,label='Healthy',kind='HEALTHY'):
        return dict(path='photo',label=label,app_replay=dict(kind=kind,label=label,confidence=.99))

    def test_unchanged_verdicts_pass_both_rules(self):
        row=self.case();r=evaluate_clinical_retention([row],[copy.deepcopy(row)]);self.assertTrue(r['original_disease_and_healthy_exact']);self.assertFalse(r['active_promotion_gate_changed'])

    def test_labeled_false_healthy_can_only_become_unlabeled_unsure(self):
        row=self.case('Nitrogen Deficiency');new=copy.deepcopy(row);new['app_replay']=dict(kind='UNSURE',label=None)
        r=evaluate_clinical_retention([row],[new]);self.assertFalse(r['original_disease_and_healthy_exact']);self.assertTrue(r['proposed_alternative_passed']);self.assertEqual(r['label_supported_abstention_corrections'],1)
        new['app_replay']=dict(kind='DISEASE',label='Nitrogen Deficiency');self.assertFalse(evaluate_clinical_retention([row],[new])['proposed_alternative_passed'])

    def test_correct_healthy_and_all_named_disease_remain_exact(self):
        for row in (self.case(),self.case('Other','DISEASE')):
            new=copy.deepcopy(row);new['app_replay']=dict(kind='UNSURE',label=None)
            self.assertFalse(evaluate_clinical_retention([row],[new])['proposed_alternative_passed'])

    def test_missing_label_is_protected_and_case_identity_cannot_change(self):
        for label in (None,'','unreviewed'):
            row=self.case(label);new=copy.deepcopy(row);new['app_replay']=dict(kind='UNSURE',label=None)
            self.assertFalse(evaluate_clinical_retention([row],[new])['proposed_alternative_passed'])
        new=copy.deepcopy(row);new['path']='different'
        with self.assertRaises(ValueError):evaluate_clinical_retention([row],[new])
