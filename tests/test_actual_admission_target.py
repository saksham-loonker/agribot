from pathlib import Path
import copy,importlib.util,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
READY=importlib.util.find_spec('torch') is not None and importlib.util.find_spec('ai_edge_litert') is not None
if READY:
    from actual_admission_target import actual_admission_target
    from phase83_eligible_operating_points import eligible


@unittest.skipUnless(READY,'Optional training Torch/LiteRT environment required')
class ActualAdmissionTargetTests(unittest.TestCase):
    def fixture(self):
        row=dict(app_replay=dict(kind='UNSURE',label='Other',probabilities=[.025]*8+[.8]),contrast=.2,sharpness=.2)
        packet=dict(usable=True,whole=[.9,1.,.9,0.],leaves=[[[.6,.9],[.6,.9]]])
        return row,copy.deepcopy(row),packet,{}

    def test_original_eligibility_does_not_override_retained_suspected_disease(self):
        row,retained,packet,manifest=self.fixture();retained['app_replay']['label']='Early_blight'
        self.assertTrue(eligible(row,packet));self.assertFalse(actual_admission_target(row,retained,packet,manifest))

    def test_original_eligibility_does_not_bypass_final_visibility_guard(self):
        row,retained,packet,manifest=self.fixture();retained.update(contrast=0.,sharpness=0.)
        self.assertTrue(eligible(row,packet));self.assertFalse(actual_admission_target(row,retained,packet,manifest))

    def test_fully_admissible_photo_can_be_a_training_target(self):
        row,retained,packet,manifest=self.fixture();self.assertTrue(actual_admission_target(row,retained,packet,manifest))

    def test_old_healthy_and_disease_are_never_new_admission_targets(self):
        for kind in ('HEALTHY','DISEASE'):
            row,retained,packet,manifest=self.fixture();retained['app_replay']['kind']=kind
            self.assertFalse(actual_admission_target(row,retained,packet,manifest))
