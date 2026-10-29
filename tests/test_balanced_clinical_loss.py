from pathlib import Path
import importlib.util,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
READY=importlib.util.find_spec('torch') is not None
if READY:
    import torch
    from balanced_clinical_loss import balanced_clinical_loss


@unittest.skipUnless(READY,'Optional training Torch environment required')
class ClinicalPhotoWeightTests(unittest.TestCase):
    def test_repeating_one_photos_views_does_not_change_its_weight(self):
        a=torch.tensor([[[3.,0.,0.],[2.,0.,0.]],[[0.,0.,2.],[0.,0.,3.]]]);labels=torch.tensor([0,0])
        original=balanced_clinical_loss(a,labels,torch.ones(2,2,dtype=torch.bool),1)
        repeated=torch.full((2,8,3),float('nan'));repeated[0,:2]=a[0];repeated[1]=a[1].repeat(4,1)
        valid=torch.zeros(2,8,dtype=torch.bool);valid[0,:2]=True;valid[1]=True
        torch.testing.assert_close(balanced_clinical_loss(repeated,labels,valid,1),original)

    def test_coarse_disease_excludes_healthy_without_inventing_a_condition(self):
        a=torch.tensor([[[4.,-4.,-4.]]]);b=torch.tensor([[[-4.,-4.,4.]]]);valid=torch.ones(1,1,dtype=torch.bool)
        torch.testing.assert_close(balanced_clinical_loss(a,torch.tensor([-1]),valid,1),balanced_clinical_loss(b,torch.tensor([-1]),valid,1))

    def test_missing_photo_views_fail_before_training(self):
        with self.assertRaisesRegex(ValueError,'real view'):balanced_clinical_loss(torch.zeros(2,3,9),torch.tensor([1,8]),torch.zeros(2,3,dtype=torch.bool),1)
