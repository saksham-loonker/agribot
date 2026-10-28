from pathlib import Path
import importlib.util,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
READY=importlib.util.find_spec('torch') is not None
if READY:
    import torch
    from image_bag_supervision import image_bag_losses


@unittest.skipUnless(READY,'Optional training Torch environment required')
class ImageBagSupervisionTests(unittest.TestCase):
    def mixed(self):
        z=torch.full((1,1,2,1,9),-5.);z[0,0,0,0,1]=5.;z[0,0,1,0,0]=5.
        return z,torch.ones(z.shape[:-1],dtype=torch.bool)

    def test_one_disease_witness_does_not_relabel_the_healthy_crop(self):
        z,valid=self.mixed();z.requires_grad_();clinical,binary,admission=image_bag_losses(z,torch.tensor([0]),valid)
        self.assertLess(float(clinical.detach()),.01);self.assertLess(float(binary.detach()),.01);self.assertLess(float(admission.detach()[0]),0.)
        clinical.backward();torch.testing.assert_close(z.grad[0,0,0],torch.zeros_like(z.grad[0,0,0]),rtol=0,atol=0)

    def test_healthy_label_requires_every_leaf(self):
        z,valid=self.mixed();clinical,binary,admission=image_bag_losses(z,torch.tensor([1]),valid)
        self.assertGreater(float(clinical),4.);self.assertGreater(float(binary),5.);self.assertLess(float(admission[0]),0.)

    def test_rotations_have_equal_photo_weight_and_padding_is_ignored(self):
        z,valid=self.mixed();expected=image_bag_losses(z,torch.tensor([-1]),valid)
        repeated=z.repeat(1,4,1,1,1);mask=valid.repeat(1,4,1,1)
        padded=torch.cat((repeated,torch.full_like(repeated,float('nan'))),dim=3);padded_mask=torch.cat((mask,torch.zeros_like(mask)),dim=3)
        actual=image_bag_losses(padded,torch.tensor([-1]),padded_mask)
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b)

    def test_missing_photos_fail_before_optimization(self):
        z,valid=self.mixed()
        with self.assertRaisesRegex(ValueError,'real ROI'):image_bag_losses(z,torch.tensor([0]),torch.zeros_like(valid))
