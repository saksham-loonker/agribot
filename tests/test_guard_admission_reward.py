from pathlib import Path
import importlib.util,math,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
READY=importlib.util.find_spec('torch') is not None
if READY:
    import torch
    from guard_admission_reward import guard_admission_reward_loss


@unittest.skipUnless(READY,'Optional training Torch environment required')
class GuardAdmissionRewardTests(unittest.TestCase):
    def test_gradients_raise_healthy_and_lower_disease_at_the_boundary(self):
        z=torch.full((2,),math.log(19.),requires_grad=True);loss=guard_admission_reward_loss(z,torch.tensor([1.,0.]),torch.ones(2));loss.backward()
        self.assertLess(float(z.grad[0]),0.);self.assertGreater(float(z.grad[1]),0.);self.assertAlmostEqual(float(z.grad[1]/-z.grad[0]),8.)

    def test_exact_reward_enumerates_both_binary_actions(self):
        z=torch.tensor([math.log(19.)]);actual=guard_admission_reward_loss(z,torch.ones(1),torch.ones(1))
        torch.testing.assert_close(actual,torch.tensor(-.5))

    def test_ineligible_cases_receive_no_policy_gradient(self):
        z=torch.tensor([0.,3.],requires_grad=True);loss=guard_admission_reward_loss(z,torch.tensor([1.,0.]),torch.zeros(2));loss.backward()
        torch.testing.assert_close(z.grad,torch.zeros_like(z),rtol=0,atol=0)
