import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
try:
    import torch
    from meta_domain_loss import meta_domain_loss
except ImportError:torch=None

@unittest.skipIf(torch is None,'PyTorch unavailable')
class MetaDomainTests(unittest.TestCase):
    def test_gradient_contains_inner_update_hessian(self):
        model=torch.nn.Linear(1,1,bias=False).double();model.weight.data.fill_(2)
        def loss(target):
            def compute(params):
                w=model.weight if params is None else params['weight']
                return (w.reshape(())-target)**2
            return compute
        objective,support,query=meta_domain_loss(model,loss(0),loss(1),.1)
        objective.backward();self.assertAlmostEqual(float(objective.detach()),2.18);self.assertAlmostEqual(float(model.weight.grad),2.48)
        self.assertAlmostEqual(float(support),4.);self.assertAlmostEqual(float(query),.36)
        self.assertEqual(float(model.weight.detach()),2.)

    def test_zero_step_matches_pooled_erm_gradient(self):
        model=torch.nn.Linear(1,1,bias=False).double();model.weight.data.fill_(2)
        support=lambda params:(model.weight if params is None else params['weight']).square().sum()
        query=lambda params:((model.weight if params is None else params['weight'])-1).square().sum()
        objective,_,_=meta_domain_loss(model,support,query,0.);objective.backward()
        self.assertAlmostEqual(float(objective.detach()),2.5);self.assertAlmostEqual(float(model.weight.grad),3.)

    def test_frozen_parameters_remain_present_and_unchanged(self):
        model=torch.nn.Linear(1,1).double();model.weight.data.fill_(2);model.bias.data.fill_(.5);model.bias.requires_grad_(False)
        def support(params):return (model.weight+model.bias).square().sum()
        def query(params):
            self.assertIs(params['bias'],model.bias)
            return (params['weight']+params['bias']-1).square().sum()
        objective,_,_=meta_domain_loss(model,support,query,.1);objective.backward()
        self.assertIsNotNone(model.weight.grad);self.assertIsNone(model.bias.grad);self.assertEqual(float(model.bias),.5)
