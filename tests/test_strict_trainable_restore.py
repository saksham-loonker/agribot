import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
try:
    import torch
    from torch import nn
    from strict_trainable_restore import restore_trainable_exact
except ImportError:
    torch=None

@unittest.skipIf(torch is None,'PyTorch unavailable')
class RestoreTests(unittest.TestCase):
    def test_batchnorm_and_frozen_state_survive_partial_restore(self):
        model=nn.Sequential(nn.Linear(3,3),nn.BatchNorm1d(3),nn.Linear(3,1))
        model[2].requires_grad_(False)
        model[1].running_mean.fill_(2);model[1].num_batches_tracked.fill_(17)
        before={k:v.clone() for k,v in model.state_dict().items()}
        state={k:torch.full_like(v,.25) for k,v in model.named_parameters() if v.requires_grad}
        restore_trainable_exact(model,state)
        for name,value in model.state_dict().items():
            self.assertTrue(torch.equal(value,state.get(name,before[name])),name)

    def test_contract_and_dtype_rejected_before_mutation(self):
        model=nn.Linear(2,1);before={k:v.clone() for k,v in model.state_dict().items()}
        for state in ({'weight':before['weight']},{**before,'bias':before['bias'].double()}):
            with self.assertRaises(ValueError):restore_trainable_exact(model,state)
            self.assertTrue(all(torch.equal(v,before[k]) for k,v in model.state_dict().items()))
