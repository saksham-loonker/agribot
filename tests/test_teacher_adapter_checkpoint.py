from pathlib import Path
import importlib.util,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
READY=all(importlib.util.find_spec(name) is not None for name in ('torch','timm'))
if READY:
    import torch
    from large_photometric_dino import trainable_state,restore_trainable


@unittest.skipUnless(READY,'Optional training Torch/timm environment required')
class TeacherCheckpointTests(unittest.TestCase):
    def test_checkpoint_round_trip_excludes_and_preserves_frozen_weights(self):
        model=torch.nn.ModuleDict(dict(base=torch.nn.Linear(4,4),adapter=torch.nn.Linear(4,4),head=torch.nn.Linear(4,9)))
        model['base'].requires_grad_(False);frozen={k:v.clone() for k,v in model['base'].state_dict().items()};saved=trainable_state(model)
        self.assertFalse(any(k.startswith('base.') for k in saved))
        with torch.no_grad():
            for p in model['adapter'].parameters():p.add_(7.)
        restore_trainable(model,saved)
        for k,v in frozen.items():torch.testing.assert_close(model['base'].state_dict()[k],v)
        for k,v in saved.items():torch.testing.assert_close(model.state_dict()[k],v)

    def test_incomplete_adapter_cannot_silently_restore(self):
        model=torch.nn.Linear(4,9);saved=trainable_state(model);saved.pop('bias')
        with self.assertRaisesRegex(AssertionError,'contract changed'):restore_trainable(model,saved)
