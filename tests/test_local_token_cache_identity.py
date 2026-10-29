from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
from local_token_cache_identity import token_calculation_identity


class LocalTokenCacheIdentityTests(unittest.TestCase):
    def test_header_only_shape_fix_does_not_change_frozen_token_math(self):
        before="shape = [2, 3]\nwith torch.inference_mode():\n    value = backbone.norm(backbone.blocks[-2:](prefix))\n"
        after=before.replace('[2, 3]','(2, 3)')
        self.assertEqual(token_calculation_identity(before),token_calculation_identity(after))

    def test_changed_encoder_suffix_cannot_reuse_the_same_math_identity(self):
        source="with torch.inference_mode():\n    value = backbone.norm(backbone.blocks[-2:](prefix))\n"
        self.assertNotEqual(token_calculation_identity(source),token_calculation_identity(source.replace('[-2:]','[-3:]')))
