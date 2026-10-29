from pathlib import Path
import importlib.util
import sys
import unittest

READY = importlib.util.find_spec("torch") is not None
if READY:
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_training/v3"))
    from healthy_tail_risk import weighted_upper_tail


@unittest.skipUnless(READY, "Optional training environment required")
class HealthyTailRiskTests(unittest.TestCase):
    def test_fractional_family_mass_is_not_replaced_by_example_count(self):
        values = torch.tensor([1.0, 9.0, 3.0], requires_grad=True)
        risk = weighted_upper_tail(values, torch.tensor([0.8, 0.05, 0.15]), 0.1)
        self.assertAlmostEqual(float(risk), 6.0, places=5)
        risk.backward()
        torch.testing.assert_close(values.grad, torch.tensor([0.0, 0.5, 0.5]))

    def test_all_healthy_batch_has_finite_zero_negative_risk(self):
        empty = torch.empty(0, requires_grad=True)
        risk = weighted_upper_tail(empty)
        self.assertEqual(float(risk), 0.0)
        risk.backward()
        self.assertEqual(empty.grad.numel(), 0)

    def test_full_mass_retains_source_weighted_mean(self):
        values = torch.tensor([1.0, 4.0, 2.0])
        weights = torch.tensor([2.0, 1.0, 1.0])
        self.assertAlmostEqual(float(weighted_upper_tail(values, weights, 1.0)), 2.0)
