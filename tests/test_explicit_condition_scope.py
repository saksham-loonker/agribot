from pathlib import Path
import importlib.util
import sys
import unittest

READY = all(importlib.util.find_spec(name) is not None for name in ("torch", "timm"))
if READY:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_training/v3"))
    from phase51_dino_train import condition_target


@unittest.skipUnless(READY, "Optional training environment required")
class ExplicitConditionScopeTests(unittest.TestCase):
    def test_named_unsupported_condition_is_other_only_in_explicit_control(self):
        row = dict(label="Other", raw_label="Yellow Leaf Curl Virus")
        self.assertEqual(condition_target(row, False), -1)
        self.assertEqual(condition_target(row, False, True), 8)

    def test_coarse_and_missing_condition_never_become_other(self):
        for raw in (
            None,
            "",
            "Unknown",
            " Diseased ",
            "Diseased_unspecified",
            "Unhealthy",
        ):
            with self.subTest(raw=raw):
                self.assertEqual(
                    condition_target(dict(label="Other", raw_label=raw), False, True),
                    -1,
                )
        self.assertEqual(condition_target(None, False, True), -1)

    def test_original_named_disease_and_healthy_targets_survive(self):
        self.assertEqual(condition_target(dict(label="Late_blight"), False, True), 2)
        self.assertEqual(condition_target(None, True, True), 1)
