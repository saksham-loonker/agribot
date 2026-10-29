from pathlib import Path
import importlib.util
import sys
import unittest

READY = all(importlib.util.find_spec(name) is not None for name in ("torch", "timm"))
if READY:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_training/v3"))
    from phase62_primary_unsupported import publisher_target


@unittest.skipUnless(READY, "Optional training environment required")
class PrimaryConditionScopeTests(unittest.TestCase):
    def test_unsupported_conditions_can_share_one_primary_target(self):
        self.assertEqual(
            publisher_target({"Tomato leaf yellow virus", "Tomato mold leaf"}), 1
        )
        self.assertEqual(publisher_target({"Tomato leaf"}), 0)

    def test_mixed_healthy_supported_and_foreign_taxonomy_are_masked(self):
        for labels in (
            {"Tomato leaf", "Tomato leaf yellow virus"},
            {"Tomato Early blight leaf"},
            {"Tomato leaf late blight", "Tomato mold leaf"},
            {"Apple leaf"},
            set(),
        ):
            with self.subTest(labels=labels):
                self.assertEqual(publisher_target(labels), -1)
