"""Regression checks for recovery defects, optional ML dependencies required."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

if any(
    importlib.util.find_spec(name) is None
    for name in ("torch", "timm", "torchvision", "sklearn")
):
    raise unittest.SkipTest(
        "v3 recovery checks require the optional training environment"
    )

import numpy as np
import torch

V3 = Path(__file__).resolve().parents[1] / "model_training" / "v3"
sys.path.insert(0, str(V3))
from training_support import (
    checkpoint_weights,
    validate_cache,
    parameter_groups,
    set_trainable,
    capture_rng,
    restore_rng,
    assert_resume_compatible,
    retention_pass,
)
from train import LABELS, sampling_weights
from runner import normalized_cache_input
from calibrate import ece


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = torch.nn.Linear(3, 3)
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(3, 3) for _ in range(3)])
        self.norm = torch.nn.LayerNorm(3)
        self.head = torch.nn.Linear(3, 9)


class TrainingRecoveryTest(unittest.TestCase):
    def test_normalization_preserves_uint8_contract(self):
        raw = torch.tensor([[[[0, 128, 255]]]], dtype=torch.uint8)
        got = normalized_cache_input(raw, 1).flatten().numpy()
        expected = (
            np.array([0, 128, 255]) / 255 - np.array([0.485, 0.456, 0.406])
        ) / np.array([0.229, 0.224, 0.225])
        np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_checkpoint_rejects_permuted_head_labels(self):
        labels = list(reversed(LABELS))
        with self.assertRaisesRegex(ValueError, "label order"):
            checkpoint_weights({"model": {}, "labels": labels}, LABELS)

    def test_legacy_weights_require_metadata(self):
        with self.assertRaisesRegex(ValueError, "meta.json"):
            checkpoint_weights({"head.weight": torch.zeros(9, 3)}, LABELS)

    def test_architecture_is_not_inferred_from_shapes(self):
        state = dict(model={}, labels=LABELS, args=dict(model="wrong", img=224))
        with self.assertRaisesRegex(ValueError, "architecture"):
            checkpoint_weights(state, LABELS, "correct", 224)

    def test_cache_rejects_float_or_stale_label_order(self):
        records = [
            dict(cache_index=0, label="Healthy", split="train", group="a"),
            dict(cache_index=1, label="Other", split="val", group="b"),
        ]
        with self.assertRaisesRegex(ValueError, "uint8"):
            validate_cache(np.zeros((2, 4, 4, 3), np.float32), records, LABELS, 4)
        records[1]["cache_index"] = 0
        with self.assertRaisesRegex(ValueError, "alignment"):
            validate_cache(np.zeros((2, 4, 4, 3), np.uint8), records, LABELS, 4)

    def test_cache_rejects_group_leakage(self):
        rows = [
            dict(cache_index=i, label="Healthy", split=s, group="same")
            for i, s in enumerate(["train", "val"])
        ]
        with self.assertRaisesRegex(ValueError, "crosses"):
            validate_cache(np.zeros((2, 4, 4, 3), np.uint8), rows, LABELS, 4)

    def test_uniform_sampler_does_not_reweight_repeated_groups(self):
        meta = dict(
            split=["train"] * 3,
            label=["Healthy", "Healthy", "Other"],
            source=["a"] * 3,
            group=["x", "x", "y"],
        )
        _, weights, prior = sampling_weights(
            meta, SimpleNamespace(sampling="uniform", max_class_weight=4)
        )
        np.testing.assert_allclose(weights.numpy(), np.ones(3) / 3)
        self.assertAlmostEqual(prior[1], 2 / 3)

    def test_optimizer_covers_norm_and_future_unfrozen_blocks(self):
        m = TinyClassifier()
        groups = parameter_groups(m, 1e-3, 1e-4, 0.8, 0.03)
        ids = [id(p) for g in groups for p in g["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in m.parameters()})
        rates = {id(p): g["lr"] for g in groups for p in g["params"]}
        self.assertGreater(rates[id(m.blocks[2].weight)], rates[id(m.blocks[0].weight)])
        self.assertEqual(rates[id(m.norm.weight)], 1e-4)
        set_trainable(m, True, 1)
        self.assertFalse(m.blocks[2].weight.requires_grad)
        set_trainable(m, False, 1)
        self.assertTrue(m.blocks[2].weight.requires_grad)
        self.assertFalse(m.blocks[1].weight.requires_grad)

    def test_rng_restore_reproduces_draws(self):
        state = capture_rng()
        a = torch.rand(3)
        b = np.random.rand(3)
        restore_rng(state)
        torch.testing.assert_close(torch.rand(3), a)
        np.testing.assert_array_equal(np.random.rand(3), b)

    def test_resume_rejects_legacy_or_changed_data(self):
        with self.assertRaisesRegex(ValueError, "Legacy"):
            assert_resume_compatible({}, {}, "manifest", "cache")
        state = dict(schema=2, data_manifest_sha256="old", cache_sha256="cache")
        with self.assertRaisesRegex(ValueError, "identity"):
            assert_resume_compatible(state, {}, "manifest", "cache")

    def test_resume_rejects_loss_change(self):
        state = dict(
            schema=2, data_manifest_sha256="m", cache_sha256="c", args=dict(loss="ce")
        )
        with self.assertRaisesRegex(ValueError, "configuration"):
            assert_resume_compatible(state, dict(loss="weighted"), "m", "c")

    def test_healthy_gain_cannot_hide_disease_collapse(self):
        reference = dict(
            per_class_recall={x: 0.9 for x in LABELS}, false_healthy_rate=0.01
        )
        candidate = dict(
            per_class_recall=dict(reference["per_class_recall"]),
            false_healthy_rate=0.01,
        )
        candidate["per_class_recall"]["Healthy"] = 1.0
        candidate["per_class_recall"]["Early_blight"] = 0.3
        ok, failures = retention_pass(candidate, reference, LABELS, 0.02, 0.005)
        self.assertFalse(ok)
        self.assertIn("Early_blight", failures)

    def test_ece_bins_do_not_double_count_boundary_confidence(self):
        value, rows = ece(
            np.array([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]]), np.array([0, 0, 1]), bins=2
        )
        self.assertEqual(sum(r["n"] for r in rows), 3)

    @unittest.skipUnless(
        importlib.util.find_spec("ai_edge_litert"), "Optional LiteRT runtime required"
    )
    def test_mixed_known_scene_is_not_an_unsupported_disease(self):
        from app_eval import scope_of, summarize

        row = dict(
            label="Mixed",
            source="plantdoc",
            objects=[
                dict(source_label="Tomato leaf"),
                dict(source_label="Tomato Early blight leaf"),
            ],
            app_replay=dict(kind="HEALTHY", label="Healthy"),
        )
        self.assertEqual(scope_of(row), "mixed")
        metrics = summarize([row])
        self.assertEqual(metrics["excluded_mixed_photos"], 1)
        self.assertEqual(metrics["unsupported_tomato"]["n"], 0)

    @unittest.skipUnless(
        importlib.util.find_spec("ai_edge_litert"), "Optional LiteRT runtime required"
    )
    def test_coarse_disease_does_not_turn_abstention_into_a_named_diagnosis_error(self):
        from app_eval import scope_of, summarize

        row = dict(
            label="Diseased_unspecified",
            source="binary",
            species="tomato",
            app_replay=dict(kind="UNKNOWN_CONDITION", label="Other"),
        )
        self.assertEqual(scope_of(row), "coarse_diseased")
        metrics = summarize([row])
        self.assertEqual(metrics["supported_disease"]["n"], 0)
        self.assertEqual(metrics["coarse_diseased"]["n"], 1)
        self.assertEqual(metrics["all"]["false_other"], 0)
        row["app_replay"] = dict(kind="HEALTHY", label="Healthy")
        self.assertEqual(summarize([row])["coarse_diseased"]["false_healthy"], 1)

    def test_paired_comparison_rejects_duplicate_predictions(self):
        import json
        from compare import read

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "rows.jsonl"
            p.write_text((json.dumps(dict(id="same")) + "\n") * 2)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read(p)

    def test_retention_does_not_anchor_teacher_healthy_mistakes(self):
        from retention import reliable_retention

        student = torch.tensor([[0.0, 3.0, 0.0], [0.0, 0.0, 3.0]], requires_grad=True)
        teacher = torch.tensor([[3.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
        labels = torch.tensor([1, 2])
        loss = reliable_retention(student, teacher, labels, 1, 2.0)
        self.assertLess(abs(float(loss)), 1e-6)
        loss.backward()
        torch.testing.assert_close(student.grad[0], torch.zeros(3))

    def test_conservative_adapter_preserves_strong_disease_and_other(self):
        from export_head_blend import conservative_mix

        base = torch.full((2, 9), 0.01)
        base[0, 0] = 0.92
        base[1, 8] = 0.92
        adapted = torch.full((2, 9), 0.005)
        adapted[:, 1] = 0.96
        got = conservative_mix(
            base, adapted, 0.5, 0.4, torch.tensor([0.0] * 8 + [0.75]), 0.881470084
        )
        torch.testing.assert_close(got, base)

    def test_conservative_adapter_only_raises_uncertain_healthy_probability(self):
        from export_head_blend import conservative_mix

        base = torch.full((1, 9), 0.25 / 7)
        base[0, 1] = 0.45
        base[0, 3] = 0.3
        adapted = torch.full((1, 9), 0.0125)
        adapted[0, 1] = 0.9
        got = conservative_mix(
            base, adapted, 0.5, 0.4, torch.tensor([0.0] * 8 + [0.75]), 0.881470084
        )
        self.assertGreater(float(got[0, 1]), 0.6)
        torch.testing.assert_close(got.sum(1), torch.ones(1))
        unchanged = conservative_mix(
            adapted, base, 0.5, 0.4, torch.tensor([0.0] * 8 + [0.75]), 0.881470084
        )
        torch.testing.assert_close(unchanged, adapted)


if __name__ == "__main__":
    unittest.main()
