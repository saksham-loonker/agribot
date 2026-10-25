"""Semantic and gradient checks for the independently compared loss families."""

import importlib.util
import sys
from pathlib import Path
import unittest

READY = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "timm", "sklearn", "ai_edge_litert")
)
if READY:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_training/v3"))
    import torch
    import campaign_heads as campaign
    import campaign_factorized as factorized
    from campaign_cnn_fusion import ExpertFusion
    from campaign_consensus import agreement
    from campaign_adapter import LoRALinear, LastBlockExpert, prefix


@unittest.skipUnless(READY, "full training/evaluation dependencies are optional")
class CampaignLossTests(unittest.TestCase):
    def test_whole_image_evidence_cannot_replace_disease_or_fabricate_a_leaf(self):
        from context_abstention import route_context

        disease = dict(kind="DISEASE", label="Early_blight")
        row = dict(contrast=0.2, sharpness=0.1, leaves=[], app_replay=disease)
        result = route_context(row, row, {}, True, False, 0.99, False, False)
        self.assertEqual(result["app_replay"], disease)
        self.assertNotIn("whole_frame_evidence", result)
        unknown = dict(row, app_replay=dict(kind="UNKNOWN_CONDITION", label="Other"))
        blocked = route_context(unknown, unknown, {}, True, False, 0.99, True, True)
        self.assertEqual(blocked["app_replay"], unknown["app_replay"])
        accepted = route_context(unknown, unknown, {}, True, False, 0.99, False, False)
        self.assertEqual(accepted["app_replay"]["kind"], "HEALTHY")
        self.assertEqual(accepted["app_replay"]["scope"], "whole_frame")
        self.assertEqual(accepted["app_replay"]["leaves"], 0)
        self.assertEqual(accepted["leaves"], [])
        self.assertEqual(unknown["app_replay"]["kind"], "UNKNOWN_CONDITION")

    def test_new_healthy_leaf_verdict_requires_context_agreement(self):
        from context_abstention import route_context

        reference = dict(
            contrast=0.2,
            sharpness=0.1,
            leaves=[],
            app_replay=dict(kind="UNSURE", label="Other"),
        )
        row = dict(reference, app_replay=dict(kind="HEALTHY", label="Healthy"))
        result = route_context(row, reference, {}, False, False, 0.8, False, False)
        self.assertEqual(result["app_replay"], reference["app_replay"])
        low_detail = dict(row, contrast=0.0, sharpness=0.0)
        self.assertEqual(
            route_context(low_detail, reference, {}, True, False, 0.99, False, False)[
                "app_replay"
            ],
            row["app_replay"],
        )

    def test_strong_healthy_contradiction_abstains_without_naming_a_disease(self):
        import numpy as np
        from app_eval import replay, posterior
        from context_abstention import abstain_healthy

        manifest = dict(
            classifier=dict(
                labels=campaign.LABELS,
                other_index=8,
                other_logit_bias=0.75,
                temperature=0.8814700841903687,
            )
        )
        neutral = np.zeros(9, np.float32)
        neutral[8] = -0.75
        np.testing.assert_allclose(
            posterior([neutral], manifest), np.full(9, 1 / 9), atol=1e-7
        )
        rng = np.random.default_rng(307)
        checked = 0
        for _ in range(250):
            leaves = []
            for i in range(4):
                z = rng.normal(0, 0.8, 9)
                z[1] += 5 if i < 3 else 0
                leaves.append(
                    dict(member_logits=[z.tolist()], score=float(rng.uniform(0.4, 1.0)))
                )
            verdict = replay(leaves, manifest)
            if verdict["kind"] != "HEALTHY":
                continue
            row = dict(leaves=leaves, quality_weight=1.0, app_replay=verdict)
            after = abstain_healthy(row, manifest)
            self.assertNotIn(
                after["app_replay"]["kind"], ("HEALTHY", "DISEASE", "UNKNOWN_CONDITION")
            )
            self.assertEqual(row["app_replay"]["kind"], "HEALTHY")
            checked += 1
        self.assertGreater(checked, 100)

    def test_species_targets_mask_mixed_scenes_and_unverified_field_rois(self):
        from phase7_species_train import species_targets

        original = [
            dict(
                id="tomato",
                source="plantdoc",
                path="mixed.jpg",
                raw_label="Tomato leaf",
            ),
            dict(
                id="apple", source="plantdoc", path="mixed.jpg", raw_label="Apple leaf"
            ),
        ]
        rows = [
            dict(id="tomato", source="plantdoc", path="mixed.jpg", box=[0, 0, 8, 8]),
            dict(id="apple", source="plantdoc", path="mixed.jpg", box=[8, 8, 16, 16]),
            dict(id="scene", source="plantdoc", path="mixed.jpg", box=None),
            dict(id="field", source="tom2024", path="field.jpg", box=None),
            dict(id="field_roi", source="tom2024", path="field.jpg", box=[0, 0, 8, 8]),
        ]
        self.assertEqual(species_targets(rows, original).tolist(), [1, 0, -1, 1, -1])

    def test_optional_unknown_guard_preserves_the_old_policy_default(self):
        import numpy as np
        from view_guard_policy import needs_confirmation

        healthy = np.eye(9, dtype=np.float32)[1]
        unknown = np.eye(9, dtype=np.float32)[8]
        self.assertFalse(needs_confirmation(healthy, unknown))
        self.assertTrue(needs_confirmation(healthy, unknown, confirm_unknown=True))
        self.assertFalse(needs_confirmation(unknown, unknown, confirm_unknown=True))

    def test_coarse_negative_cost_changes_the_healthy_gradient(self):
        from phase5_train import supervised_loss

        target = torch.tensor([-1])
        health = torch.zeros(1)
        reference = torch.zeros(1, 9)
        gradients = []
        for cost in (2.0, 8.0):
            logits = torch.zeros(1, 9, requires_grad=True)
            supervised_loss(
                logits, target, health, reference, negative_cost=cost
            ).backward()
            gradients.append(logits.grad.clone())
        torch.testing.assert_close(gradients[1], 4 * gradients[0])
        self.assertGreater(float(gradients[1][0, 1]), 0.0)

    def test_batched_guard_keys_match_canonical_app_calibration(self):
        import numpy as np
        from app_eval import posterior
        from phase5_guard_select import keys
        from view_guard_policy import decision_key

        rng = np.random.default_rng(211)
        logits = rng.normal(0, 10, (400, 9)).astype(np.float32)
        logits[0] = 0.0
        manifest = dict(
            classifier=dict(
                other_index=8, other_logit_bias=0.75, temperature=0.8814700841903687
            )
        )
        observed, positive = keys(logits)
        kinds = dict(UNSURE=0, HEALTHY=1, DISEASE=2, UNKNOWN_CONDITION=3)
        expected = [decision_key(posterior([z], manifest)) for z in logits]
        self.assertEqual(
            observed.tolist(), [kinds[k] * 9 + label for k, label in expected]
        )
        self.assertEqual(
            positive.tolist(), [k in ("HEALTHY", "DISEASE") for k, _ in expected]
        )

    def test_deeper_adapter_warmstart_preserves_the_trained_tail(self):
        from timm.models.vision_transformer import VisionTransformer
        from phase2_multiclass import MulticlassExpert
        from phase5_train import extend_lora

        model = VisionTransformer(
            img_size=28,
            patch_size=14,
            embed_dim=32,
            depth=5,
            num_heads=4,
            num_classes=9,
        ).eval()
        model.requires_grad_(False)
        old = MulticlassExpert(model, "lora", rank=2, depth=1).eval()
        with torch.no_grad():
            for layer in old.modules():
                if isinstance(layer, LoRALinear):
                    layer.b.weight.normal_(0, 0.01)
        new = MulticlassExpert(model, "lora", rank=2, depth=4).eval()
        extend_lora(new, dict(mode="lora", depth=1, model=old.state_dict()))
        x = torch.randn(2, 3, 28, 28)
        torch.testing.assert_close(
            new(prefix(model, x, 4)), old(prefix(model, x, 1)), atol=1e-5, rtol=1e-5
        )

    def test_binary_only_disease_reduces_healthy_without_inventing_a_name(self):
        from phase5_train import supervised_loss

        logits = torch.zeros(1, 9, requires_grad=True)
        supervised_loss(
            logits, torch.tensor([-1]), torch.tensor([0.0]), torch.zeros(1, 9)
        ).backward()
        self.assertGreater(float(logits.grad[0, 1]), 0.0)
        alternatives = torch.cat([logits.grad[0, :1], logits.grad[0, 2:]])
        torch.testing.assert_close(alternatives, alternatives[0].repeat(8))

    def test_absent_class_metrics_are_null_and_serialize_as_strict_json(self):
        import json

        metrics = campaign.classification_metrics(
            torch.zeros(2, 9), torch.tensor([1, 1])
        )
        self.assertIsNone(metrics["per_class_recall"]["Late_blight"])
        json.dumps(metrics, allow_nan=False)
        self.assertTrue(torch.isfinite(torch.tensor(campaign.val_score(metrics))))

    def test_adapted_reward_arm_increases_false_healthy_penalty(self):
        from phase2_multiclass import adaptation_loss

        gradients = {}
        for method in ["ce", "reward_ce"]:
            logits = torch.zeros(2, 9, requires_grad=True)
            adaptation_loss(method, logits, torch.tensor([1, 0])).backward()
            gradients[method] = logits.grad.clone()
        self.assertLess(
            float(gradients["reward_ce"][0, 1]), float(gradients["ce"][0, 1])
        )
        self.assertGreater(
            float(gradients["reward_ce"][1, 1]), float(gradients["ce"][1, 1])
        )

    def test_adapted_ppo_has_finite_gradients_and_keeps_old_policy_frozen(self):
        from phase2_multiclass import adaptation_loss

        logits = torch.zeros(18, 9, requires_grad=True)
        old = torch.zeros(18, 9, requires_grad=True)
        loss = adaptation_loss(
            "ppo",
            logits,
            torch.arange(18) % 9,
            old_logits=old,
            generator=torch.Generator().manual_seed(9),
        )
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertIsNone(old.grad)

    def test_false_healthy_has_higher_cost_than_generic_confusion(self):
        rewards = campaign.reward_table(torch.tensor([0, 1, 8]))
        self.assertEqual(float(rewards[0, 0]), 1.0)
        self.assertEqual(float(rewards[1, 1]), 2.0)
        self.assertEqual(float(rewards[2, 8]), 1.0)
        self.assertLess(float(rewards[0, 1]), float(rewards[0, 2]))
        self.assertLess(float(rewards[0, 8]), float(rewards[0, 2]))
        self.assertEqual(float(rewards[2, 1]), -8.0)

    def test_expected_reward_moves_correct_healthy_and_disease_in_opposite_directions(
        self,
    ):
        z = torch.zeros(2, 9, requires_grad=True)
        y = torch.tensor([1, 0])
        loss = campaign.loss_value(
            "expected_reward", z, y, z.detach(), dict(sample=torch.ones(2))
        )
        loss.backward()
        self.assertLess(float(z.grad[0, 1]), 0.0)
        self.assertGreater(float(z.grad[1, 1]), 0.0)
        self.assertLess(float(z.grad[1, 0]), 0.0)

    def test_ppo_clips_improvements_beyond_the_trust_region(self):
        for ratio, advantage in [(1.3, 1.0), (0.7, -1.0)]:
            logp = torch.tensor(ratio).log().requires_grad_()
            campaign.ppo_objective(
                logp, torch.tensor(0.0), torch.tensor(advantage)
            ).backward()
            self.assertEqual(float(logp.grad), 0.0)

    def test_residual_head_initially_preserves_original_logits_exactly(self):
        layer = torch.nn.Linear(384, 9)
        x = torch.randn(5, 384)
        model = campaign.FeatureHead(layer.state_dict(), "residual_mlp")
        torch.testing.assert_close(model(x), layer(x), rtol=0, atol=0)

    def test_all_loss_families_have_finite_gradients(self):
        torch.manual_seed(17)
        y = torch.arange(9).repeat(2)
        for method in campaign.METHODS:
            if method == "supcon":
                continue
            z = torch.randn(18, 9, requires_grad=True)
            loss = campaign.loss_value(
                method,
                z,
                y,
                torch.randn(18, 9),
                dict(sample=torch.ones(18), prior=torch.ones(9) / 9),
            )
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(z.grad).all(), method)

    def test_contrastive_loss_handles_no_positive_pairs(self):
        hidden = torch.randn(3, 128, requires_grad=True)
        loss = campaign.supervised_contrastive(hidden, torch.arange(3))
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertTrue(torch.isfinite(hidden.grad).all())

    def test_factorized_head_initially_preserves_original_logits_exactly(self):
        layer = torch.nn.Linear(384, 9)
        x = torch.randn(12, 384)
        model = factorized.FactorizedHead(layer.state_dict())
        torch.testing.assert_close(model(x), layer(x), rtol=0, atol=0)

    def test_factorized_head_retains_non_miner_disease_and_strong_miner(self):
        layer = torch.nn.Linear(384, 9)
        with torch.no_grad():
            layer.weight.zero_()
            layer.bias.zero_()
        for winner in [0, 2, 3, 4, 5, 6, 7]:
            with torch.no_grad():
                layer.bias.zero_()
                layer.bias[winner] = 8.0
            model = factorized.FactorizedHead(layer.state_dict())
            with torch.no_grad():
                model.pair[-1].bias.fill_(100.0)
                model.known[-1].bias.fill_(100.0)
            x = torch.randn(2, 384)
            torch.testing.assert_close(model(x), layer(x), rtol=0, atol=0)

    def test_factorized_head_leaves_unknown_unchanged_without_knownness_confirmation(
        self,
    ):
        layer = torch.nn.Linear(384, 9)
        with torch.no_grad():
            layer.weight.zero_()
            layer.bias.zero_()
            layer.bias[8] = 8.0
        model = factorized.FactorizedHead(layer.state_dict())
        with torch.no_grad():
            model.pair[-1].bias.fill_(100.0)
        x = torch.randn(2, 384)
        torch.testing.assert_close(model(x), layer(x), rtol=0, atol=0)

    def test_fast_single_leaf_consensus_matches_full_reference(self):
        import numpy as np

        manifest = {
            "classifier": {
                "other_index": 8,
                "other_logit_bias": 0.75,
                "temperature": 0.8814700841903687,
            }
        }
        rng = np.random.default_rng(19)
        for logits in rng.normal(size=(150, 9)) * 3:
            leaves = [dict(member_logits=[logits.tolist()], score=0.41)]
            slow = campaign.replay(leaves, manifest, 0.3)
            fast = campaign.fast_static_replay(leaves, manifest, 0.3)
            for key in ["kind", "label", "frames", "leaves", "partial"]:
                self.assertEqual(fast[key], slow[key])
            self.assertAlmostEqual(fast["confidence"], slow["confidence"], places=6)

    def test_cnn_disabled_fusion_is_exact_identity(self):
        reference = torch.randn(50, 9)
        actual = ExpertFusion(0.0, 0.0, 0.0)(reference, torch.randn(50, 2) * 100)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_cnn_cannot_override_even_an_uncertain_named_disease(self):
        reference = torch.zeros(7, 9)
        for row, winner in enumerate([0, 2, 3, 4, 5, 6, 7]):
            reference[row, winner] = 1.0
        actual = ExpertFusion(1.0, 0.0, 1.0)(reference, torch.full((7, 2), 100.0))
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_cnn_preserves_confident_original_healthy_on_unfamiliar_domain(self):
        reference = torch.zeros(1, 9)
        reference[0, 1] = 8.0
        actual = ExpertFusion(1.0, 1.0, 1.0)(reference, torch.tensor([[-100.0, 100.0]]))
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_cnn_can_resolve_uncertain_healthy_without_changing_disease_odds(self):
        reference = torch.zeros(1, 9)
        reference[0, 1] = 2.0
        actual = ExpertFusion(1.0, 0.0, 0.0)(reference, torch.tensor([[8.0, 8.0]]))
        before = torch.softmax(
            (reference + torch.tensor([0.0] * 8 + [0.75])) / factorized.T, 1
        )
        after = torch.softmax(
            (actual + torch.tensor([0.0] * 8 + [0.75])) / factorized.T, 1
        )
        self.assertLess(float(before[0, 1]), 0.6)
        self.assertGreater(float(after[0, 1]), 0.99)
        torch.testing.assert_close(
            before[0, 0] / before[0, 3], after[0, 0] / after[0, 3]
        )
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(after.sum(1), torch.ones(1))

    def test_cnn_healthy_vote_alone_does_not_erase_unknown(self):
        reference = torch.zeros(1, 9)
        reference[0, 8] = 4.0
        actual = ExpertFusion(1.0, 0.0, 1.0)(reference, torch.tensor([[100.0, -100.0]]))
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_binary_knownness_expert_cannot_invent_a_named_disease(self):
        reference = torch.zeros(7, 9)
        reference[:, 8] = 4.0
        for row, winner in enumerate([0, 2, 3, 4, 5, 6, 7]):
            reference[row, winner] = 2.0
        actual = ExpertFusion(1.0, 0.0, 1.0)(reference, torch.full((7, 2), 100.0))
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_healthy_boost_requires_independent_knownness_confirmation(self):
        reference = torch.zeros(1, 9)
        reference[0, 1] = 2.0
        actual = ExpertFusion(1.0, 0.0, 0.0)(reference, torch.tensor([[100.0, -100.0]]))
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_independent_expert_agreement_uses_the_less_confident_vote(self):
        head = torch.zeros(3, 9)
        head[:, 1] = torch.tensor([10.0, -10.0, 10.0])
        cnn = torch.tensor([[6.0, 10.0], [-6.0, 10.0], [-6.0, 10.0]])
        combined = agreement(head, cnn)
        self.assertAlmostEqual(float(combined[0, 0]), 6.0)
        self.assertAlmostEqual(float(combined[1, 0]), -6.0)
        self.assertEqual(float(combined[2, 0]), 0.0)
        self.assertTrue(torch.isfinite(combined).all())

    def test_lora_starts_as_identity_and_keeps_base_weights_frozen(self):
        base = torch.nn.Linear(5, 7)
        layer = LoRALinear(base, rank=2)
        x = torch.randn(3, 5)
        torch.testing.assert_close(layer(x), base(x), rtol=0, atol=0)
        layer(x).square().sum().backward()
        self.assertIsNone(layer.base.weight.grad)
        self.assertGreater(float(layer.b.weight.grad.abs().sum()), 0.0)

    def test_shared_prefix_preserves_original_model_and_initial_adapter(self):
        from timm.models.vision_transformer import VisionTransformer

        model = VisionTransformer(
            img_size=28,
            patch_size=14,
            embed_dim=32,
            depth=2,
            num_heads=4,
            num_classes=9,
        ).eval()
        x = torch.randn(2, 3, 28, 28)
        tokens = prefix(model, x)
        original = model.head(model.norm(model.blocks[-1](tokens))[:, 0])
        torch.testing.assert_close(original, model(x))
        expert = LastBlockExpert(model, "lora", rank=2).eval()
        lp = torch.log_softmax(
            (original + torch.tensor([0.0] * 8 + [0.75])) / factorized.T, 1
        )
        expected = torch.stack(
            [
                lp[:, 1] - torch.cat([lp[:, :1], lp[:, 2:]], 1).logsumexp(1),
                lp[:, :8].logsumexp(1) - lp[:, 8],
            ],
            1,
        )
        torch.testing.assert_close(expert(tokens), expected)

    def test_full_block_control_unfreezes_its_copy_only(self):
        from timm.models.vision_transformer import VisionTransformer

        model = VisionTransformer(
            img_size=28,
            patch_size=14,
            embed_dim=32,
            depth=2,
            num_heads=4,
            num_classes=9,
        )
        model.requires_grad_(False)
        expert = LastBlockExpert(model, "block")
        self.assertTrue(all(p.requires_grad for p in expert.block.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        self.assertTrue(
            all(not p.requires_grad for p in expert.original_head.parameters())
        )
        self.assertTrue(all(not p.requires_grad for p in expert.norm.parameters()))

    def test_negative_binary_evidence_preserves_named_disease_scores(self):
        for winner in [0, 2, 3, 4, 5, 6, 7]:
            reference = torch.zeros(1, 9)
            reference[0, winner] = 1.0
            actual = ExpertFusion(
                1.0, 1.0, 1.0, negative_policy="abstain", require_healthy_for_known=True
            )(reference, torch.tensor([[-100.0, 100.0]]))
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_not_healthy_evidence_abstains_without_inventing_a_diagnosis(self):
        p = torch.tensor([[0.005, 0.58, 0.005, 0.26, 0.005, 0.01, 0.11, 0.02, 0.005]])
        reference = p.log() * factorized.T - torch.tensor([0.0] * 8 + [0.75])
        actual = ExpertFusion(1.0, 1.0, 0.0, negative_policy="abstain")(
            reference, torch.tensor([[-100.0, 100.0]])
        )
        after = torch.softmax(
            (actual + torch.tensor([0.0] * 8 + [0.75])) / factorized.T, 1
        )
        self.assertLess(float(after[0, 1]), float(p[0, 1]))
        self.assertLess(float(torch.cat([after[:, :1], after[:, 2:]], 1).max()), 0.5)
        torch.testing.assert_close(after.sum(1), torch.ones(1))

    def test_knownness_cannot_recover_healthy_when_healthy_expert_disagrees(self):
        p = torch.tensor([[0.005, 0.22, 0.005, 0.04, 0.005, 0.01, 0.01, 0.005, 0.7]])
        reference = p.log() * factorized.T - torch.tensor([0.0] * 8 + [0.75])
        actual = ExpertFusion(
            1.0, 1.0, 1.0, negative_policy="abstain", require_healthy_for_known=True
        )(reference, torch.tensor([[-100.0, 100.0]]))
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_healthy_boost_preserves_ambiguity_with_a_competing_disease(self):
        p = torch.tensor(
            [[0.006, 0.518, 0.002, 0.326, 0.004, 0.011, 0.098, 0.033, 0.002]]
        )
        reference = p.log() * factorized.T - torch.tensor([0.0] * 8 + [0.75])
        actual = ExpertFusion(1.0, 0.0, 0.0, healthy_disease_margin=0.2)(
            reference, torch.tensor([[100.0, 100.0]])
        )
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_multiclass_lora_tail_starts_as_exact_original_classifier(self):
        from timm.models.vision_transformer import VisionTransformer
        from phase2_multiclass import MulticlassExpert

        model = VisionTransformer(
            img_size=28,
            patch_size=14,
            embed_dim=32,
            depth=5,
            num_heads=4,
            num_classes=9,
        ).eval()
        model.requires_grad_(False)
        image = torch.randn(2, 3, 28, 28)
        for depth in [1, 4]:
            expert = MulticlassExpert(model, "lora", rank=2, depth=depth).eval()
            torch.testing.assert_close(
                expert(prefix(model, image, depth)), model(image)
            )
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
            self.assertTrue(all(p.requires_grad for p in expert.head.parameters()))
            self.assertTrue(all(not p.requires_grad for p in expert.norm.parameters()))

    def test_domain_policy_requires_nine_class_diagnosis_and_corroboration(self):
        from phase2_policy import DomainPolicy

        reference = torch.zeros(1, 9)
        reference[:, 8] = 5
        student = torch.zeros(1, 9)
        student[:, 1] = 10
        policy = DomainPolicy()
        for binary in [[100.0, -100.0], [-100.0, 100.0]]:
            torch.testing.assert_close(
                policy(reference, student, torch.tensor([binary])),
                reference,
                rtol=0,
                atol=0,
            )
        actual = policy(reference, student, torch.tensor([[100.0, 100.0]]))
        self.assertEqual(int(actual.argmax()), 1)

    def test_domain_policy_preserves_matching_settled_disease(self):
        from phase2_policy import DomainPolicy

        reference = torch.zeros(7, 9)
        for row, label in enumerate([0, 2, 3, 4, 5, 6, 7]):
            reference[row, label] = 8
        torch.testing.assert_close(
            DomainPolicy()(
                reference, reference * 2, torch.tensor([[-100.0, 100.0]]).repeat(7, 1)
            ),
            reference,
            rtol=0,
            atol=0,
        )

    def test_healthy_verifier_cannot_invent_healthy_without_multiclass_support(self):
        from phase2_healthy import healthy_override

        reference = torch.zeros(2, 9)
        reference[:, 3] = 7
        student = reference.clone()
        student[1, 3] = 0
        student[1, 1] = 10
        actual = healthy_override(reference, student, torch.full((2,), 100.0), 0.99)
        torch.testing.assert_close(actual[0], reference[0], rtol=0, atol=0)
        self.assertEqual(int(actual[1].argmax()), 1)

    def test_healthy_recovery_can_protect_other_named_conditions(self):
        from phase2_policy import DomainPolicy

        reference = torch.zeros(2, 9)
        reference[0, 3] = 8
        reference[1, 7] = 8
        student = torch.zeros(2, 9)
        student[:, 1] = 10
        policy = DomainPolicy(healthy_reference_classes=[1, 3, 8])
        actual = policy(reference, student, torch.full((2, 2), 100.0))
        self.assertEqual(int(actual[0].argmax()), 1)
        torch.testing.assert_close(actual[1], reference[1], rtol=0, atol=0)

    def test_external_gate_does_not_round_small_sample_allowances_up(self):
        import copy
        from phase2_select import retain

        base = dict(
            healthy=dict(correct=30),
            supported_disease=dict(n=133, correct=28, false_healthy=0),
            all=dict(false_healthy=4),
            supported=dict(n=177, false_other=55),
            unsupported_tomato=dict(n=161, false_known=16),
            non_tomato=dict(n=0, false_known=None),
            disease_classes={
                "Early_blight": dict(n=120, correct=27),
                "Late_blight": dict(n=13, correct=1),
            },
        )
        candidate = copy.deepcopy(base)
        candidate["healthy"]["correct"] += 1
        self.assertTrue(retain({"slif": candidate}, {"slif": base}))
        candidate["supported_disease"]["false_healthy"] += 1
        self.assertFalse(retain({"slif": candidate}, {"slif": base}))
        candidate = copy.deepcopy(base)
        candidate["disease_classes"]["Late_blight"]["correct"] = 0
        self.assertFalse(retain({"slif": candidate}, {"slif": base}))


if __name__ == "__main__":
    unittest.main()
