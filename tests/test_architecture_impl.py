from __future__ import annotations

import unittest

import torch

from fmops.architecture_impl import (
    DiscreteDiffusionLanguageModel,
    HybridArchitectureModel,
    LatentReasoningModel,
    MemoryAugmentedLM,
    MoETransformerBlock,
    ModelConfig,
    MultiTokenPredictionModel,
    OmniModalArchitecture,
    RNNBackboneBlock,
    ReasoningNativeArchitecture,
    SparseLinearAttentionBlock,
    build_reference_implementation,
)


class ArchitectureImplementationTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.config = ModelConfig(
            vocab_size=32,
            d_model=16,
            n_heads=4,
            n_layers=2,
            d_ff=32,
            max_seq_len=32,
            dropout=0.0,
        )
        self.input_ids = torch.randint(0, self.config.vocab_size, (2, 8))

    def test_moe_transformer_block(self) -> None:
        block = MoETransformerBlock(self.config, num_experts=4, top_k=2)
        x = torch.randn(2, 8, self.config.d_model)
        y, aux_loss = block(x)
        self.assertEqual(x.shape, y.shape)
        self.assertEqual(0, aux_loss.ndim)

    def test_sparse_linear_attention_block(self) -> None:
        block = SparseLinearAttentionBlock(self.config, window_size=4)
        x = torch.randn(2, 8, self.config.d_model)
        y = block(x)
        self.assertEqual(x.shape, y.shape)

    def test_rnn_like_backbone_block(self) -> None:
        block = RNNBackboneBlock(self.config)
        x = torch.randn(2, 8, self.config.d_model)
        y, state = block(x)
        self.assertEqual(x.shape, y.shape)
        self.assertEqual((2, self.config.d_model), state.shape)

    def test_hybrid_architecture_model(self) -> None:
        model = HybridArchitectureModel(self.config, attention_every=2)
        output = model(self.input_ids, labels=self.input_ids)
        self.assertEqual((2, 8, self.config.vocab_size), output["logits"].shape)
        self.assertIn("loss", output)

    def test_multi_token_prediction_model(self) -> None:
        model = MultiTokenPredictionModel(self.config, prediction_offsets=3)
        output = model(self.input_ids, labels=self.input_ids)
        self.assertEqual((2, 8, self.config.vocab_size), output["logits"].shape)
        self.assertEqual({1, 2, 3}, set(output["mtp_logits"]))
        self.assertIn("loss", output)

    def test_latent_reasoning_model(self) -> None:
        model = LatentReasoningModel(self.config, num_latents=3)
        output = model(self.input_ids, labels=self.input_ids, latent_insert_index=4)
        self.assertEqual((2, 8, self.config.vocab_size), output["logits"].shape)
        self.assertEqual((2, 3, self.config.d_model), output["latent_hidden"].shape)
        self.assertIn("loss", output)

    def test_discrete_diffusion_language_model(self) -> None:
        model = DiscreteDiffusionLanguageModel(self.config, num_diffusion_steps=8)
        timesteps = torch.full((2,), 7, dtype=torch.long)
        output = model.training_step(self.input_ids, timesteps=timesteps)
        self.assertEqual((2, 8, self.config.vocab_size), output["logits"].shape)
        self.assertTrue(output["loss_mask"].all())
        self.assertIn("loss", output)

    def test_memory_augmented_lm(self) -> None:
        model = MemoryAugmentedLM(self.config, memory_slots=5)
        external_memory = torch.randn(2, 5, self.config.d_model)
        output = model(self.input_ids, labels=self.input_ids, external_memory=external_memory)
        self.assertEqual((2, 8, self.config.vocab_size), output["logits"].shape)
        self.assertEqual(self.config.n_layers, len(output["memory_weights"]))
        self.assertIn("loss", output)

    def test_omni_modal_architecture(self) -> None:
        model = OmniModalArchitecture(
            self.config,
            image_dim=6,
            video_dim=7,
            audio_dim=5,
            action_vocab_size=11,
        )
        output = model(
            input_ids=self.input_ids[:, :3],
            image_features=torch.randn(2, 2, 6),
            video_features=torch.randn(2, 2, 7),
            audio_features=torch.randn(2, 1, 5),
            action_ids=torch.randint(0, 11, (2, 2)),
        )
        self.assertEqual((2, 10, self.config.d_model), output["hidden"].shape)
        self.assertEqual((2, 10, self.config.vocab_size), output["text_logits"].shape)
        self.assertEqual((2, 10, 11), output["action_logits"].shape)

    def test_reasoning_native_architecture(self) -> None:
        model = ReasoningNativeArchitecture(self.config, num_plan_tokens=9)
        process_labels = torch.ones(2, 8)
        plan_labels = torch.randint(0, 9, (2, 8))
        output = model(
            self.input_ids,
            labels=self.input_ids,
            process_labels=process_labels,
            plan_labels=plan_labels,
        )
        self.assertEqual((2, 8, self.config.vocab_size), output["logits"].shape)
        self.assertEqual((2, 8), output["verifier_logits"].shape)
        self.assertEqual((2, 8, 9), output["planner_logits"].shape)
        self.assertIn("loss", output)

    def test_registry_builds_all_requested_families(self) -> None:
        families = [
            "MoE",
            "Sparse / Linear Attention",
            "RNN-like Backbone",
            "Hybrid Architecture",
            "MTP",
            "Latent Reasoning",
            "dLLM",
            "Memory-augmented LLM",
            "Omni-modal Architecture",
            "Reasoning-native Architecture",
        ]
        for family in families:
            with self.subTest(family=family):
                module = build_reference_implementation(family, self.config)
                self.assertIsInstance(module, torch.nn.Module)


if __name__ == "__main__":
    unittest.main()

