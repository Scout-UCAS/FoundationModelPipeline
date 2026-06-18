from __future__ import annotations

import unittest
from pathlib import Path

from fmops.architectures import DEFAULT_REQUIRED_FAMILIES
from fmops.evaluation import DEFAULT_REQUIRED_DIMENSIONS
from fmops.pipeline import load_platform
from fmops.training import REQUIRED_TRAINING_STAGES


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


class PlatformConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.platform = load_platform(CONFIG_DIR)

    def test_configs_validate(self) -> None:
        self.assertEqual([], self.platform.all_issues())

    def test_data_system_covers_requested_scale_and_modalities(self) -> None:
        summary = self.platform.data.summary()
        self.assertGreaterEqual(summary["actual_tb"], 2500)
        self.assertGreaterEqual(summary["language_count"], 20)
        self.assertEqual(
            {"pure_text", "multimodal", "video_pretraining", "audio_speech", "vla"},
            set(summary["modalities"]),
        )

    def test_architecture_suite_covers_all_required_families(self) -> None:
        families = {candidate.family for candidate in self.platform.architectures.candidates}
        self.assertTrue(set(DEFAULT_REQUIRED_FAMILIES).issubset(families))
        comparison = self.platform.architectures.comparison_table()
        self.assertEqual(len(self.platform.architectures.candidates), len(comparison))
        self.assertGreaterEqual(comparison[0]["utility_score"], comparison[-1]["utility_score"])

    def test_training_pipeline_is_four_hundred_gpu_end_to_end(self) -> None:
        self.assertEqual(400, self.platform.training.hardware.total_gpus)
        stage_names = {stage.name for stage in self.platform.training.stages}
        self.assertTrue(set(REQUIRED_TRAINING_STAGES).issubset(stage_names))
        self.assertEqual(4, len(self.platform.training.launch_commands()))

    def test_evaluation_suite_covers_required_dimensions(self) -> None:
        dimensions = {benchmark.dimension for benchmark in self.platform.evaluation.benchmarks}
        self.assertTrue(set(DEFAULT_REQUIRED_DIMENSIONS).issubset(dimensions))
        weights = self.platform.evaluation.dimension_weights()
        self.assertAlmostEqual(1.0, sum(weights.values()), places=3)


if __name__ == "__main__":
    unittest.main()
