from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fmops.benchmark_catalog import BenchmarkCatalog
from fmops.evaluation_runner import EvaluationRunner
from fmops.pipeline import load_platform


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


class EvaluationRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.platform = load_platform(CONFIG_DIR)

    def test_benchmark_catalog_is_comprehensive_and_valid(self) -> None:
        catalog = BenchmarkCatalog.from_file(CONFIG_DIR / "benchmark_catalog.json")
        self.assertEqual([], catalog.validate())
        summary = catalog.summary()
        self.assertGreaterEqual(summary["benchmark_count"], 80)
        for dimension in self.platform.evaluation.required_dimensions:
            self.assertIn(dimension, summary["dimensions"])
        for modality in ("text", "image", "video", "audio", "action"):
            self.assertIn(modality, summary["modalities"])
        self.assertIn("local-jsonl", summary["harnesses"])

    def test_default_runner_executes_real_smoke_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "eval.json"
            path = EvaluationRunner(self.platform.evaluation).write_report(output, model_id="unit-smoke")
            payload = json.loads(path.read_text())
            self.assertEqual(9, payload["summary"]["benchmark_count"])
            self.assertEqual(9, payload["summary"]["sample_count"])
            self.assertEqual(9, payload["summary"]["passed"])
            transcripts = output.parent / "eval_transcripts"
            self.assertTrue((transcripts / "general_multilingual_core.jsonl").exists())

    def test_prediction_file_and_filter_are_used(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            samples_dir = base / "samples"
            samples_dir.mkdir()
            (samples_dir / "general_multilingual_core.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "sample-1",
                                "benchmark": "general_multilingual_core",
                                "dataset": "MMLU",
                                "prompt": "2 + 2?",
                                "answer": "4",
                                "prediction": "wrong",
                                "confidence": 0.5,
                            }
                        ),
                        json.dumps(
                            {
                                "id": "sample-2",
                                "benchmark": "general_multilingual_core",
                                "dataset": "CMMLU",
                                "prompt": "Capital of France?",
                                "answer": "Paris",
                                "prediction": "wrong",
                                "confidence": 0.5,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            predictions = base / "predictions.jsonl"
            predictions.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "sample-1", "prediction": "4"}),
                        json.dumps({"id": "sample-2", "prediction": "Paris"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            output = base / "report.json"
            EvaluationRunner(
                self.platform.evaluation,
                samples_dir=samples_dir,
                predictions_path=predictions,
                benchmark_filter={"general_multilingual_core"},
            ).write_report(output, model_id="prediction-file")
            payload = json.loads(output.read_text())
            self.assertEqual(1, payload["summary"]["benchmark_count"])
            result = payload["results"][0]
            self.assertEqual(2, result["sample_count"])
            self.assertEqual(1.0, result["metrics"]["accuracy"])
            self.assertEqual(1.0, result["metrics"]["macro_average"])


if __name__ == "__main__":
    unittest.main()
