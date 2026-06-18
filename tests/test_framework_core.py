from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fmops.checkpoint import CheckpointConverter, load_checkpoint_manifest
from fmops.dashboard import DashboardBuilder
from fmops.data_pipeline import DataPipelineRunner
from fmops.dataset_catalog import DatasetCatalog
from fmops.deployment import DeploymentValidator
from fmops.evaluation_runner import EvaluationRunner
from fmops.pipeline import load_platform
from fmops.plugins import PluginManager
from fmops.registry import MODEL_REGISTRY
from fmops.schema import validate_config_dir
from fmops.tracking import ExperimentTracker
from fmops.training_runner import TrainingRunner


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


class FrameworkCoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.platform = load_platform(CONFIG_DIR)
        self.catalog = DatasetCatalog.from_file(CONFIG_DIR / "datasets_catalog.json")

    def test_schema_and_dataset_catalog_validate(self) -> None:
        schema_validation = validate_config_dir(CONFIG_DIR)
        self.assertEqual([], [issue for issues in schema_validation.values() for issue in issues])
        self.assertEqual([], self.catalog.validate())
        summary = self.catalog.summary()
        self.assertGreaterEqual(summary["dataset_count"], 10)
        self.assertIn("VLA-robotics", summary["families"])

    def test_registry_exposes_reference_models(self) -> None:
        names = MODEL_REGISTRY.names()
        self.assertIn("MoE", names)
        self.assertIn("Reasoning-native Architecture", names)

    def test_data_training_eval_and_deploy_runners_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            data_path = DataPipelineRunner(self.platform.data, self.catalog).run_dry(base / "data.json")
            train_path = TrainingRunner(self.platform.training).dry_run(base / "train.json", stage_name="SFT")
            eval_path = EvaluationRunner(self.platform.evaluation).write_report(base / "eval.json", model_id="unit")
            deploy_path = DeploymentValidator.default().write_report(base / "deploy.json")

            self.assertTrue(data_path.exists())
            self.assertEqual(1, len(json.loads(train_path.read_text())["runs"]))
            self.assertEqual(9, json.loads(eval_path.read_text())["summary"]["benchmark_count"])
            self.assertEqual(3, json.loads(deploy_path.read_text())["summary"]["target_count"])

    def test_checkpoint_tracking_dashboard_and_plugins(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            source = base / "checkpoint"
            source.mkdir()
            (source / "model.bin").write_bytes(b"weights")
            manifest = CheckpointConverter().convert(source, base / "converted")
            loaded = load_checkpoint_manifest(base / "converted" / "checkpoint_manifest.json")
            self.assertEqual(manifest.checkpoint_id, loaded.checkpoint_id)

            tracker = ExperimentTracker(base / "runs")
            run = tracker.complete_run(tracker.start_run("unit", "test"), metrics={"loss": 1.0})
            self.assertEqual("completed", run.status)
            self.assertEqual(1, len(tracker.list_runs()))

            dashboard = DashboardBuilder(self.platform).write(base / "dashboard.html")
            self.assertIn("Foundation Model Ops Dashboard", dashboard.read_text())

            self.assertEqual([], PluginManager(base / "plugins").discover())
            self.assertEqual([], PluginManager(base / "plugins").validate())


if __name__ == "__main__":
    unittest.main()
