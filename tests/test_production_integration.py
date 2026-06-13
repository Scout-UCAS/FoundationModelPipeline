from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fmops.production import ProductionIntegration, REQUIRED_PRODUCTION_AREAS


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


class ProductionIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.integration = ProductionIntegration.from_file(CONFIG_DIR / "production_integration.json")

    def test_production_config_validates_and_covers_required_areas(self) -> None:
        self.assertEqual([], self.integration.validate())
        areas = {task.area for task in self.integration.tasks}
        self.assertTrue(set(REQUIRED_PRODUCTION_AREAS).issubset(areas))
        self.assertGreaterEqual(len(self.integration.release_gates), 8)

    def test_plan_and_preflight_are_serializable(self) -> None:
        plan = self.integration.plan(config_dir=CONFIG_DIR)
        self.assertEqual(14, plan["summary"]["task_count"])
        self.assertIn("training", plan["summary"]["areas"])

        checks = self.integration.preflight(config_dir=CONFIG_DIR)
        self.assertEqual(14, len(checks))
        self.assertTrue(any(check.area == "training" for check in checks))

    def test_write_plan_check_and_non_execute_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            plan_path = self.integration.write_plan(base / "plan.json", config_dir=CONFIG_DIR)
            check_path = self.integration.write_preflight(base / "preflight.json", config_dir=CONFIG_DIR)
            run_path = self.integration.run(base / "run.json", config_dir=CONFIG_DIR, area="monitoring")

            self.assertEqual(14, json.loads(plan_path.read_text())["summary"]["task_count"])
            self.assertEqual(14, json.loads(check_path.read_text())["summary"]["task_count"])
            run_payload = json.loads(run_path.read_text())
            self.assertEqual("plan", run_payload["mode"])
            self.assertEqual(1, run_payload["summary"]["planned"])


if __name__ == "__main__":
    unittest.main()
