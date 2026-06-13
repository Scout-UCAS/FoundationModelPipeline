from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from fmops.training_entrypoint import run_stage_entrypoint


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


class TrainingEntrypointTest(unittest.TestCase):
    def test_dry_run_writes_stage_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "pretrain.json"
            code = run_stage_entrypoint(
                config_dir=CONFIG_DIR,
                stage_name="Pre-training",
                mixture="core_pretrain",
                world_size=400,
                output=output,
                mode="dry-run",
            )
            payload = json.loads(output.read_text())
            self.assertEqual(0, code)
            self.assertEqual("Pre-training", payload["runs"][0]["stage"])
            self.assertEqual(400, payload["runs"][0]["world_size"])

    def test_external_mode_blocks_without_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "blocked.json"
            code = run_stage_entrypoint(
                config_dir=CONFIG_DIR,
                stage_name="Agentic RL",
                mixture="agentic_env_rollouts",
                world_size=400,
                output=output,
                mode="external",
            )
            payload = json.loads(output.read_text())
            self.assertEqual(2, code)
            self.assertEqual("blocked", payload["status"])

    def test_external_mode_dispatches_backend_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "sft.json"
            command = f"{sys.executable} -c \"import os; assert os.environ['FMOPS_STAGE'] == 'SFT'\""
            code = run_stage_entrypoint(
                config_dir=CONFIG_DIR,
                stage_name="SFT",
                mixture="reasoning_final",
                world_size=400,
                output=output,
                mode="external",
                backend_command=command,
            )
            payload = json.loads(output.read_text())
            self.assertEqual(0, code)
            self.assertEqual("succeeded", payload["status"])
            self.assertEqual("argument", payload["command_source"])

    def test_native_mode_trains_all_stages_and_writes_checkpoints(self) -> None:
        for stage, mixture in (
            ("Pre-training", "core_pretrain"),
            ("SFT", "reasoning_final"),
            ("RL", "rl_reasoning_tool_mix"),
            ("Agentic RL", "agentic_env_rollouts"),
        ):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temp_dir:
                base = Path(temp_dir)
                output = base / "result.json"
                code = run_stage_entrypoint(
                    config_dir=CONFIG_DIR,
                    stage_name=stage,
                    mixture=mixture,
                    world_size=1,
                    output=output,
                    mode="native",
                    native_options={
                        "output_dir": str(base / "native"),
                        "max_steps": 1,
                        "batch_size": 1 if stage in {"Pre-training", "SFT"} else 2,
                        "seq_len": 16,
                        "d_model": 32,
                        "n_heads": 4,
                        "n_layers": 1,
                        "d_ff": 64,
                        "device": "cpu",
                        "use_ddp": False,
                    },
                )
                payload = json.loads(output.read_text())
                self.assertEqual(0, code)
                self.assertEqual("succeeded", payload["status"])
                self.assertTrue(Path(payload["checkpoint_path"]).exists())
                self.assertTrue(Path(payload["metrics_path"]).exists())
                self.assertEqual(1, payload["steps"])


if __name__ == "__main__":
    unittest.main()
