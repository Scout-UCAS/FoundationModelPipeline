from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fmops.pipeline import load_platform
from fmops.training_runner import TrainingRunner


def build_sbatch_script(command: str, stage: str, nodes: int, gpus_per_node: int, output: Path) -> Path:
    script = output.with_suffix(".sbatch")
    script.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(
        [
            "#!/usr/bin/env bash",
            f"#SBATCH --job-name=fmops-{stage.lower().replace(' ', '-')}",
            f"#SBATCH --nodes={nodes}",
            f"#SBATCH --gres=gpu:{gpus_per_node}",
            "#SBATCH --ntasks-per-node=1",
            "#SBATCH --exclusive",
            f"#SBATCH --account={os.environ.get('FMOPS_CLUSTER_ACCOUNT', 'foundation-model')}",
            "set -euo pipefail",
            "export NCCL_ASYNC_ERROR_HANDLING=1",
            "export TORCH_NCCL_BLOCKING_WAIT=1",
            command,
            "",
        ]
    )
    script.write_text(body, encoding="utf-8")
    return script


def build_payload(config_dir: Path, stage_name: str, scheduler: str, output: Path) -> dict[str, object]:
    platform = load_platform(config_dir)
    runner = TrainingRunner(platform.training)
    run = runner.plan(stage_name)[0]
    sbatch_script = build_sbatch_script(
        run.command,
        stage_name,
        platform.training.hardware.nodes,
        platform.training.hardware.gpus_per_node,
        output,
    )
    return {
        "job": "training_launch",
        "scheduler": scheduler,
        "stage": stage_name,
        "status": "prepared",
        "world_size": run.world_size,
        "command": run.command,
        "environment": run.environment,
        "gates": run.gates,
        "sbatch_script": str(sbatch_script),
        "artifacts": {
            "checkpoint_root": os.environ.get("FMOPS_CHECKPOINT_ROOT", "unset"),
            "container_image": os.environ.get("FMOPS_CONTAINER_IMAGE", "unset"),
        },
    }


def submit(payload: dict[str, object], scheduler: str) -> dict[str, object]:
    if scheduler != "slurm":
        return {"submitted": False, "reason": f"unsupported scheduler {scheduler}"}
    script = str(payload["sbatch_script"])
    completed = subprocess.run(["sbatch", script], check=False, capture_output=True, text=True)
    return {
        "submitted": completed.returncode == 0,
        "exit_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare or submit a production training stage")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--scheduler", default="slurm", choices=["slurm"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload(Path(args.config_dir), args.stage, args.scheduler, output)
    if args.submit:
        payload["submission"] = submit(payload, args.scheduler)
        payload["status"] = "submitted" if payload["submission"]["submitted"] else "submission_failed"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output}")
    return 0 if payload.get("status") != "submission_failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
