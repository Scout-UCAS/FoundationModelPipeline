from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from .training_runner import run_stage_from_config


STAGE_BACKEND_ENV = {
    "Pre-training": "FMOPS_PRETRAIN_BACKEND_COMMAND",
    "SFT": "FMOPS_SFT_BACKEND_COMMAND",
    "RL": "FMOPS_RL_BACKEND_COMMAND",
    "Agentic RL": "FMOPS_AGENTIC_RL_BACKEND_COMMAND",
}


class SafeFormatDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _is_rank_zero() -> bool:
    return os.environ.get("RANK", "0") == "0"


def _write_manifest(output: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if _is_rank_zero():
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _backend_command(stage_name: str, explicit_command: str | None) -> tuple[str | None, str]:
    if explicit_command:
        return explicit_command, "argument"
    env_name = STAGE_BACKEND_ENV[stage_name]
    value = os.environ.get(env_name) or os.environ.get("FMOPS_TRAINING_BACKEND_COMMAND")
    return value, env_name if os.environ.get(env_name) else "FMOPS_TRAINING_BACKEND_COMMAND"


def run_stage_entrypoint(
    *,
    config_dir: str | Path,
    stage_name: str,
    mixture: str,
    world_size: int,
    output: str | Path,
    mode: str = "dry-run",
    backend_command: str | None = None,
    native_options: dict[str, Any] | None = None,
) -> int:
    if mode == "dry-run":
        run_stage_from_config(config_dir, stage_name, output, dry_run=True)
        return 0

    if mode == "native":
        from .native_training import native_config_from_options, run_native_training

        options = dict(native_options or {})
        options.setdefault("output", str(output))
        result = run_native_training(native_config_from_options(stage_name, mixture, options))
        _write_manifest(output, result.__dict__)
        return 0

    if mode != "external":
        raise ValueError(f"unsupported training entrypoint mode: {mode}")

    command_template, command_source = _backend_command(stage_name, backend_command)
    context = SafeFormatDict(
        {
            "config_dir": str(config_dir),
            "stage": stage_name,
            "mixture": mixture,
            "world_size": world_size,
            "output": str(output),
            "rank": os.environ.get("RANK", "0"),
            "local_rank": os.environ.get("LOCAL_RANK", "0"),
        }
    )
    if not command_template:
        _write_manifest(
            output,
            {
                "stage": stage_name,
                "mode": mode,
                "status": "blocked",
                "message": (
                    f"set {STAGE_BACKEND_ENV[stage_name]} or FMOPS_TRAINING_BACKEND_COMMAND "
                    "to dispatch to the real trainer"
                ),
                "mixture": mixture,
                "world_size": world_size,
            },
        )
        return 2

    command = command_template.format_map(context)
    env = os.environ.copy()
    env.update(
        {
            "FMOPS_STAGE": stage_name,
            "FMOPS_DATA_MIXTURE": mixture,
            "FMOPS_WORLD_SIZE": str(world_size),
            "FMOPS_CONFIG_DIR": str(config_dir),
        }
    )
    started = time.monotonic()
    completed = subprocess.run(command, shell=True, env=env, check=False)
    duration = round(time.monotonic() - started, 3)
    _write_manifest(
        output,
        {
            "stage": stage_name,
            "mode": mode,
            "status": "succeeded" if completed.returncode == 0 else "failed",
            "command": command,
            "command_source": command_source,
            "exit_code": completed.returncode,
            "duration_seconds": duration,
            "mixture": mixture,
            "world_size": world_size,
        },
    )
    return completed.returncode
