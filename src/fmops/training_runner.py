from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .training import TrainingPipeline, TrainingStage


@dataclass(frozen=True)
class TrainingRun:
    stage: str
    command: str
    status: str
    world_size: int
    environment: dict[str, str]
    gates: dict[str, Any]


class TrainingRunner:
    def __init__(self, pipeline: TrainingPipeline) -> None:
        self.pipeline = pipeline

    def plan_stage(self, stage: TrainingStage) -> TrainingRun:
        command = stage.launcher.format(
            nodes=self.pipeline.hardware.nodes,
            gpus_per_node=self.pipeline.hardware.gpus_per_node,
            total_gpus=self.pipeline.hardware.total_gpus,
            data_mixture=stage.data_mixture,
        )
        env = {
            "WORLD_SIZE": str(self.pipeline.hardware.total_gpus),
            "N_NODES": str(self.pipeline.hardware.nodes),
            "GPUS_PER_NODE": str(self.pipeline.hardware.gpus_per_node),
            "FMOPS_STAGE": stage.name,
            "FMOPS_DATA_MIXTURE": stage.data_mixture,
        }
        return TrainingRun(stage.name, command, "planned", self.pipeline.hardware.total_gpus, env, stage.gates)

    def plan(self, stage_name: str | None = None) -> list[TrainingRun]:
        stages = self.pipeline.stages
        if stage_name:
            stages = [stage for stage in stages if stage.name == stage_name]
            if not stages:
                raise ValueError(f"unknown training stage: {stage_name}")
        return [self.plan_stage(stage) for stage in stages]

    def dry_run(self, output: str | Path, stage_name: str | None = None) -> Path:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "hardware": {
                "nodes": self.pipeline.hardware.nodes,
                "gpus_per_node": self.pipeline.hardware.gpus_per_node,
                "total_gpus": self.pipeline.hardware.total_gpus,
                "gpu_type": self.pipeline.hardware.gpu_type,
            },
            "runs": [asdict(item) for item in self.plan(stage_name)],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path


def run_stage_from_config(config_dir: str | Path, stage_name: str, output: str | Path, dry_run: bool = True) -> Path:
    from .pipeline import load_platform

    pipeline = load_platform(config_dir).training
    runner = TrainingRunner(pipeline)
    if not dry_run:
        raise NotImplementedError("use train/* --mode external or fmops production-run for production training dispatch")
    return runner.dry_run(output, stage_name=stage_name)
