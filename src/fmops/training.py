from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_json, require_keys


REQUIRED_TRAINING_STAGES = ("Pre-training", "SFT", "RL", "Agentic RL")


@dataclass(frozen=True)
class HardwareSpec:
    nodes: int
    gpus_per_node: int
    gpu_type: str
    interconnect: str
    storage: str

    @property
    def total_gpus(self) -> int:
        return self.nodes * self.gpus_per_node

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HardwareSpec":
        return cls(
            nodes=int(payload["nodes"]),
            gpus_per_node=int(payload["gpus_per_node"]),
            gpu_type=str(payload.get("gpu_type", "unknown")),
            interconnect=str(payload.get("interconnect", "unknown")),
            storage=str(payload.get("storage", "unknown")),
        )


@dataclass(frozen=True)
class TrainingStage:
    name: str
    objective: str
    data_mixture: str
    framework: str
    launcher: str
    parallelism: dict[str, int]
    monitors: tuple[str, ...]
    artifacts: tuple[str, ...]
    gates: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingStage":
        return cls(
            name=str(payload["name"]),
            objective=str(payload.get("objective", "")),
            data_mixture=str(payload.get("data_mixture", "")),
            framework=str(payload.get("framework", "")),
            launcher=str(payload.get("launcher", "")),
            parallelism={str(key): int(value) for key, value in payload.get("parallelism", {}).items()},
            monitors=tuple(str(item) for item in payload.get("monitors", [])),
            artifacts=tuple(str(item) for item in payload.get("artifacts", [])),
            gates=dict(payload.get("gates", {})),
        )


class TrainingPipeline:
    def __init__(
        self,
        target_gpus: int,
        hardware: HardwareSpec,
        stages: list[TrainingStage],
        integration: dict[str, Any],
    ) -> None:
        self.target_gpus = target_gpus
        self.hardware = hardware
        self.stages = stages
        self.integration = integration

    @classmethod
    def from_file(cls, path: str | Path) -> "TrainingPipeline":
        payload = load_json(path)
        hardware = HardwareSpec.from_dict(payload.get("hardware", {}))
        stages = [TrainingStage.from_dict(item) for item in payload.get("stages", [])]
        return cls(
            target_gpus=int(payload.get("target_gpus", 400)),
            hardware=hardware,
            stages=stages,
            integration=payload.get("integration", {}),
        )

    def validate(self) -> list[str]:
        issues: list[str] = []
        if self.hardware.total_gpus != self.target_gpus:
            issues.append(
                f"training.hardware: expected {self.target_gpus} GPUs, got {self.hardware.total_gpus} "
                f"({self.hardware.nodes}x{self.hardware.gpus_per_node})"
            )

        stage_names = {stage.name for stage in self.stages}
        for required in REQUIRED_TRAINING_STAGES:
            if required not in stage_names:
                issues.append(f"training.stages: missing required stage '{required}'")

        for index, stage in enumerate(self.stages):
            context = f"training.stages[{index}].{stage.name}"
            issues.extend(require_keys(stage.parallelism, ["tensor", "pipeline", "data"], f"{context}.parallelism"))
            if not stage.data_mixture:
                issues.append(f"{context}: data_mixture must not be empty")
            if not stage.launcher:
                issues.append(f"{context}: launcher must not be empty")
            if not stage.monitors:
                issues.append(f"{context}: monitors must not be empty")
            if not stage.artifacts:
                issues.append(f"{context}: artifacts must not be empty")

        integration_text = " ".join(
            str(item).lower()
            for value in self.integration.values()
            for item in (value if isinstance(value, list) else [value])
        )
        for required in ("data", "distributed", "stability", "conversion", "deployment"):
            if required not in integration_text:
                issues.append(f"training.integration: missing {required} flow")

        return issues

    def launch_commands(self) -> list[dict[str, str]]:
        commands: list[dict[str, str]] = []
        for stage in self.stages:
            commands.append(
                {
                    "stage": stage.name,
                    "command": stage.launcher.format(
                        nodes=self.hardware.nodes,
                        gpus_per_node=self.hardware.gpus_per_node,
                        total_gpus=self.hardware.total_gpus,
                        data_mixture=stage.data_mixture,
                    ),
                }
            )
        return commands

    @staticmethod
    def _format_gate(name: str, value: Any) -> str:
        if name.startswith("min_"):
            return f"{name}>={value}"
        if name.startswith("max_"):
            return f"{name}<={value}"
        return f"{name}={value}"

    def to_markdown(self) -> str:
        lines = [
            "## Training Pipeline",
            "",
            f"- Hardware: {self.hardware.nodes} nodes x {self.hardware.gpus_per_node} {self.hardware.gpu_type} GPUs = {self.hardware.total_gpus} GPUs",
            f"- Interconnect: {self.hardware.interconnect}",
            f"- Storage: {self.hardware.storage}",
            "",
            "| Stage | Objective | Data | Framework | Parallelism | Gates |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for stage in self.stages:
            parallelism = ", ".join(f"{key}={value}" for key, value in sorted(stage.parallelism.items()))
            gates = ", ".join(self._format_gate(key, value) for key, value in sorted(stage.gates.items()))
            lines.append(
                f"| {stage.name} | {stage.objective} | {stage.data_mixture} | {stage.framework} | {parallelism} | {gates} |"
            )

        lines.extend(["", "| Stage | Launch Command |", "| --- | --- |"])
        for item in self.launch_commands():
            lines.append(f"| {item['stage']} | `{item['command']}` |")
        return "\n".join(lines)
