from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_json


DEFAULT_REQUIRED_DIMENSIONS = (
    "general",
    "reasoning",
    "multimodal_understanding",
    "vla",
    "long_context",
    "tool_calling",
    "agent",
    "edge_efficiency",
)


@dataclass(frozen=True)
class Benchmark:
    name: str
    dimension: str
    datasets: tuple[str, ...]
    metrics: tuple[str, ...]
    weight: float
    cadence: str
    gate: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Benchmark":
        return cls(
            name=str(payload["name"]),
            dimension=str(payload["dimension"]),
            datasets=tuple(str(item) for item in payload.get("datasets", [])),
            metrics=tuple(str(item) for item in payload.get("metrics", [])),
            weight=float(payload.get("weight", 1.0)),
            cadence=str(payload.get("cadence", "per-release")),
            gate=dict(payload.get("gate", {})),
        )


class EvaluationSuite:
    def __init__(self, required_dimensions: tuple[str, ...], benchmarks: list[Benchmark]) -> None:
        self.required_dimensions = required_dimensions
        self.benchmarks = benchmarks

    @classmethod
    def from_file(cls, path: str | Path) -> "EvaluationSuite":
        payload = load_json(path)
        required = tuple(str(item) for item in payload.get("required_dimensions", DEFAULT_REQUIRED_DIMENSIONS))
        benchmarks = [Benchmark.from_dict(item) for item in payload.get("benchmarks", [])]
        return cls(required_dimensions=required, benchmarks=benchmarks)

    def validate(self) -> list[str]:
        issues: list[str] = []
        covered = {benchmark.dimension for benchmark in self.benchmarks}
        for dimension in self.required_dimensions:
            if dimension not in covered:
                issues.append(f"evaluation.benchmarks: missing dimension '{dimension}'")

        for index, benchmark in enumerate(self.benchmarks):
            context = f"evaluation.benchmarks[{index}].{benchmark.name}"
            if not benchmark.datasets:
                issues.append(f"{context}: datasets must not be empty")
            if not benchmark.metrics:
                issues.append(f"{context}: metrics must not be empty")
            if benchmark.weight <= 0:
                issues.append(f"{context}: weight must be positive")
            if not benchmark.gate:
                issues.append(f"{context}: gate must not be empty")
        return issues

    def dimension_weights(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for benchmark in self.benchmarks:
            totals[benchmark.dimension] = totals.get(benchmark.dimension, 0.0) + benchmark.weight
        total_weight = sum(totals.values())
        if total_weight == 0:
            return totals
        return {dimension: round(weight / total_weight, 4) for dimension, weight in sorted(totals.items())}

    @staticmethod
    def _format_gate(name: str, value: Any) -> str:
        if name.startswith("min_"):
            return f"{name}>={value}"
        if name.startswith("max_"):
            return f"{name}<={value}"
        return f"{name}>={value}"

    def to_markdown(self) -> str:
        lines = [
            "## Evaluation Suite",
            "",
            "| Dimension | Benchmark | Datasets | Metrics | Weight | Gate | Cadence |",
            "| --- | --- | --- | --- | ---: | --- | --- |",
        ]
        for benchmark in self.benchmarks:
            datasets = ", ".join(benchmark.datasets)
            metrics = ", ".join(benchmark.metrics)
            gates = ", ".join(self._format_gate(key, value) for key, value in sorted(benchmark.gate.items()))
            lines.append(
                f"| {benchmark.dimension} | {benchmark.name} | {datasets} | {metrics} | "
                f"{benchmark.weight:.2f} | {gates} | {benchmark.cadence} |"
            )

        lines.extend(["", "| Dimension | Normalized Weight |", "| --- | ---: |"])
        for dimension, weight in self.dimension_weights().items():
            lines.append(f"| {dimension} | {weight:.4f} |")
        return "\n".join(lines)
