from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_json, require_keys


DEFAULT_REQUIRED_FAMILIES = (
    "MoE",
    "Sparse / Linear Attention",
    "RNN-like Backbone",
    "SSM / Selective Scan",
    "Retention / RetNet",
    "Long Convolution",
    "MLA / KV-Compressed Attention",
    "Hybrid Architecture",
    "MTP",
    "Latent Reasoning",
    "dLLM",
    "Memory-augmented LLM",
    "Mixture-of-Depths",
    "Test-Time Memory",
    "Token-free Byte-level LLM",
    "Omni-modal Architecture",
    "VLA / Robotics Transformer",
    "JEPA / Latent World Model",
    "Neuromorphic / Spiking Backbone",
    "Reasoning-native Architecture",
)


@dataclass(frozen=True)
class ArchitectureCandidate:
    name: str
    family: str
    hypothesis: str
    total_params_b: float
    active_params_b: float
    components: dict[str, Any]
    metrics: dict[str, float]
    risks: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ArchitectureCandidate":
        return cls(
            name=str(payload["name"]),
            family=str(payload["family"]),
            hypothesis=str(payload.get("hypothesis", "")),
            total_params_b=float(payload.get("total_params_b", 0.0)),
            active_params_b=float(payload.get("active_params_b", 0.0)),
            components=dict(payload.get("components", {})),
            metrics={str(key): float(value) for key, value in payload.get("metrics", {}).items()},
            risks=tuple(str(item) for item in payload.get("risks", [])),
        )


class ArchitectureSuite:
    def __init__(
        self,
        unified_setup: dict[str, Any],
        required_families: tuple[str, ...],
        candidates: list[ArchitectureCandidate],
    ) -> None:
        self.unified_setup = unified_setup
        self.required_families = required_families
        self.candidates = candidates

    @classmethod
    def from_file(cls, path: str | Path) -> "ArchitectureSuite":
        payload = load_json(path)
        candidates = [ArchitectureCandidate.from_dict(item) for item in payload.get("candidates", [])]
        required = tuple(str(item) for item in payload.get("required_families", DEFAULT_REQUIRED_FAMILIES))
        return cls(unified_setup=payload.get("unified_setup", {}), required_families=required, candidates=candidates)

    def validate(self) -> list[str]:
        issues: list[str] = []
        issues.extend(
            require_keys(
                self.unified_setup,
                ["tokenizer", "training_tokens_billion", "context_length", "optimizer", "hardware_budget_gpus"],
                "architecture.unified_setup",
            )
        )
        families = {candidate.family for candidate in self.candidates}
        for family in self.required_families:
            if family not in families:
                issues.append(f"architecture.candidates: missing required family '{family}'")

        required_metrics = {
            "validation_loss",
            "tokens_per_second_per_gpu",
            "reasoning_score",
            "memory_gb_per_gpu",
            "stability_score",
        }
        for index, candidate in enumerate(self.candidates):
            context = f"architecture.candidates[{index}].{candidate.name}"
            if candidate.total_params_b <= 0:
                issues.append(f"{context}: total_params_b must be positive")
            if candidate.active_params_b <= 0:
                issues.append(f"{context}: active_params_b must be positive")
            missing_metrics = required_metrics - set(candidate.metrics)
            if missing_metrics:
                issues.append(f"{context}: missing metrics {sorted(missing_metrics)}")
        return issues

    def comparison_table(self) -> list[dict[str, Any]]:
        if not self.candidates:
            return []

        metric_names = [
            "validation_loss",
            "tokens_per_second_per_gpu",
            "reasoning_score",
            "memory_gb_per_gpu",
            "stability_score",
        ]
        values = {
            metric: [candidate.metrics.get(metric, 0.0) for candidate in self.candidates]
            for metric in metric_names
        }

        def normalize(metric: str, value: float, higher_is_better: bool = True) -> float:
            items = values[metric]
            low = min(items)
            high = max(items)
            if high == low:
                return 1.0
            scaled = (value - low) / (high - low)
            return scaled if higher_is_better else 1.0 - scaled

        rows: list[dict[str, Any]] = []
        for candidate in self.candidates:
            metrics = candidate.metrics
            utility = (
                0.25 * normalize("validation_loss", metrics["validation_loss"], higher_is_better=False)
                + 0.20 * normalize("tokens_per_second_per_gpu", metrics["tokens_per_second_per_gpu"])
                + 0.25 * normalize("reasoning_score", metrics["reasoning_score"])
                + 0.15 * normalize("memory_gb_per_gpu", metrics["memory_gb_per_gpu"], higher_is_better=False)
                + 0.15 * normalize("stability_score", metrics["stability_score"])
            )
            rows.append(
                {
                    "name": candidate.name,
                    "family": candidate.family,
                    "total_params_b": candidate.total_params_b,
                    "active_params_b": candidate.active_params_b,
                    "validation_loss": metrics["validation_loss"],
                    "tokens_per_second_per_gpu": metrics["tokens_per_second_per_gpu"],
                    "reasoning_score": metrics["reasoning_score"],
                    "memory_gb_per_gpu": metrics["memory_gb_per_gpu"],
                    "stability_score": metrics["stability_score"],
                    "utility_score": round(utility, 4),
                }
            )
        return sorted(rows, key=lambda row: row["utility_score"], reverse=True)

    def to_markdown(self) -> str:
        setup = self.unified_setup
        lines = [
            "## Architecture Experiments",
            "",
            f"- Unified tokenizer: {setup.get('tokenizer', 'unknown')}",
            f"- Training budget: {setup.get('training_tokens_billion', 'unknown')}B tokens",
            f"- Context length: {setup.get('context_length', 'unknown')}",
            f"- Hardware budget: {setup.get('hardware_budget_gpus', 'unknown')} GPUs",
            "",
            "| Rank | Candidate | Family | Active/Total Params | Loss | Tok/s/GPU | Reasoning | Memory GB | Stability | Utility |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for rank, row in enumerate(self.comparison_table(), start=1):
            params = f"{row['active_params_b']:.1f}B/{row['total_params_b']:.1f}B"
            lines.append(
                "| "
                f"{rank} | {row['name']} | {row['family']} | {params} | "
                f"{row['validation_loss']:.3f} | {row['tokens_per_second_per_gpu']:.0f} | "
                f"{row['reasoning_score']:.1f} | {row['memory_gb_per_gpu']:.1f} | "
                f"{row['stability_score']:.2f} | {row['utility_score']:.3f} |"
            )
        return "\n".join(lines)
