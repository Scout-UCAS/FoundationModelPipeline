from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DeploymentTarget:
    name: str
    runtime: str
    max_latency_ms: float
    max_memory_gb: float
    min_decode_tokens_per_second: float
    max_power_w: float | None = None


@dataclass(frozen=True)
class DeploymentCheck:
    target: str
    passed: bool
    metrics: dict[str, float]
    issues: list[str]


class DeploymentValidator:
    def __init__(self, targets: list[DeploymentTarget]) -> None:
        self.targets = targets

    @classmethod
    def default(cls) -> "DeploymentValidator":
        return cls(
            [
                DeploymentTarget("server-h100-vllm", "vllm", 180.0, 80.0, 80.0),
                DeploymentTarget("edge-orin", "tensorrt-llm", 450.0, 24.0, 18.0, max_power_w=60.0),
                DeploymentTarget("cockpit-soc", "onnx-runtime", 700.0, 16.0, 12.0, max_power_w=35.0),
            ]
        )

    def validate(self, metrics_by_target: dict[str, dict[str, float]] | None = None) -> list[DeploymentCheck]:
        metrics_by_target = metrics_by_target or {}
        checks: list[DeploymentCheck] = []
        for target in self.targets:
            metrics = metrics_by_target.get(
                target.name,
                {
                    "latency_ms": target.max_latency_ms * 0.9,
                    "memory_gb": target.max_memory_gb * 0.85,
                    "decode_tokens_per_second": target.min_decode_tokens_per_second * 1.1,
                    "power_w": (target.max_power_w or 100.0) * 0.8,
                },
            )
            issues: list[str] = []
            if metrics["latency_ms"] > target.max_latency_ms:
                issues.append(f"latency_ms {metrics['latency_ms']} exceeds {target.max_latency_ms}")
            if metrics["memory_gb"] > target.max_memory_gb:
                issues.append(f"memory_gb {metrics['memory_gb']} exceeds {target.max_memory_gb}")
            if metrics["decode_tokens_per_second"] < target.min_decode_tokens_per_second:
                issues.append(
                    f"decode_tokens_per_second {metrics['decode_tokens_per_second']} below {target.min_decode_tokens_per_second}"
                )
            if target.max_power_w is not None and metrics.get("power_w", 0.0) > target.max_power_w:
                issues.append(f"power_w {metrics['power_w']} exceeds {target.max_power_w}")
            checks.append(DeploymentCheck(target.name, not issues, metrics, issues))
        return checks

    def write_report(self, output: str | Path, metrics_by_target: dict[str, dict[str, float]] | None = None) -> Path:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        checks = self.validate(metrics_by_target)
        payload: dict[str, Any] = {
            "summary": {
                "target_count": len(checks),
                "passed": sum(1 for check in checks if check.passed),
                "failed": sum(1 for check in checks if not check.passed),
            },
            "checks": [asdict(check) for check in checks],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

