from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .evaluation import Benchmark, EvaluationSuite


@dataclass(frozen=True)
class EvaluationResult:
    benchmark: str
    dimension: str
    metrics: dict[str, float]
    passed: bool
    artifacts: dict[str, str]


class EvaluationRunner:
    def __init__(self, suite: EvaluationSuite) -> None:
        self.suite = suite

    @staticmethod
    def _synthetic_metric(benchmark: Benchmark, metric: str) -> float:
        digest = hashlib.sha256(f"{benchmark.name}:{metric}".encode("utf-8")).hexdigest()
        raw = int(digest[:8], 16) / 0xFFFFFFFF
        return round(0.45 + raw * 0.5, 4)

    def run(self, model_id: str = "reference-model") -> list[EvaluationResult]:
        results: list[EvaluationResult] = []
        for benchmark in self.suite.benchmarks:
            metrics = {metric: self._synthetic_metric(benchmark, metric) for metric in benchmark.metrics}
            passed = True
            for gate_metric, gate_value in benchmark.gate.items():
                value = metrics.get(gate_metric)
                if value is None:
                    passed = False
                    continue
                passed = passed and value >= float(gate_value)
            results.append(
                EvaluationResult(
                    benchmark=benchmark.name,
                    dimension=benchmark.dimension,
                    metrics=metrics,
                    passed=passed,
                    artifacts={"model_id": model_id, "transcript": f"artifact://eval/{model_id}/{benchmark.name}.jsonl"},
                )
            )
        return results

    def write_report(self, output: str | Path, model_id: str = "reference-model") -> Path:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        results = self.run(model_id=model_id)
        payload = {
            "model_id": model_id,
            "summary": {
                "benchmark_count": len(results),
                "passed": sum(1 for result in results if result.passed),
                "failed": sum(1 for result in results if not result.passed),
            },
            "results": [asdict(result) for result in results],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

