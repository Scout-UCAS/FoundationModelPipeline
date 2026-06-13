from __future__ import annotations

import json
import math
import re
import shlex
import string
import subprocess
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .evaluation import Benchmark, EvaluationSuite


LOWER_IS_BETTER = {
    "action_l2",
    "calibration_error",
    "cost_per_success",
    "frame_action_latency_ms",
    "memory_gb",
    "power_w",
    "prefill_ms",
    "steps_to_success",
    "tool_error_rate",
    "unsafe_action_rate",
}


@dataclass(frozen=True)
class EvaluationResult:
    benchmark: str
    dimension: str
    metrics: dict[str, float]
    passed: bool
    artifacts: dict[str, str]
    sample_count: int


@dataclass(frozen=True)
class EvaluationSample:
    id: str
    benchmark: str
    dataset: str
    prompt: str
    reference: Any
    payload: dict[str, Any]


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_").lower()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_text(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = [word for word in text.split() if word not in {"a", "an", "the"}]
    return " ".join(words)


def _token_f1(prediction: Any, reference: Any) -> float:
    pred_tokens = _normalize_text(prediction).split()
    ref_tokens = _normalize_text(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    ref_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    overlap = 0
    for token in pred_tokens:
        count = ref_counts.get(token, 0)
        if count > 0:
            overlap += 1
            ref_counts[token] = count - 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _as_references(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _parse_json_object(text: Any) -> dict[str, Any] | None:
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped:
        return None
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _to_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def _to_vector(value: Any) -> list[float] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            nums = re.findall(r"-?\d+(?:\.\d+)?", value)
            return [float(item) for item in nums] if nums else None
    if isinstance(value, dict):
        for key in ("action", "predicted_action", "trajectory"):
            if key in value:
                return _to_vector(value[key])
        return None
    if isinstance(value, list):
        output = []
        for item in value:
            number = _to_number(item)
            if number is None:
                return None
            output.append(number)
        return output
    return None


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _extract_prompt(payload: dict[str, Any]) -> str:
    for key in ("prompt", "question", "instruction", "input", "context"):
        value = payload.get(key)
        if value:
            return str(value)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _extract_reference(payload: dict[str, Any]) -> Any:
    for key in ("answer", "reference", "target", "gold", "label", "expected"):
        if key in payload:
            return payload[key]
    if "reference_action" in payload:
        return payload["reference_action"]
    return None


def _sample_from_payload(payload: dict[str, Any], *, fallback_benchmark: str, index: int) -> EvaluationSample:
    sample_id = str(payload.get("id") or payload.get("sample_id") or f"{fallback_benchmark}-{index:06d}")
    return EvaluationSample(
        id=sample_id,
        benchmark=str(payload.get("benchmark") or fallback_benchmark),
        dataset=str(payload.get("dataset") or payload.get("source") or fallback_benchmark),
        prompt=_extract_prompt(payload),
        reference=_extract_reference(payload),
        payload=payload,
    )


class EvaluationRunner:
    def __init__(
        self,
        suite: EvaluationSuite,
        *,
        samples_dir: str | Path | None = None,
        predictions_path: str | Path | None = None,
        model_command: str | None = None,
        model_endpoint: str | None = None,
        timeout_seconds: float = 30.0,
        benchmark_filter: set[str] | None = None,
    ) -> None:
        self.suite = suite
        self.samples_dir = Path(samples_dir) if samples_dir else _repo_root() / "eval" / "smoke"
        self.predictions = self._load_predictions(predictions_path)
        self.model_command = model_command
        self.model_endpoint = model_endpoint
        self.timeout_seconds = timeout_seconds
        self.benchmark_filter = benchmark_filter

    @staticmethod
    def _load_predictions(path: str | Path | None) -> dict[str, Any]:
        if path is None:
            return {}
        prediction_path = Path(path)
        if not prediction_path.exists():
            raise FileNotFoundError(f"prediction file not found: {prediction_path}")
        if prediction_path.suffix == ".jsonl":
            predictions: dict[str, Any] = {}
            with prediction_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    payload = json.loads(stripped)
                    sample_id = str(payload.get("id") or payload.get("sample_id"))
                    predictions[sample_id] = payload.get("prediction", payload.get("response", payload))
            return predictions
        payload = json.loads(prediction_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if "predictions" in payload and isinstance(payload["predictions"], list):
                return {
                    str(item.get("id") or item.get("sample_id")): item.get("prediction", item.get("response", item))
                    for item in payload["predictions"]
                }
            return payload
        if isinstance(payload, list):
            return {
                str(item.get("id") or item.get("sample_id")): item.get("prediction", item.get("response", item))
                for item in payload
            }
        raise ValueError(f"unsupported prediction file format: {prediction_path}")

    def _benchmarks(self) -> list[Benchmark]:
        if not self.benchmark_filter:
            return list(self.suite.benchmarks)
        return [
            benchmark
            for benchmark in self.suite.benchmarks
            if benchmark.name in self.benchmark_filter or benchmark.dimension in self.benchmark_filter
        ]

    def _load_samples(self, benchmark: Benchmark) -> list[EvaluationSample]:
        samples: list[EvaluationSample] = []
        if self.samples_dir.exists():
            candidates = [
                self.samples_dir / f"{benchmark.name}.jsonl",
                self.samples_dir / f"{_safe_name(benchmark.name)}.jsonl",
                self.samples_dir / f"{benchmark.dimension}.jsonl",
                self.samples_dir / f"{_safe_name(benchmark.dimension)}.jsonl",
            ]
            seen_paths: set[Path] = set()
            for candidate in candidates:
                if candidate in seen_paths or not candidate.exists():
                    continue
                seen_paths.add(candidate)
                samples.extend(self._read_jsonl_samples(candidate, benchmark.name))
            if not samples:
                for candidate in sorted(self.samples_dir.glob("*.jsonl")):
                    for sample in self._read_jsonl_samples(candidate, candidate.stem):
                        if sample.benchmark == benchmark.name or sample.dataset in benchmark.datasets:
                            samples.append(sample)
        if samples:
            return samples
        return self._generated_smoke_samples(benchmark)

    @staticmethod
    def _read_jsonl_samples(path: Path, fallback_benchmark: str) -> list[EvaluationSample]:
        samples: list[EvaluationSample] = []
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    payload = {"prompt": str(payload), "answer": str(payload)}
                samples.append(_sample_from_payload(payload, fallback_benchmark=fallback_benchmark, index=index))
        return samples

    @staticmethod
    def _generated_smoke_samples(benchmark: Benchmark) -> list[EvaluationSample]:
        payload = {
            "id": f"{_safe_name(benchmark.name)}-generated-0",
            "benchmark": benchmark.name,
            "dataset": benchmark.datasets[0] if benchmark.datasets else benchmark.name,
            "prompt": f"Generated smoke sample for {benchmark.name}",
            "answer": "yes",
            "prediction": "yes",
            "confidence": 0.9,
            "success": True,
            "collision_free": True,
            "reference_action": [0.0, 1.0],
            "predicted_action": [0.0, 1.0],
            "steps": 1,
            "cost": 1.0,
            "prefill_ms": 20.0,
            "decode_tok_s": 40.0,
            "memory_gb": 4.0,
            "power_w": 25.0,
            "frame_action_latency_ms": 30.0,
        }
        return [_sample_from_payload(payload, fallback_benchmark=benchmark.name, index=0)]

    def _call_model_command(self, sample: EvaluationSample) -> Any:
        if not self.model_command:
            raise RuntimeError("model_command is not configured")
        completed = subprocess.run(
            shlex.split(self.model_command),
            input=json.dumps(sample.payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            check=False,
            timeout=self.timeout_seconds,
        )
        if completed.returncode != 0:
            return {
                "error": "model_command_failed",
                "exit_code": completed.returncode,
                "stderr": completed.stderr[-2000:],
            }
        return self._extract_prediction_from_model_payload(completed.stdout.strip())

    def _call_model_endpoint(self, sample: EvaluationSample) -> Any:
        if not self.model_endpoint:
            raise RuntimeError("model_endpoint is not configured")
        request = urllib.request.Request(
            self.model_endpoint,
            data=json.dumps(sample.payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            body = response.read().decode("utf-8")
        return self._extract_prediction_from_model_payload(body)

    @staticmethod
    def _extract_prediction_from_model_payload(payload: Any) -> Any:
        parsed = _parse_json_object(payload)
        if parsed is None:
            return payload
        for key in ("prediction", "response", "output", "text", "answer"):
            if key in parsed:
                return parsed[key]
        return parsed

    def _prediction_for(self, sample: EvaluationSample) -> Any:
        if sample.id in self.predictions:
            return self.predictions[sample.id]
        if self.model_command:
            return self._call_model_command(sample)
        if self.model_endpoint:
            return self._call_model_endpoint(sample)
        payload = sample.payload
        for key in ("prediction", "model_output", "response", "output"):
            if key in payload:
                return payload[key]
        return ""

    @staticmethod
    def _is_correct(sample: EvaluationSample, prediction: Any) -> float:
        references = _as_references(sample.reference)
        if not references:
            return 0.0
        choices = sample.payload.get("choices")
        normalized_prediction = _normalize_text(prediction)
        for reference in references:
            if _normalize_text(reference) == normalized_prediction:
                return 1.0
            if isinstance(choices, list):
                for idx, choice in enumerate(choices):
                    letter = chr(ord("A") + idx)
                    if _normalize_text(reference) == _normalize_text(letter):
                        if normalized_prediction in {_normalize_text(letter), _normalize_text(choice)}:
                            return 1.0
        return 0.0

    @staticmethod
    def _best_f1(sample: EvaluationSample, prediction: Any) -> float:
        references = _as_references(sample.reference)
        if not references:
            return 0.0
        return max(_token_f1(prediction, reference) for reference in references)

    @staticmethod
    def _schema_validity(sample: EvaluationSample, prediction: Any) -> float:
        parsed = _parse_json_object(prediction)
        if parsed is None:
            return 0.0
        expected_tool = sample.payload.get("expected_tool")
        if expected_tool and parsed.get("tool") != expected_tool and parsed.get("name") != expected_tool:
            return 0.0
        expected_arguments = sample.payload.get("expected_arguments")
        if isinstance(expected_arguments, dict):
            arguments = parsed.get("arguments", parsed.get("args", {}))
            if not isinstance(arguments, dict):
                return 0.0
            for key, value in expected_arguments.items():
                if arguments.get(key) != value:
                    return 0.0
        required_fields = sample.payload.get("required_fields", [])
        if isinstance(required_fields, list):
            for field in required_fields:
                if str(field) not in parsed:
                    return 0.0
        return 1.0

    @staticmethod
    def _action_l2(sample: EvaluationSample, prediction: Any) -> float:
        reference = _to_vector(sample.payload.get("reference_action", sample.reference))
        predicted = _to_vector(sample.payload.get("predicted_action", prediction))
        if not reference or not predicted or len(reference) != len(predicted):
            return 0.0
        return math.sqrt(sum((lhs - rhs) ** 2 for lhs, rhs in zip(reference, predicted)))

    @staticmethod
    def _sample_numeric(sample: EvaluationSample, prediction: Any, metric: str) -> float | None:
        scores = sample.payload.get("scores", sample.payload.get("metrics", {}))
        if isinstance(scores, dict) and metric in scores:
            return _to_number(scores[metric])
        parsed = _parse_json_object(prediction)
        if parsed is not None and metric in parsed:
            return _to_number(parsed[metric])
        aliases = {
            "steps_to_success": ("steps_to_success", "steps"),
            "cost_per_success": ("cost_per_success", "cost"),
            "prefill_ms": ("prefill_ms", "ttft_ms"),
            "decode_tok_s": ("decode_tok_s", "tokens_per_second"),
            "memory_gb": ("memory_gb", "peak_memory_gb"),
            "power_w": ("power_w", "power_watts"),
            "frame_action_latency_ms": ("frame_action_latency_ms", "latency_ms"),
        }
        for key in aliases.get(metric, (metric,)):
            if key in sample.payload:
                return _to_number(sample.payload[key])
        return None

    def _metric_values(self, metric: str, samples: list[EvaluationSample], predictions: dict[str, Any]) -> list[float]:
        values: list[float] = []
        for sample in samples:
            prediction = predictions[sample.id]
            correct = self._is_correct(sample, prediction)
            if metric in {"accuracy", "exact_match", "pass_at_1"}:
                values.append(correct)
            elif metric in {"f1", "ocr_f1"}:
                values.append(self._best_f1(sample, prediction))
            elif metric in {
                "consensus_pass",
                "verified_solution_rate",
                "temporal_reasoning",
                "long_range_consistency",
                "citation_precision",
            }:
                value = self._sample_numeric(sample, prediction, metric)
                values.append(value if value is not None else max(correct, self._best_f1(sample, prediction)))
            elif metric == "calibration_error":
                confidence = self._sample_numeric(sample, prediction, "confidence")
                if confidence is None:
                    confidence = _to_number(sample.payload.get("confidence"))
                values.append(abs((confidence if confidence is not None else correct) - correct))
            elif metric == "schema_validity":
                values.append(self._schema_validity(sample, prediction))
            elif metric in {"task_success", "success_rate", "recovery_rate"}:
                value = self._sample_numeric(sample, prediction, metric)
                if value is None:
                    key = "success" if metric in {"task_success", "success_rate"} else "recovered"
                    value = _to_number(sample.payload.get(key))
                values.append(value if value is not None else correct)
            elif metric == "tool_error_rate":
                value = self._sample_numeric(sample, prediction, metric)
                if value is None:
                    tool_error = _to_number(sample.payload.get("tool_error"))
                    value = tool_error if tool_error is not None else 1.0 - self._schema_validity(sample, prediction)
                values.append(value)
            elif metric == "unsafe_action_rate":
                value = self._sample_numeric(sample, prediction, metric)
                if value is None:
                    value = _to_number(sample.payload.get("unsafe_action"))
                values.append(value if value is not None else 0.0)
            elif metric == "collision_free_rate":
                value = self._sample_numeric(sample, prediction, metric)
                if value is None:
                    value = _to_number(sample.payload.get("collision_free"))
                values.append(value if value is not None else correct)
            elif metric == "action_l2":
                values.append(self._action_l2(sample, prediction))
            elif metric in {
                "steps_to_success",
                "cost_per_success",
                "prefill_ms",
                "decode_tok_s",
                "memory_gb",
                "power_w",
                "frame_action_latency_ms",
            }:
                value = self._sample_numeric(sample, prediction, metric)
                if value is not None:
                    values.append(value)
            else:
                value = self._sample_numeric(sample, prediction, metric)
                values.append(value if value is not None else correct)
        return values

    @staticmethod
    def _macro_average(samples: list[EvaluationSample], predictions: dict[str, Any]) -> float:
        by_dataset: dict[str, list[float]] = {}
        for sample in samples:
            by_dataset.setdefault(sample.dataset, []).append(EvaluationRunner._is_correct(sample, predictions[sample.id]))
        return _mean([_mean(values) for values in by_dataset.values()])

    @staticmethod
    def _passes_gate(metric: str, value: float, threshold: Any) -> bool:
        target = float(threshold)
        if metric.startswith("max_"):
            return value <= target
        if metric.startswith("min_"):
            return value >= target
        if metric in LOWER_IS_BETTER:
            return value <= target
        return value >= target

    def _evaluate_benchmark(
        self,
        benchmark: Benchmark,
        artifact_dir: Path,
        model_id: str,
    ) -> EvaluationResult:
        samples = self._load_samples(benchmark)
        predictions = {sample.id: self._prediction_for(sample) for sample in samples}
        metrics: dict[str, float] = {}
        for metric in benchmark.metrics:
            if metric == "macro_average":
                metrics[metric] = self._macro_average(samples, predictions)
            else:
                metrics[metric] = _mean(self._metric_values(metric, samples, predictions))

        passed = True
        for gate_metric, gate_value in benchmark.gate.items():
            value = metrics.get(gate_metric)
            if value is None:
                passed = False
            else:
                passed = passed and self._passes_gate(gate_metric, value, gate_value)

        artifact_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = artifact_dir / f"{_safe_name(benchmark.name)}.jsonl"
        with transcript_path.open("w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(
                    json.dumps(
                        {
                            "id": sample.id,
                            "benchmark": benchmark.name,
                            "dataset": sample.dataset,
                            "prompt": sample.prompt,
                            "reference": sample.reference,
                            "prediction": predictions[sample.id],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        return EvaluationResult(
            benchmark=benchmark.name,
            dimension=benchmark.dimension,
            metrics=metrics,
            passed=passed,
            artifacts={"model_id": model_id, "transcript": str(transcript_path)},
            sample_count=len(samples),
        )

    def run(self, model_id: str = "reference-model", artifact_dir: str | Path | None = None) -> list[EvaluationResult]:
        started = time.monotonic()
        root = Path(artifact_dir) if artifact_dir else Path("artifacts/runs/eval_transcripts")
        results = [self._evaluate_benchmark(benchmark, root, model_id) for benchmark in self._benchmarks()]
        duration_path = root / "run_metadata.json"
        root.mkdir(parents=True, exist_ok=True)
        duration_path.write_text(
            json.dumps(
                {
                    "model_id": model_id,
                    "duration_seconds": round(time.monotonic() - started, 4),
                    "benchmark_count": len(results),
                    "sample_count": sum(result.sample_count for result in results),
                    "samples_dir": str(self.samples_dir),
                    "mode": "command" if self.model_command else "http" if self.model_endpoint else "local-jsonl",
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return results

    def write_report(self, output: str | Path, model_id: str = "reference-model") -> Path:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact_dir = path.parent / f"{path.stem}_transcripts"
        results = self.run(model_id=model_id, artifact_dir=artifact_dir)
        dimension_scores: dict[str, list[float]] = {}
        for result in results:
            dimension_scores.setdefault(result.dimension, []).append(1.0 if result.passed else 0.0)
        payload = {
            "model_id": model_id,
            "summary": {
                "benchmark_count": len(results),
                "sample_count": sum(result.sample_count for result in results),
                "passed": sum(1 for result in results if result.passed),
                "failed": sum(1 for result in results if not result.passed),
                "dimension_scores": {
                    dimension: _mean(scores) for dimension, scores in sorted(dimension_scores.items())
                },
            },
            "results": [asdict(result) for result in results],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
