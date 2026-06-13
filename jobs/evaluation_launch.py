from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fmops.evaluation_runner import EvaluationRunner
from fmops.pipeline import load_platform


def command_for_harness(harness: str, model_id: str, benchmark: str, output: Path) -> str:
    if harness == "lm-eval":
        return f"lm_eval --model hf --model_args pretrained={model_id} --tasks {benchmark} --output_path {output}"
    if harness == "vlmevalkit":
        return f"vlm_eval --model {model_id} --data {benchmark} --work-dir {output.parent}"
    if harness == "opencompass":
        return f"opencompass --models {model_id} --datasets {benchmark} --work-dir {output.parent}"
    if harness == "simulator":
        return f"{sys.executable} -m fmops.cli eval-run --model-id {model_id} --output {output}"
    raise ValueError(f"unsupported harness: {harness}")


def build_payload(config_dir: Path, model_id: str, benchmark: str, harness: str, output: Path) -> dict[str, object]:
    platform = load_platform(config_dir)
    selected = {item.strip() for item in benchmark.split(",") if item.strip()}
    configured = [item for item in platform.evaluation.benchmarks if item.name in selected or item.dimension in selected]
    synthetic = EvaluationRunner(platform.evaluation).run(model_id=model_id)
    synthetic_by_name = {item.benchmark: item for item in synthetic}
    return {
        "job": "evaluation_launch",
        "harness": harness,
        "model_id": model_id,
        "benchmark_selector": benchmark,
        "status": "prepared",
        "command": command_for_harness(harness, model_id, benchmark, output),
        "configured_benchmarks": [
            {
                "name": item.name,
                "dimension": item.dimension,
                "datasets": item.datasets,
                "metrics": item.metrics,
                "gate": item.gate,
                "synthetic_smoke_result": synthetic_by_name[item.name].metrics,
            }
            for item in configured
        ],
    }


def submit(payload: dict[str, object]) -> dict[str, object]:
    completed = subprocess.run(str(payload["command"]), check=False, shell=True, capture_output=True, text=True)
    return {
        "submitted": completed.returncode == 0,
        "exit_code": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare or submit production evaluation harness run")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--harness", required=True, choices=["lm-eval", "vlmevalkit", "opencompass", "simulator"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload(Path(args.config_dir), args.model_id, args.benchmark, args.harness, output)
    if args.submit:
        payload["submission"] = submit(payload)
        payload["status"] = "submitted" if payload["submission"]["submitted"] else "submission_failed"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output}")
    return 0 if payload.get("status") != "submission_failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
