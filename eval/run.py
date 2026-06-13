from __future__ import annotations

import argparse

from fmops.evaluation_runner import EvaluationRunner
from fmops.pipeline import load_platform


def main() -> int:
    parser = argparse.ArgumentParser(description="Run configured evaluation suite")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--model-id", default="reference-model")
    parser.add_argument("--output", default="artifacts/runs/evaluation_report.json")
    parser.add_argument("--samples-dir", help="Directory of JSONL evaluation samples")
    parser.add_argument("--predictions", help="JSON/JSONL prediction file keyed by sample id")
    parser.add_argument("--model-command", help="External command that reads one sample JSON from stdin")
    parser.add_argument("--model-endpoint", help="HTTP endpoint that accepts one sample JSON per request")
    parser.add_argument("--benchmark", action="append", help="Benchmark name or dimension to evaluate")
    parser.add_argument("--fail-on-gate", action="store_true", help="Return non-zero if any gate fails")
    args = parser.parse_args()
    suite = load_platform(args.config_dir).evaluation
    path = EvaluationRunner(
        suite,
        samples_dir=args.samples_dir,
        predictions_path=args.predictions,
        model_command=args.model_command,
        model_endpoint=args.model_endpoint,
        benchmark_filter=set(args.benchmark or []),
    ).write_report(args.output, model_id=args.model_id)
    print(f"Wrote {path}")
    if args.fail_on_gate:
        import json
        from pathlib import Path

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return 1 if payload["summary"]["failed"] else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
