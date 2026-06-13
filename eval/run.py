from __future__ import annotations

import argparse

from fmops.evaluation_runner import EvaluationRunner
from fmops.pipeline import load_platform


def main() -> int:
    parser = argparse.ArgumentParser(description="Run configured evaluation suite")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--model-id", default="reference-model")
    parser.add_argument("--output", default="artifacts/runs/evaluation_report.json")
    args = parser.parse_args()
    suite = load_platform(args.config_dir).evaluation
    path = EvaluationRunner(suite).write_report(args.output, model_id=args.model_id)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

