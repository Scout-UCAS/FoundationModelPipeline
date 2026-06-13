from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fmops.deployment import DeploymentValidator


def build_payload(model_path: str, submitted: dict[str, object] | None = None) -> dict[str, object]:
    validator = DeploymentValidator.default()
    checks = validator.validate()
    return {
        "job": "deployment_validate",
        "model_path": model_path,
        "status": "validated",
        "summary": {
            "target_count": len(checks),
            "passed": sum(1 for check in checks if check.passed),
            "failed": sum(1 for check in checks if not check.passed),
        },
        "checks": [check.__dict__ for check in checks],
        "benchmarks": {
            "server": "vllm benchmark + GenAI-Perf",
            "edge": "TensorRT-LLM benchmark",
            "car_side": "streaming video VLA replay",
        },
        "submission": submitted or {"submitted": False, "reason": "not requested"},
    }


def submit(model_path: str, output: Path) -> dict[str, object]:
    command = f"genai-perf profile --model {model_path} --artifact-dir {output.parent}"
    completed = subprocess.run(command, shell=True, check=False, capture_output=True, text=True)
    return {
        "submitted": completed.returncode == 0,
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate serving and car-side deployment envelopes")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    submitted = submit(args.model_path, output) if args.submit else None
    payload = build_payload(args.model_path, submitted=submitted)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output}")
    return 0 if not submitted or submitted["submitted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
