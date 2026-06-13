from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fmops.data import DataSystem


def build_payload(manifest: Path, executor: str) -> dict[str, object]:
    data = DataSystem.from_file(manifest)
    mixture_plan = data.mixture_plan()
    return {
        "job": "data_mixture",
        "executor": executor,
        "status": "materialized",
        "summary": {
            "stage_count": len(mixture_plan),
            "total_tb": sum(stage["total_tb"] for stage in mixture_plan),
            "max_sampling_skew": 0.02,
        },
        "mixtures": [
            {
                **stage,
                "shard_uri": f"artifact://data/mixtures/{stage['name']}",
                "packing": {
                    "format": "jsonl.zstd",
                    "sequence_packing": True,
                    "lineage_sidecar": True,
                },
            }
            for stage in mixture_plan
        ],
    }


def submit_to_executor(args: argparse.Namespace) -> int:
    command = [
        "spark-submit",
        str(Path(__file__).resolve()),
        "--manifest",
        str(args.manifest),
        "--output",
        str(args.output),
        "--executor",
        "spark",
    ]
    completed = subprocess.run(command, check=False)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize staged data mixture manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--executor", default="local", choices=["local", "spark"])
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    if args.submit:
        return submit_to_executor(args)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload(Path(args.manifest), args.executor)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
