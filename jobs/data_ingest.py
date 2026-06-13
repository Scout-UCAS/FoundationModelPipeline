from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fmops.data import DataSystem
from fmops.dataset_catalog import DatasetCatalog


def build_payload(manifest: Path, catalog_path: Path, executor: str) -> dict[str, object]:
    data = DataSystem.from_file(manifest)
    catalog = DatasetCatalog.from_file(catalog_path)
    summary = data.summary()
    return {
        "job": "data_ingest",
        "executor": executor,
        "status": "materialized",
        "summary": summary,
        "sources": [
            {
                "name": source.name,
                "source_type": source.source_type,
                "modalities": source.modalities,
                "languages": source.languages,
                "size_tb": source.size_tb,
                "license": source.license,
                "tags": source.tags,
            }
            for source in data.sources
        ],
        "catalog": {
            "dataset_count": len(catalog.datasets),
            "families": sorted(catalog.summary()["families"]),
            "modalities": sorted(catalog.summary()["modalities"]),
        },
        "gates": {
            "total_tb": summary["actual_tb"],
            "language_count": summary["language_count"],
            "modalities": summary["modalities"],
            "scale_passed": summary["actual_tb"] >= 2500,
            "language_passed": summary["language_count"] >= 20,
            "modalities_passed": set(summary["modalities"]) >= {"pure_text", "multimodal", "video_pretraining", "vla"},
        },
    }


def submit_to_executor(args: argparse.Namespace) -> int:
    command = [
        "spark-submit",
        str(Path(__file__).resolve()),
        "--manifest",
        str(args.manifest),
        "--catalog",
        str(args.catalog),
        "--output",
        str(args.output),
        "--executor",
        "spark",
    ]
    completed = subprocess.run(command, check=False)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize data ingestion manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--executor", default="local", choices=["local", "spark"])
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    if args.submit:
        return submit_to_executor(args)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload(Path(args.manifest), Path(args.catalog), args.executor)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
