from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .data import DataSystem
from .dataset_catalog import DatasetCatalog


@dataclass(frozen=True)
class PipelineArtifact:
    stage: str
    status: str
    output_uri: str
    checks: dict[str, Any]


class DataPipelineRunner:
    def __init__(self, data_system: DataSystem, catalog: DatasetCatalog) -> None:
        self.data_system = data_system
        self.catalog = catalog

    def build_plan(self) -> list[PipelineArtifact]:
        artifacts: list[PipelineArtifact] = []
        for stage in self.data_system.processing_stages:
            digest = hashlib.sha256(
                json.dumps(
                    {
                        "stage": stage.name,
                        "operations": stage.operations,
                        "sources": [source.name for source in self.data_system.sources],
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()[:12]
            artifacts.append(
                PipelineArtifact(
                    stage=stage.name,
                    status="planned",
                    output_uri=f"artifact://data/{stage.output}/{digest}",
                    checks={
                        "operation_count": len(stage.operations),
                        "owner": stage.owner,
                        "source_count": len(self.data_system.sources),
                    },
                )
            )
        return artifacts

    def run_dry(self, output: str | Path) -> Path:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "data_summary": self.data_system.summary(),
            "catalog_summary": self.catalog.summary(),
            "artifacts": [asdict(item) for item in self.build_plan()],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

