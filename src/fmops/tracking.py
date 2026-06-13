from __future__ import annotations

import json
import os
import platform
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    name: str
    kind: str
    started_at: float
    status: str
    config_refs: dict[str, str]
    artifacts: dict[str, str]
    metrics: dict[str, float]
    environment: dict[str, str]


class ExperimentTracker:
    def __init__(self, root: str | Path = "artifacts/runs") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def start_run(
        self,
        name: str,
        kind: str,
        config_refs: dict[str, str] | None = None,
        metrics: dict[str, float] | None = None,
        artifacts: dict[str, str] | None = None,
    ) -> RunManifest:
        run_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
        manifest = RunManifest(
            run_id=run_id,
            name=name,
            kind=kind,
            started_at=time.time(),
            status="started",
            config_refs=config_refs or {},
            artifacts=artifacts or {},
            metrics=metrics or {},
            environment={
                "python": platform.python_version(),
                "platform": platform.platform(),
                "pid": str(os.getpid()),
            },
        )
        self.write_manifest(manifest)
        return manifest

    def write_manifest(self, manifest: RunManifest) -> Path:
        path = self.root / manifest.run_id / "run_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def complete_run(
        self,
        manifest: RunManifest,
        status: str = "completed",
        metrics: dict[str, float] | None = None,
        artifacts: dict[str, str] | None = None,
    ) -> RunManifest:
        updated = RunManifest(
            run_id=manifest.run_id,
            name=manifest.name,
            kind=manifest.kind,
            started_at=manifest.started_at,
            status=status,
            config_refs=manifest.config_refs,
            artifacts={**manifest.artifacts, **(artifacts or {})},
            metrics={**manifest.metrics, **(metrics or {})},
            environment=manifest.environment,
        )
        self.write_manifest(updated)
        return updated

    def list_runs(self) -> list[RunManifest]:
        runs: list[RunManifest] = []
        for path in sorted(self.root.glob("*/run_manifest.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            runs.append(RunManifest(**payload))
        return runs

