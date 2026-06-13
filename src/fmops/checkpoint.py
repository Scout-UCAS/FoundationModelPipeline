from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CheckpointManifest:
    checkpoint_id: str
    source_path: str
    target_path: str
    source_format: str
    target_format: str
    files: list[str]
    metadata: dict[str, Any]


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CheckpointConverter:
    def convert(
        self,
        source: str | Path,
        target: str | Path,
        source_format: str = "training",
        target_format: str = "inference",
        copy_files: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> CheckpointManifest:
        source_path = Path(source)
        target_path = Path(target)
        target_path.mkdir(parents=True, exist_ok=True)
        if not source_path.exists():
            raise FileNotFoundError(source_path)

        files: list[str] = []
        if source_path.is_file():
            files = [source_path.name]
            if copy_files:
                shutil.copy2(source_path, target_path / source_path.name)
            checkpoint_seed = _file_digest(source_path)
        else:
            source_files = sorted(path for path in source_path.rglob("*") if path.is_file())
            for file_path in source_files:
                relative = file_path.relative_to(source_path)
                files.append(str(relative))
                if copy_files:
                    out_file = target_path / relative
                    out_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, out_file)
            checkpoint_seed = hashlib.sha256("\n".join(files).encode("utf-8")).hexdigest()

        manifest = CheckpointManifest(
            checkpoint_id=checkpoint_seed[:16],
            source_path=str(source_path),
            target_path=str(target_path),
            source_format=source_format,
            target_format=target_format,
            files=files,
            metadata=metadata or {},
        )
        (target_path / "checkpoint_manifest.json").write_text(
            json.dumps(asdict(manifest), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return manifest


def load_checkpoint_manifest(path: str | Path) -> CheckpointManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return CheckpointManifest(**payload)

