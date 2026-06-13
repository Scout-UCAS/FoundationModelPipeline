from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PluginManifest:
    name: str
    version: str
    kind: str
    module: str
    entrypoint: str
    description: str = ""

    @classmethod
    def from_file(cls, path: str | Path) -> "PluginManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=str(payload["name"]),
            version=str(payload.get("version", "0.0.0")),
            kind=str(payload["kind"]),
            module=str(payload["module"]),
            entrypoint=str(payload["entrypoint"]),
            description=str(payload.get("description", "")),
        )


class PluginManager:
    def __init__(self, plugin_dir: str | Path = "plugins") -> None:
        self.plugin_dir = Path(plugin_dir)

    def discover(self) -> list[PluginManifest]:
        if not self.plugin_dir.exists():
            return []
        manifests = []
        for path in sorted(self.plugin_dir.glob("*/plugin.json")):
            manifests.append(PluginManifest.from_file(path))
        return manifests

    def load(self, manifest: PluginManifest) -> Any:
        module = importlib.import_module(manifest.module)
        return getattr(module, manifest.entrypoint)

    def validate(self) -> list[str]:
        issues: list[str] = []
        for manifest in self.discover():
            try:
                self.load(manifest)
            except Exception as exc:
                issues.append(f"{manifest.name}: failed to load {manifest.module}.{manifest.entrypoint}: {exc}")
        return issues

