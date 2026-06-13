from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_json


@dataclass(frozen=True)
class BenchmarkCatalogEntry:
    name: str
    dimension: str
    family: str
    modalities: tuple[str, ...]
    metrics: tuple[str, ...]
    harnesses: tuple[str, ...]
    download_url: str
    license: str
    primary_metric: str
    tags: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BenchmarkCatalogEntry":
        return cls(
            name=str(payload["name"]),
            dimension=str(payload["dimension"]),
            family=str(payload.get("family", "")),
            modalities=tuple(str(item) for item in payload.get("modalities", [])),
            metrics=tuple(str(item) for item in payload.get("metrics", [])),
            harnesses=tuple(str(item) for item in payload.get("harnesses", [])),
            download_url=str(payload.get("download_url", "")),
            license=str(payload.get("license", "unknown")),
            primary_metric=str(payload.get("primary_metric", "")),
            tags=tuple(str(item) for item in payload.get("tags", [])),
        )


class BenchmarkCatalog:
    def __init__(self, version: str, benchmarks: list[BenchmarkCatalogEntry]) -> None:
        self.version = version
        self.benchmarks = benchmarks

    @classmethod
    def from_file(cls, path: str | Path) -> "BenchmarkCatalog":
        payload = load_json(path)
        return cls(
            version=str(payload.get("version", "unknown")),
            benchmarks=[BenchmarkCatalogEntry.from_dict(item) for item in payload.get("benchmarks", [])],
        )

    def validate(self) -> list[str]:
        issues: list[str] = []
        seen: set[str] = set()
        for index, benchmark in enumerate(self.benchmarks):
            context = f"benchmark_catalog.benchmarks[{index}].{benchmark.name}"
            if benchmark.name in seen:
                issues.append(f"{context}: duplicate benchmark name")
            seen.add(benchmark.name)
            if not benchmark.dimension:
                issues.append(f"{context}: dimension must not be empty")
            if not benchmark.family:
                issues.append(f"{context}: family must not be empty")
            if not benchmark.modalities:
                issues.append(f"{context}: modalities must not be empty")
            if not benchmark.metrics:
                issues.append(f"{context}: metrics must not be empty")
            if not benchmark.primary_metric:
                issues.append(f"{context}: primary_metric must not be empty")
            if benchmark.primary_metric and benchmark.primary_metric not in benchmark.metrics:
                issues.append(f"{context}: primary_metric must be listed in metrics")
            if not benchmark.download_url:
                issues.append(f"{context}: download_url must not be empty")
        return issues

    def filter(
        self,
        *,
        dimension: str | None = None,
        modality: str | None = None,
        harness: str | None = None,
    ) -> list[BenchmarkCatalogEntry]:
        items = self.benchmarks
        if dimension:
            items = [item for item in items if item.dimension == dimension]
        if modality:
            items = [item for item in items if modality in item.modalities]
        if harness:
            items = [item for item in items if harness in item.harnesses]
        return items

    def summary(self) -> dict[str, Any]:
        dimensions: dict[str, int] = {}
        modalities: dict[str, int] = {}
        harnesses: dict[str, int] = {}
        families: dict[str, int] = {}
        for item in self.benchmarks:
            dimensions[item.dimension] = dimensions.get(item.dimension, 0) + 1
            families[item.family] = families.get(item.family, 0) + 1
            for modality in item.modalities:
                modalities[modality] = modalities.get(modality, 0) + 1
            for harness in item.harnesses:
                harnesses[harness] = harnesses.get(harness, 0) + 1
        return {
            "version": self.version,
            "benchmark_count": len(self.benchmarks),
            "dimensions": dict(sorted(dimensions.items())),
            "families": dict(sorted(families.items())),
            "modalities": dict(sorted(modalities.items())),
            "harnesses": dict(sorted(harnesses.items())),
        }
