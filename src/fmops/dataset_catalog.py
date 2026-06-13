from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_json


@dataclass(frozen=True)
class DatasetEntry:
    name: str
    family: str
    modalities: tuple[str, ...]
    languages: tuple[str, ...]
    license: str
    access: str
    download_url: str
    size: str
    priority: str
    risks: tuple[str, ...]
    schema: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetEntry":
        return cls(
            name=str(payload["name"]),
            family=str(payload["family"]),
            modalities=tuple(str(item) for item in payload.get("modalities", [])),
            languages=tuple(str(item) for item in payload.get("languages", [])),
            license=str(payload.get("license", "unknown")),
            access=str(payload.get("access", "unknown")),
            download_url=str(payload.get("download_url", "")),
            size=str(payload.get("size", "unknown")),
            priority=str(payload.get("priority", "P2")),
            risks=tuple(str(item) for item in payload.get("risks", [])),
            schema=dict(payload.get("schema", {})),
        )


class DatasetCatalog:
    def __init__(self, version: str, datasets: list[DatasetEntry]) -> None:
        self.version = version
        self.datasets = datasets

    @classmethod
    def from_file(cls, path: str | Path) -> "DatasetCatalog":
        payload = load_json(path)
        datasets = [DatasetEntry.from_dict(item) for item in payload.get("datasets", [])]
        return cls(version=str(payload.get("version", "0")), datasets=datasets)

    def validate(self) -> list[str]:
        issues: list[str] = []
        seen: set[str] = set()
        for index, dataset in enumerate(self.datasets):
            context = f"datasets[{index}].{dataset.name}"
            if dataset.name in seen:
                issues.append(f"{context}: duplicate dataset name")
            seen.add(dataset.name)
            if not dataset.modalities:
                issues.append(f"{context}: modalities must not be empty")
            if not dataset.download_url.startswith(("https://", "http://", "s3://", "gs://")):
                issues.append(f"{context}: download_url must be an http(s), s3, or gs URI")
            if dataset.priority not in {"P0", "P1", "P2"}:
                issues.append(f"{context}: priority must be P0, P1, or P2")
            if not dataset.schema:
                issues.append(f"{context}: schema must not be empty")
        return issues

    def filter(
        self,
        family: str | None = None,
        modality: str | None = None,
        priority: str | None = None,
    ) -> list[DatasetEntry]:
        items = self.datasets
        if family:
            items = [item for item in items if item.family == family]
        if modality:
            items = [item for item in items if modality in item.modalities]
        if priority:
            items = [item for item in items if item.priority == priority]
        return items

    def summary(self) -> dict[str, Any]:
        families: dict[str, int] = {}
        modalities: dict[str, int] = {}
        priorities: dict[str, int] = {}
        risk_counts: dict[str, int] = {}
        for item in self.datasets:
            families[item.family] = families.get(item.family, 0) + 1
            priorities[item.priority] = priorities.get(item.priority, 0) + 1
            for modality in item.modalities:
                modalities[modality] = modalities.get(modality, 0) + 1
            for risk in item.risks:
                risk_counts[risk] = risk_counts.get(risk, 0) + 1
        return {
            "version": self.version,
            "dataset_count": len(self.datasets),
            "families": dict(sorted(families.items())),
            "modalities": dict(sorted(modalities.items())),
            "priorities": dict(sorted(priorities.items())),
            "risks": dict(sorted(risk_counts.items())),
        }

    def to_markdown(self) -> str:
        lines = [
            "## Dataset Catalog",
            "",
            "| Priority | Family | Dataset | Modalities | Access | License | Download |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for item in sorted(self.datasets, key=lambda row: (row.priority, row.family, row.name)):
            modalities = ", ".join(item.modalities)
            lines.append(
                f"| {item.priority} | {item.family} | {item.name} | {modalities} | "
                f"{item.access} | {item.license} | {item.download_url} |"
            )
        return "\n".join(lines)

