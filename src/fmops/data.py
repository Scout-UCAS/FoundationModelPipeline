from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ensure_probability_map, load_json, percent, require_keys


REQUIRED_MODALITIES = {"pure_text", "multimodal", "video_pretraining", "vla"}
REQUIRED_PROCESSING_CAPABILITIES = {
    "clean": ("clean", "filter", "redaction", "repair"),
    "deduplicate": ("dedup",),
    "cluster": ("cluster", "taxonomy"),
    "quality": ("quality", "score", "perplexity"),
}


@dataclass(frozen=True)
class DataSource:
    name: str
    source_type: str
    modalities: tuple[str, ...]
    languages: tuple[str, ...]
    size_tb: float
    quality_score: float
    duplicate_rate: float
    license: str
    tags: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DataSource":
        modalities = payload.get("modalities", [])
        languages = payload.get("languages", [])
        tags = payload.get("tags", [])
        return cls(
            name=str(payload["name"]),
            source_type=str(payload["source_type"]),
            modalities=tuple(str(item) for item in modalities),
            languages=tuple(str(item) for item in languages),
            size_tb=float(payload["size_tb"]),
            quality_score=float(payload.get("quality_score", 0.0)),
            duplicate_rate=float(payload.get("duplicate_rate", 1.0)),
            license=str(payload.get("license", "unknown")),
            tags=tuple(str(item) for item in tags),
        )


@dataclass(frozen=True)
class ProcessingStage:
    name: str
    operations: tuple[str, ...]
    output: str
    owner: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProcessingStage":
        return cls(
            name=str(payload["name"]),
            operations=tuple(str(item) for item in payload.get("operations", [])),
            output=str(payload.get("output", "")),
            owner=str(payload.get("owner", "data-platform")),
        )


@dataclass(frozen=True)
class MixtureStage:
    name: str
    total_tb: float
    objective: str
    modality_targets: dict[str, float]
    language_targets: dict[str, float]
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MixtureStage":
        return cls(
            name=str(payload["name"]),
            total_tb=float(payload["total_tb"]),
            objective=str(payload.get("objective", "")),
            modality_targets={str(key): float(value) for key, value in payload.get("modality_targets", {}).items()},
            language_targets={str(key): float(value) for key, value in payload.get("language_targets", {}).items()},
            notes=str(payload.get("notes", "")),
        )


class DataSystem:
    def __init__(
        self,
        target: dict[str, Any],
        sources: list[DataSource],
        processing_stages: list[ProcessingStage],
        mixture_stages: list[MixtureStage],
    ) -> None:
        self.target = target
        self.sources = sources
        self.processing_stages = processing_stages
        self.mixture_stages = mixture_stages

    @classmethod
    def from_file(cls, path: str | Path) -> "DataSystem":
        payload = load_json(path)
        sources = [DataSource.from_dict(item) for item in payload.get("sources", [])]
        stages = [ProcessingStage.from_dict(item) for item in payload.get("processing_stages", [])]
        mixtures = [MixtureStage.from_dict(item) for item in payload.get("mixture_stages", [])]
        return cls(target=payload.get("target", {}), sources=sources, processing_stages=stages, mixture_stages=mixtures)

    @property
    def total_size_tb(self) -> float:
        return sum(source.size_tb for source in self.sources)

    @property
    def languages(self) -> set[str]:
        return {language for source in self.sources for language in source.languages}

    @property
    def modalities(self) -> set[str]:
        return {modality for source in self.sources for modality in source.modalities}

    def size_by_modality(self) -> dict[str, float]:
        totals = {modality: 0.0 for modality in sorted(self.modalities)}
        for source in self.sources:
            if not source.modalities:
                continue
            share = source.size_tb / len(source.modalities)
            for modality in source.modalities:
                totals[modality] = totals.get(modality, 0.0) + share
        return totals

    def validate(self) -> list[str]:
        issues: list[str] = []
        issues.extend(require_keys(self.target, ["total_tb", "minimum_languages", "required_modalities"], "data.target"))

        target_total = float(self.target.get("total_tb", 0.0))
        if self.total_size_tb < target_total:
            issues.append(f"data.sources: total size {self.total_size_tb:.1f} TB is below target {target_total:.1f} TB")

        minimum_languages = int(self.target.get("minimum_languages", 0))
        if len(self.languages) < minimum_languages:
            issues.append(f"data.sources: language count {len(self.languages)} is below target {minimum_languages}")

        required_modalities = set(self.target.get("required_modalities", REQUIRED_MODALITIES))
        missing_modalities = required_modalities - self.modalities
        if missing_modalities:
            issues.append(f"data.sources: missing modalities {sorted(missing_modalities)}")

        quality_gates = self.target.get("quality_gates", {})
        min_quality = float(quality_gates.get("min_quality_score", 0.0))
        max_duplicate_rate = float(quality_gates.get("max_duplicate_rate", 1.0))

        for index, source in enumerate(self.sources):
            context = f"data.sources[{index}]"
            if source.size_tb <= 0:
                issues.append(f"{context}.{source.name}: size_tb must be positive")
            if not source.languages:
                issues.append(f"{context}.{source.name}: languages must not be empty")
            if not source.modalities:
                issues.append(f"{context}.{source.name}: modalities must not be empty")
            if source.quality_score < min_quality:
                issues.append(
                    f"{context}.{source.name}: quality_score {source.quality_score:.3f} is below gate {min_quality:.3f}"
                )
            if source.duplicate_rate > max_duplicate_rate:
                issues.append(
                    f"{context}.{source.name}: duplicate_rate {source.duplicate_rate:.3f} exceeds gate {max_duplicate_rate:.3f}"
                )

        operations_text = " ".join(
            operation.lower()
            for stage in self.processing_stages
            for operation in (stage.name, *stage.operations, stage.output)
        )
        for capability, needles in REQUIRED_PROCESSING_CAPABILITIES.items():
            if not any(needle in operations_text for needle in needles):
                issues.append(f"data.processing_stages: missing {capability} capability")

        for index, stage in enumerate(self.mixture_stages):
            context = f"data.mixture_stages[{index}].{stage.name}"
            if stage.total_tb <= 0:
                issues.append(f"{context}: total_tb must be positive")
            issues.extend(ensure_probability_map(stage.modality_targets, f"{context}.modality_targets"))
            issues.extend(ensure_probability_map(stage.language_targets, f"{context}.language_targets"))

        return issues

    def mixture_plan(self) -> list[dict[str, Any]]:
        plan: list[dict[str, Any]] = []
        for stage in self.mixture_stages:
            plan.append(
                {
                    "name": stage.name,
                    "objective": stage.objective,
                    "total_tb": stage.total_tb,
                    "modalities": {
                        modality: round(stage.total_tb * ratio, 3)
                        for modality, ratio in sorted(stage.modality_targets.items())
                    },
                    "languages": {
                        language: round(stage.total_tb * ratio, 3)
                        for language, ratio in sorted(stage.language_targets.items())
                    },
                }
            )
        return plan

    def summary(self) -> dict[str, Any]:
        return {
            "target_tb": float(self.target.get("total_tb", 0.0)),
            "actual_tb": round(self.total_size_tb, 3),
            "language_count": len(self.languages),
            "modalities": sorted(self.modalities),
            "size_by_modality_tb": {key: round(value, 3) for key, value in self.size_by_modality().items()},
            "source_count": len(self.sources),
            "processing_stage_count": len(self.processing_stages),
            "mixture_stage_count": len(self.mixture_stages),
        }

    def to_markdown(self) -> str:
        summary = self.summary()
        lines = [
            "## Data System",
            "",
            f"- Target scale: {summary['target_tb']:.0f} TB",
            f"- Planned scale: {summary['actual_tb']:.0f} TB",
            f"- Languages: {summary['language_count']}",
            f"- Modalities: {', '.join(summary['modalities'])}",
            "",
            "| Modality | Approx TB |",
            "| --- | ---: |",
        ]
        for modality, size_tb in summary["size_by_modality_tb"].items():
            lines.append(f"| {modality} | {size_tb:.1f} |")

        lines.extend(["", "| Mixture Stage | Objective | TB | Modality Split | Language Split |", "| --- | --- | ---: | --- | --- |"])
        for stage in self.mixture_stages:
            modality_split = ", ".join(
                f"{name} {percent(value)}" for name, value in sorted(stage.modality_targets.items())
            )
            language_split = ", ".join(
                f"{name} {percent(value)}" for name, value in sorted(stage.language_targets.items())
            )
            lines.append(f"| {stage.name} | {stage.objective} | {stage.total_tb:.0f} | {modality_split} | {language_split} |")
        return "\n".join(lines)

