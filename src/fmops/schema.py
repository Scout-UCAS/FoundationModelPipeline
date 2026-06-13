from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_json


@dataclass(frozen=True)
class SchemaField:
    name: str
    expected_type: type | tuple[type, ...]
    required: bool = True
    minimum: float | None = None
    allowed_values: tuple[Any, ...] | None = None

    def validate(self, payload: dict[str, Any], context: str) -> list[str]:
        if self.name not in payload:
            return [f"{context}: missing required key '{self.name}'"] if self.required else []
        value = payload[self.name]
        if not isinstance(value, self.expected_type):
            expected = (
                " | ".join(item.__name__ for item in self.expected_type)
                if isinstance(self.expected_type, tuple)
                else self.expected_type.__name__
            )
            return [f"{context}.{self.name}: expected {expected}, got {type(value).__name__}"]

        issues: list[str] = []
        if self.minimum is not None and isinstance(value, (int, float)) and value < self.minimum:
            issues.append(f"{context}.{self.name}: expected >= {self.minimum}, got {value}")
        if self.allowed_values is not None and value not in self.allowed_values:
            issues.append(f"{context}.{self.name}: expected one of {self.allowed_values}, got {value!r}")
        return issues


@dataclass(frozen=True)
class ObjectSchema:
    name: str
    version: str
    fields: tuple[SchemaField, ...]

    def validate(self, payload: dict[str, Any], context: str | None = None) -> list[str]:
        ctx = context or self.name
        issues: list[str] = []
        for field in self.fields:
            issues.extend(field.validate(payload, ctx))
        return issues


SCHEMAS: dict[str, ObjectSchema] = {
    "data_manifest": ObjectSchema(
        "data_manifest",
        "1.0",
        (
            SchemaField("target", dict),
            SchemaField("sources", list),
            SchemaField("processing_stages", list),
            SchemaField("mixture_stages", list),
        ),
    ),
    "architecture_experiments": ObjectSchema(
        "architecture_experiments",
        "1.0",
        (
            SchemaField("unified_setup", dict),
            SchemaField("required_families", list),
            SchemaField("candidates", list),
        ),
    ),
    "training_pipeline": ObjectSchema(
        "training_pipeline",
        "1.0",
        (
            SchemaField("target_gpus", int, minimum=1),
            SchemaField("hardware", dict),
            SchemaField("stages", list),
            SchemaField("integration", dict),
        ),
    ),
    "evaluation_suite": ObjectSchema(
        "evaluation_suite",
        "1.0",
        (
            SchemaField("required_dimensions", list),
            SchemaField("benchmarks", list),
        ),
    ),
    "datasets_catalog": ObjectSchema(
        "datasets_catalog",
        "1.0",
        (
            SchemaField("version", str),
            SchemaField("datasets", list),
        ),
    ),
    "benchmark_catalog": ObjectSchema(
        "benchmark_catalog",
        "1.0",
        (
            SchemaField("version", str),
            SchemaField("benchmarks", list),
        ),
    ),
    "production_integration": ObjectSchema(
        "production_integration",
        "1.0",
        (
            SchemaField("artifact_root", str),
            SchemaField("execution", dict),
            SchemaField("tasks", list),
            SchemaField("release_gates", list),
        ),
    ),
}


def infer_schema_name(path: str | Path) -> str:
    stem = Path(path).stem
    if stem not in SCHEMAS:
        raise ValueError(f"no schema registered for {stem}")
    return stem


def validate_config_file(path: str | Path, schema_name: str | None = None) -> list[str]:
    payload = load_json(path)
    name = schema_name or infer_schema_name(path)
    schema = SCHEMAS[name]
    issues = schema.validate(payload)
    version = payload.get("schema_version")
    if version is not None and str(version) != schema.version:
        issues.append(f"{name}.schema_version: expected {schema.version}, got {version}")
    return issues


def validate_config_dir(config_dir: str | Path) -> dict[str, list[str]]:
    base = Path(config_dir)
    targets = {
        "data_manifest": base / "data_manifest.json",
        "architecture_experiments": base / "architecture_experiments.json",
        "training_pipeline": base / "training_pipeline.json",
        "evaluation_suite": base / "evaluation_suite.json",
        "datasets_catalog": base / "datasets_catalog.json",
        "benchmark_catalog": base / "benchmark_catalog.json",
        "production_integration": base / "production_integration.json",
    }
    return {
        name: validate_config_file(path, name) if path.exists() else [f"missing config file: {path}"]
        for name, path in targets.items()
    }


def migrate_config_payload(payload: dict[str, Any], target_version: str = "1.0") -> dict[str, Any]:
    migrated = dict(payload)
    migrated.setdefault("schema_version", target_version)
    return migrated
