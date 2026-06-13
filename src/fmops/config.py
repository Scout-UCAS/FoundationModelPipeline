from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


class ConfigError(ValueError):
    """Raised when a configuration file cannot be loaded or parsed."""


def load_json(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"missing config file: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {config_path}: {exc}") from exc

    if not isinstance(value, dict):
        raise ConfigError(f"{config_path} must contain a JSON object")
    return value


def require_keys(payload: dict[str, Any], keys: Iterable[str], context: str) -> list[str]:
    return [f"{context}: missing required key '{key}'" for key in keys if key not in payload]


def ensure_probability_map(values: dict[str, Any], context: str, tolerance: float = 1e-6) -> list[str]:
    issues: list[str] = []
    total = 0.0
    for key, value in values.items():
        if not isinstance(value, (int, float)):
            issues.append(f"{context}.{key}: expected numeric probability")
            continue
        if value < 0:
            issues.append(f"{context}.{key}: probability must be non-negative")
        total += float(value)
    if abs(total - 1.0) > tolerance:
        issues.append(f"{context}: probabilities must sum to 1.0, got {total:.6f}")
    return issues


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"

