from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import load_json


REQUIRED_PRODUCTION_AREAS = (
    "data",
    "training",
    "evaluation",
    "checkpoint",
    "deployment",
    "monitoring",
    "governance",
)

BLOCKED_COMMAND_TOKENS = ("rm -rf /", "shutdown", "reboot", "mkfs", ":(){")


class SafeFormatDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


@dataclass(frozen=True)
class ProductionTask:
    name: str
    area: str
    adapter: str
    command: str
    required_binaries: tuple[str, ...]
    required_env: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    gates: dict[str, Any]
    owner: str
    timeout_seconds: int
    description: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProductionTask":
        return cls(
            name=str(payload["name"]),
            area=str(payload["area"]),
            adapter=str(payload.get("adapter", "external-command")),
            command=str(payload["command"]),
            required_binaries=tuple(str(item) for item in payload.get("required_binaries", [])),
            required_env=tuple(str(item) for item in payload.get("required_env", [])),
            inputs=tuple(str(item) for item in payload.get("inputs", [])),
            outputs=tuple(str(item) for item in payload.get("outputs", [])),
            gates=dict(payload.get("gates", {})),
            owner=str(payload.get("owner", "platform")),
            timeout_seconds=int(payload.get("timeout_seconds", 3600)),
            description=str(payload.get("description", "")),
        )

    def render_command(self, context: dict[str, Any]) -> str:
        return self.command.format_map(SafeFormatDict(context))


@dataclass(frozen=True)
class ReleaseGate:
    name: str
    source: str
    condition: str
    owner: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReleaseGate":
        return cls(
            name=str(payload["name"]),
            source=str(payload["source"]),
            condition=str(payload["condition"]),
            owner=str(payload.get("owner", "release")),
        )


@dataclass(frozen=True)
class PreflightCheck:
    task: str
    area: str
    adapter: str
    status: str
    missing_binaries: tuple[str, ...]
    missing_env: tuple[str, ...]
    command: str


@dataclass(frozen=True)
class ExecutionRecord:
    task: str
    area: str
    adapter: str
    status: str
    command: str
    exit_code: int | None
    duration_seconds: float
    stdout_path: str | None
    stderr_path: str | None
    message: str


class ProductionIntegration:
    def __init__(
        self,
        artifact_root: str,
        execution: dict[str, Any],
        tasks: list[ProductionTask],
        release_gates: list[ReleaseGate],
    ) -> None:
        self.artifact_root = artifact_root
        self.execution = execution
        self.tasks = tasks
        self.release_gates = release_gates

    @classmethod
    def from_file(cls, path: str | Path) -> "ProductionIntegration":
        payload = load_json(path)
        tasks = [ProductionTask.from_dict(item) for item in payload.get("tasks", [])]
        gates = [ReleaseGate.from_dict(item) for item in payload.get("release_gates", [])]
        return cls(
            artifact_root=str(payload.get("artifact_root", "artifacts/production")),
            execution=dict(payload.get("execution", {})),
            tasks=tasks,
            release_gates=gates,
        )

    def context(self, config_dir: str | Path = "configs") -> dict[str, Any]:
        return {
            "artifact_root": self.artifact_root,
            "config_dir": str(config_dir),
            "python": sys.executable,
            "repo_root": str(Path.cwd()),
        }

    def validate(self) -> list[str]:
        issues: list[str] = []
        if not self.tasks:
            issues.append("production.tasks: must not be empty")

        covered = {task.area for task in self.tasks}
        for area in REQUIRED_PRODUCTION_AREAS:
            if area not in covered:
                issues.append(f"production.tasks: missing required area '{area}'")

        seen: set[tuple[str, str]] = set()
        for index, task in enumerate(self.tasks):
            context = f"production.tasks[{index}].{task.name}"
            key = (task.area, task.name)
            if key in seen:
                issues.append(f"{context}: duplicate task name in area")
            seen.add(key)
            if not task.command.strip():
                issues.append(f"{context}: command must not be empty")
            if task.timeout_seconds <= 0:
                issues.append(f"{context}: timeout_seconds must be positive")
            lowered = task.command.lower()
            if any(token in lowered for token in BLOCKED_COMMAND_TOKENS):
                issues.append(f"{context}: command contains blocked destructive token")
            if not task.outputs:
                issues.append(f"{context}: outputs must not be empty")
            for env_name in task.required_env:
                if not env_name.replace("_", "").isalnum():
                    issues.append(f"{context}: invalid env var name '{env_name}'")

        if not self.release_gates:
            issues.append("production.release_gates: must not be empty")
        return issues

    def selected_tasks(self, area: str | None = None) -> list[ProductionTask]:
        if area is None:
            return list(self.tasks)
        return [task for task in self.tasks if task.area == area]

    def plan(self, config_dir: str | Path = "configs", area: str | None = None) -> dict[str, Any]:
        context = self.context(config_dir)
        tasks = self.selected_tasks(area)
        return {
            "artifact_root": self.artifact_root,
            "execution": self.execution,
            "summary": {
                "task_count": len(tasks),
                "areas": sorted({task.area for task in tasks}),
                "release_gate_count": len(self.release_gates),
            },
            "tasks": [
                {
                    **asdict(task),
                    "command": task.render_command(context),
                }
                for task in tasks
            ],
            "release_gates": [asdict(gate) for gate in self.release_gates],
            "validation_issues": self.validate(),
        }

    def write_plan(self, output: str | Path, config_dir: str | Path = "configs", area: str | None = None) -> Path:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.plan(config_dir=config_dir, area=area), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def to_markdown(self) -> str:
        lines = [
            "## Production Integration",
            "",
            f"- Artifact root: `{self.artifact_root}`",
            f"- Execution guard: `{self.execution.get('production_execute_env', 'FMOPS_ALLOW_PRODUCTION_EXECUTE')}`",
            f"- Tasks: {len(self.tasks)}",
            f"- Release gates: {len(self.release_gates)}",
            "",
            "| Area | Task | Adapter | Owner | Required Binaries | Required Env |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for task in self.tasks:
            binaries = ", ".join(task.required_binaries) or "-"
            env = ", ".join(task.required_env) or "-"
            lines.append(f"| {task.area} | {task.name} | {task.adapter} | {task.owner} | {binaries} | {env} |")

        lines.extend(["", "| Release Gate | Source | Owner |", "| --- | --- | --- |"])
        for gate in self.release_gates:
            lines.append(f"| {gate.name} | `{gate.source}` | {gate.owner} |")
        return "\n".join(lines)

    def preflight(self, config_dir: str | Path = "configs", area: str | None = None) -> list[PreflightCheck]:
        context = self.context(config_dir)
        checks: list[PreflightCheck] = []
        for task in self.selected_tasks(area):
            missing_binaries = tuple(binary for binary in task.required_binaries if shutil.which(binary) is None)
            missing_env = tuple(env_name for env_name in task.required_env if not os.environ.get(env_name))
            status = "ready" if not missing_binaries and not missing_env else "blocked"
            checks.append(
                PreflightCheck(
                    task=task.name,
                    area=task.area,
                    adapter=task.adapter,
                    status=status,
                    missing_binaries=missing_binaries,
                    missing_env=missing_env,
                    command=task.render_command(context),
                )
            )
        return checks

    def write_preflight(self, output: str | Path, config_dir: str | Path = "configs", area: str | None = None) -> Path:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        checks = self.preflight(config_dir=config_dir, area=area)
        payload = {
            "summary": {
                "task_count": len(checks),
                "ready": sum(1 for check in checks if check.status == "ready"),
                "blocked": sum(1 for check in checks if check.status != "ready"),
            },
            "checks": [asdict(check) for check in checks],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def run(
        self,
        output: str | Path,
        config_dir: str | Path = "configs",
        area: str | None = None,
        execute: bool = False,
    ) -> Path:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        records = self._run_records(config_dir=config_dir, area=area, execute=execute, report_path=path)
        payload = {
            "mode": "execute" if execute else "plan",
            "summary": {
                "task_count": len(records),
                "succeeded": sum(1 for item in records if item.status == "succeeded"),
                "planned": sum(1 for item in records if item.status == "planned"),
                "blocked": sum(1 for item in records if item.status == "blocked"),
                "failed": sum(1 for item in records if item.status == "failed"),
            },
            "records": [asdict(record) for record in records],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def _run_records(
        self,
        config_dir: str | Path,
        area: str | None,
        execute: bool,
        report_path: Path,
    ) -> list[ExecutionRecord]:
        context = self.context(config_dir)
        tasks = self.selected_tasks(area)
        if not execute:
            return [
                ExecutionRecord(
                    task=task.name,
                    area=task.area,
                    adapter=task.adapter,
                    status="planned",
                    command=task.render_command(context),
                    exit_code=None,
                    duration_seconds=0.0,
                    stdout_path=None,
                    stderr_path=None,
                    message="execution not requested",
                )
                for task in tasks
            ]

        guard_env = str(self.execution.get("production_execute_env", "FMOPS_ALLOW_PRODUCTION_EXECUTE"))
        if os.environ.get(guard_env) not in {"1", "true", "TRUE", "yes", "YES"}:
            return [
                ExecutionRecord(
                    task=task.name,
                    area=task.area,
                    adapter=task.adapter,
                    status="blocked",
                    command=task.render_command(context),
                    exit_code=None,
                    duration_seconds=0.0,
                    stdout_path=None,
                    stderr_path=None,
                    message=f"set {guard_env}=1 to allow production execution",
                )
                for task in tasks
            ]

        preflight_by_name = {(check.area, check.task): check for check in self.preflight(config_dir=config_dir, area=area)}
        logs_dir = report_path.parent / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        records: list[ExecutionRecord] = []
        for task in tasks:
            check = preflight_by_name[(task.area, task.name)]
            command = task.render_command(context)
            if check.status != "ready":
                records.append(
                    ExecutionRecord(
                        task=task.name,
                        area=task.area,
                        adapter=task.adapter,
                        status="blocked",
                        command=command,
                        exit_code=None,
                        duration_seconds=0.0,
                        stdout_path=None,
                        stderr_path=None,
                        message=(
                            f"missing_binaries={list(check.missing_binaries)} "
                            f"missing_env={list(check.missing_env)}"
                        ),
                    )
                )
                continue

            stdout_path = logs_dir / f"{task.area}_{task.name}.stdout.log"
            stderr_path = logs_dir / f"{task.area}_{task.name}.stderr.log"
            started = time.monotonic()
            with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
                completed = subprocess.run(
                    command,
                    shell=True,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                    timeout=task.timeout_seconds,
                    check=False,
                )
            duration = round(time.monotonic() - started, 3)
            status = "succeeded" if completed.returncode == 0 else "failed"
            records.append(
                ExecutionRecord(
                    task=task.name,
                    area=task.area,
                    adapter=task.adapter,
                    status=status,
                    command=command,
                    exit_code=completed.returncode,
                    duration_seconds=duration,
                    stdout_path=str(stdout_path),
                    stderr_path=str(stderr_path),
                    message="completed" if status == "succeeded" else "command exited non-zero",
                )
            )
        return records
