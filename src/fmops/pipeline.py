from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .architectures import ArchitectureSuite
from .data import DataSystem
from .evaluation import EvaluationSuite
from .production import ProductionIntegration
from .training import TrainingPipeline


@dataclass(frozen=True)
class Platform:
    data: DataSystem
    architectures: ArchitectureSuite
    training: TrainingPipeline
    evaluation: EvaluationSuite
    production: ProductionIntegration

    def validate(self) -> dict[str, list[str]]:
        return {
            "data": self.data.validate(),
            "architectures": self.architectures.validate(),
            "training": self.training.validate(),
            "evaluation": self.evaluation.validate(),
            "production": self.production.validate(),
        }

    def all_issues(self) -> list[str]:
        return [issue for issues in self.validate().values() for issue in issues]

    def to_markdown(self) -> str:
        issues = self.all_issues()
        status = "PASS" if not issues else "FAIL"
        lines = [
            "# Foundation Model Program Plan",
            "",
            f"Validation status: **{status}**",
            "",
        ]
        if issues:
            lines.extend(["## Validation Issues", ""])
            lines.extend(f"- {issue}" for issue in issues)
            lines.append("")
        lines.extend(
            [
                self.data.to_markdown(),
                "",
                self.architectures.to_markdown(),
                "",
                self.training.to_markdown(),
                "",
                self.evaluation.to_markdown(),
                "",
                self.production.to_markdown(),
                "",
            ]
        )
        return "\n".join(lines)


def load_platform(config_dir: str | Path) -> Platform:
    base = Path(config_dir)
    return Platform(
        data=DataSystem.from_file(base / "data_manifest.json"),
        architectures=ArchitectureSuite.from_file(base / "architecture_experiments.json"),
        training=TrainingPipeline.from_file(base / "training_pipeline.json"),
        evaluation=EvaluationSuite.from_file(base / "evaluation_suite.json"),
        production=ProductionIntegration.from_file(base / "production_integration.json"),
    )
