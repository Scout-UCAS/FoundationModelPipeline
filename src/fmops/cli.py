from __future__ import annotations

import argparse
import json
from pathlib import Path

from .checkpoint import CheckpointConverter
from .benchmark_catalog import BenchmarkCatalog
from .dashboard import DashboardBuilder
from .data_pipeline import DataPipelineRunner
from .dataset_catalog import DatasetCatalog
from .deployment import DeploymentValidator
from .evaluation_runner import EvaluationRunner
from .pipeline import load_platform
from .plugins import PluginManager
from .production import ProductionIntegration
from .registry import MODEL_REGISTRY
from .schema import validate_config_dir
from .tracking import ExperimentTracker
from .training_runner import TrainingRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Foundation model operations control plane")
    parser.add_argument("--config-dir", default="configs", help="Directory containing JSON configuration files")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("validate", help="Validate data, architecture, training, and evaluation configs")
    subparsers.add_parser("data-plan", help="Print the data system plan as JSON")
    subparsers.add_parser("arch-compare", help="Print architecture comparison table as JSON")
    subparsers.add_parser("train-plan", help="Print training launch plan as JSON")
    subparsers.add_parser("eval-plan", help="Print evaluation dimension weights as JSON")
    subparsers.add_parser("schema-validate", help="Validate config files against registered schemas")
    subparsers.add_parser("registry", help="Print registered framework components")

    catalog_parser = subparsers.add_parser("datasets", help="Print dataset catalog summary or entries")
    catalog_parser.add_argument("--family", help="Filter by dataset family")
    catalog_parser.add_argument("--modality", help="Filter by modality")
    catalog_parser.add_argument("--priority", choices=["P0", "P1", "P2"], help="Filter by priority")

    data_run_parser = subparsers.add_parser("data-run", help="Write a dry-run data pipeline artifact")
    data_run_parser.add_argument("--output", default="artifacts/runs/data_pipeline_plan.json")

    train_run_parser = subparsers.add_parser("train-run", help="Write a dry-run training artifact")
    train_run_parser.add_argument("--stage", help="Optional training stage name")
    train_run_parser.add_argument("--output", default="artifacts/runs/training_plan.json")

    eval_run_parser = subparsers.add_parser("eval-run", help="Run configured evaluation suite and write JSON report")
    eval_run_parser.add_argument("--model-id", default="reference-model")
    eval_run_parser.add_argument("--output", default="artifacts/runs/evaluation_report.json")
    eval_run_parser.add_argument("--samples-dir", help="Directory of JSONL evaluation samples")
    eval_run_parser.add_argument("--predictions", help="JSON/JSONL prediction file keyed by sample id")
    eval_run_parser.add_argument("--model-command", help="External command that reads one sample JSON from stdin")
    eval_run_parser.add_argument("--model-endpoint", help="HTTP endpoint that accepts one sample JSON per request")
    eval_run_parser.add_argument("--benchmark", action="append", help="Benchmark name or dimension to evaluate")
    eval_run_parser.add_argument("--fail-on-gate", action="store_true", help="Return non-zero if any gate fails")

    deploy_parser = subparsers.add_parser("deploy-check", help="Run deployment envelope checks")
    deploy_parser.add_argument("--output", default="artifacts/runs/deployment_report.json")

    checkpoint_parser = subparsers.add_parser("checkpoint-convert", help="Write a checkpoint conversion manifest")
    checkpoint_parser.add_argument("--source", required=True)
    checkpoint_parser.add_argument("--target", required=True)
    checkpoint_parser.add_argument("--source-format", default="training")
    checkpoint_parser.add_argument("--target-format", default="inference")
    checkpoint_parser.add_argument("--copy-files", action="store_true")

    track_parser = subparsers.add_parser("track-run", help="Create an experiment run manifest")
    track_parser.add_argument("--name", default="manual-run")
    track_parser.add_argument("--kind", default="manual")
    track_parser.add_argument("--root", default="artifacts/runs")

    subparsers.add_parser("plugins", help="List and validate local plugins")

    benchmark_parser = subparsers.add_parser("benchmarks", help="Print benchmark catalog summary or entries")
    benchmark_parser.add_argument("--dimension", help="Filter by evaluation dimension")
    benchmark_parser.add_argument("--modality", help="Filter by modality")
    benchmark_parser.add_argument("--harness", help="Filter by harness")

    dashboard_parser = subparsers.add_parser("dashboard", help="Generate a static HTML dashboard")
    dashboard_parser.add_argument("--output", default="reports/dashboard.html")

    report_parser = subparsers.add_parser("report", help="Generate a Markdown program report")
    report_parser.add_argument("--output", default="reports/foundation_model_plan.md", help="Markdown report path")

    production_plan_parser = subparsers.add_parser("production-plan", help="Write production integration execution plan")
    production_plan_parser.add_argument("--area", help="Optional production area filter")
    production_plan_parser.add_argument("--output", default="artifacts/production/production_plan.json")

    production_check_parser = subparsers.add_parser("production-check", help="Check production adapter dependencies")
    production_check_parser.add_argument("--area", help="Optional production area filter")
    production_check_parser.add_argument("--output", default="artifacts/production/preflight_report.json")

    production_run_parser = subparsers.add_parser("production-run", help="Run or plan production adapter tasks")
    production_run_parser.add_argument("--area", help="Optional production area filter")
    production_run_parser.add_argument("--execute", action="store_true", help="Execute external commands after guardrail checks")
    production_run_parser.add_argument("--output", default="artifacts/production/execution_report.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    platform = load_platform(args.config_dir)

    if args.command == "validate":
        validation = platform.validate()
        issues = [issue for section in validation.values() for issue in section]
        if issues:
            print(json.dumps(validation, indent=2, ensure_ascii=False))
            return 1
        print("OK: all foundation model program configs are valid")
        return 0

    if args.command == "data-plan":
        print(json.dumps({"summary": platform.data.summary(), "mixture_plan": platform.data.mixture_plan()}, indent=2, ensure_ascii=False))
        return 0

    if args.command == "arch-compare":
        print(json.dumps(platform.architectures.comparison_table(), indent=2, ensure_ascii=False))
        return 0

    if args.command == "train-plan":
        print(json.dumps(platform.training.launch_commands(), indent=2, ensure_ascii=False))
        return 0

    if args.command == "eval-plan":
        print(json.dumps(platform.evaluation.dimension_weights(), indent=2, ensure_ascii=False))
        return 0

    if args.command == "schema-validate":
        validation = validate_config_dir(args.config_dir)
        issues = [issue for section in validation.values() for issue in section]
        print(json.dumps(validation, indent=2, ensure_ascii=False))
        return 1 if issues else 0

    if args.command == "registry":
        print(json.dumps({"models": MODEL_REGISTRY.describe()}, indent=2, ensure_ascii=False))
        return 0

    if args.command == "datasets":
        catalog = DatasetCatalog.from_file(Path(args.config_dir) / "datasets_catalog.json")
        issues = catalog.validate()
        if issues:
            print(json.dumps({"issues": issues}, indent=2, ensure_ascii=False))
            return 1
        entries = catalog.filter(family=args.family, modality=args.modality, priority=args.priority)
        payload = {
            "summary": catalog.summary(),
            "datasets": [
                {
                    "name": item.name,
                    "family": item.family,
                    "modalities": item.modalities,
                    "priority": item.priority,
                    "download_url": item.download_url,
                    "risks": item.risks,
                }
                for item in entries
            ],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if args.command == "data-run":
        catalog = DatasetCatalog.from_file(Path(args.config_dir) / "datasets_catalog.json")
        path = DataPipelineRunner(platform.data, catalog).run_dry(args.output)
        print(f"Wrote {path}")
        return 0

    if args.command == "train-run":
        path = TrainingRunner(platform.training).dry_run(args.output, stage_name=args.stage)
        print(f"Wrote {path}")
        return 0

    if args.command == "eval-run":
        path = EvaluationRunner(
            platform.evaluation,
            samples_dir=args.samples_dir,
            predictions_path=args.predictions,
            model_command=args.model_command,
            model_endpoint=args.model_endpoint,
            benchmark_filter=set(args.benchmark or []),
        ).write_report(args.output, model_id=args.model_id)
        print(f"Wrote {path}")
        if args.fail_on_gate:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            return 1 if payload["summary"]["failed"] else 0
        return 0

    if args.command == "deploy-check":
        path = DeploymentValidator.default().write_report(args.output)
        print(f"Wrote {path}")
        return 0

    if args.command == "checkpoint-convert":
        manifest = CheckpointConverter().convert(
            args.source,
            args.target,
            source_format=args.source_format,
            target_format=args.target_format,
            copy_files=args.copy_files,
        )
        print(json.dumps(manifest.__dict__, indent=2, ensure_ascii=False))
        return 0

    if args.command == "track-run":
        tracker = ExperimentTracker(args.root)
        manifest = tracker.complete_run(tracker.start_run(args.name, args.kind))
        print(json.dumps(manifest.__dict__, indent=2, ensure_ascii=False))
        return 0

    if args.command == "plugins":
        manager = PluginManager()
        manifests = manager.discover()
        issues = manager.validate()
        print(
            json.dumps(
                {
                    "plugins": [manifest.__dict__ for manifest in manifests],
                    "issues": issues,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 1 if issues else 0

    if args.command == "benchmarks":
        catalog = BenchmarkCatalog.from_file(Path(args.config_dir) / "benchmark_catalog.json")
        issues = catalog.validate()
        if issues:
            print(json.dumps({"issues": issues}, indent=2, ensure_ascii=False))
            return 1
        entries = catalog.filter(dimension=args.dimension, modality=args.modality, harness=args.harness)
        print(
            json.dumps(
                {
                    "summary": catalog.summary(),
                    "benchmarks": [entry.__dict__ for entry in entries],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "dashboard":
        path = DashboardBuilder(platform).write(args.output)
        print(f"Wrote {path}")
        return 0

    if args.command == "report":
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(platform.to_markdown(), encoding="utf-8")
        print(f"Wrote {output}")
        return 0

    if args.command == "production-plan":
        integration = ProductionIntegration.from_file(Path(args.config_dir) / "production_integration.json")
        issues = integration.validate()
        if issues:
            print(json.dumps({"issues": issues}, indent=2, ensure_ascii=False))
            return 1
        path = integration.write_plan(args.output, config_dir=args.config_dir, area=args.area)
        print(f"Wrote {path}")
        return 0

    if args.command == "production-check":
        integration = ProductionIntegration.from_file(Path(args.config_dir) / "production_integration.json")
        issues = integration.validate()
        if issues:
            print(json.dumps({"issues": issues}, indent=2, ensure_ascii=False))
            return 1
        path = integration.write_preflight(args.output, config_dir=args.config_dir, area=args.area)
        print(f"Wrote {path}")
        return 0

    if args.command == "production-run":
        integration = ProductionIntegration.from_file(Path(args.config_dir) / "production_integration.json")
        issues = integration.validate()
        if issues:
            print(json.dumps({"issues": issues}, indent=2, ensure_ascii=False))
            return 1
        path = integration.run(args.output, config_dir=args.config_dir, area=args.area, execute=args.execute)
        print(f"Wrote {path}")
        return 0

    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
