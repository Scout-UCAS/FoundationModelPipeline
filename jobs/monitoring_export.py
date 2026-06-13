from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fmops.pipeline import load_platform


def build_payload(config_dir: Path) -> dict[str, object]:
    platform = load_platform(config_dir)
    alert_rules = [
        {"name": "data_ingestion_lag", "expr": "fmops_data_ingestion_lag_hours > 6", "severity": "warning"},
        {"name": "data_quality_drop", "expr": "fmops_data_quality_score < 0.82", "severity": "critical"},
        {"name": "dedup_rate_spike", "expr": "fmops_duplicate_rate > 0.18", "severity": "critical"},
        {"name": "training_loss_spike", "expr": "fmops_loss_spike_ratio > 1.35", "severity": "critical"},
        {"name": "training_nan_detected", "expr": "fmops_nan_incidents > 0", "severity": "critical"},
        {"name": "gpu_utilization_low", "expr": "fmops_gpu_utilization < 0.88", "severity": "warning"},
        {"name": "moe_expert_imbalance", "expr": "fmops_expert_balance < 0.80", "severity": "warning"},
        {"name": "rl_kl_high", "expr": "fmops_rl_kl > 0.12", "severity": "critical"},
        {"name": "tool_error_rate_high", "expr": "fmops_tool_error_rate > 0.08", "severity": "warning"},
        {"name": "eval_gate_failed", "expr": "fmops_eval_gate_passed == 0", "severity": "critical"},
        {"name": "serving_latency_high", "expr": "fmops_prefill_latency_ms > 180", "severity": "warning"},
        {"name": "edge_power_high", "expr": "fmops_edge_power_w > 60", "severity": "critical"},
        {"name": "release_gate_blocked", "expr": "fmops_release_gate_blocked > 0", "severity": "critical"},
    ]
    return {
        "job": "monitoring_export",
        "status": "exported",
        "platform": {
            "data": platform.data.summary(),
            "training_gpus": platform.training.hardware.total_gpus,
            "evaluation_dimensions": platform.evaluation.dimension_weights(),
        },
        "prometheus_rules": alert_rules,
        "grafana_dashboards": [
            "foundation-model-data",
            "foundation-model-training",
            "foundation-model-evaluation",
            "foundation-model-serving",
            "foundation-model-governance",
        ],
        "slo": {
            "data_pipeline_success": 0.995,
            "training_job_recovery_minutes": 30,
            "eval_report_freshness_hours": 24,
            "deployment_rollback_minutes": 15,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export production monitoring bundle")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_payload(Path(args.config_dir)), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
