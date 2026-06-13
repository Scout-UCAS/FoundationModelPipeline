from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fmops.production import ProductionIntegration


def path_exists(path_text: str) -> bool:
    if path_text.startswith("artifact://") or path_text.startswith("$"):
        return False
    return Path(path_text).exists()


def build_payload(production_config: Path, artifact_root: str) -> dict[str, object]:
    integration = ProductionIntegration.from_file(production_config)
    gates = []
    for gate in integration.release_gates:
        source = gate.source.replace("artifacts/production", artifact_root)
        gates.append(
            {
                "name": gate.name,
                "source": source,
                "condition": gate.condition,
                "owner": gate.owner,
                "source_present": path_exists(source),
                "status": "pending_external_evidence" if not path_exists(source) else "ready_for_review",
            }
        )
    blocked = [gate for gate in gates if gate["status"] != "ready_for_review"]
    return {
        "job": "release_gate",
        "status": "blocked" if blocked else "ready_for_promotion",
        "summary": {
            "gate_count": len(gates),
            "ready": len(gates) - len(blocked),
            "blocked": len(blocked),
        },
        "gates": gates,
        "required_signoffs": ["data-governance", "training-platform", "evaluation-platform", "serving-platform", "safety", "release"],
        "rollback_plan": {
            "checkpoint_alias_previous": "production-previous",
            "checkpoint_alias_candidate": "production-candidate",
            "rollback_slo_minutes": 15,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate production release gates")
    parser.add_argument("--production-config", required=True)
    parser.add_argument("--artifact-root", default="artifacts/production")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_payload(Path(args.production_config), args.artifact_root), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
