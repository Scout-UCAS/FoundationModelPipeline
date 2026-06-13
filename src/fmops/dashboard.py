from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .pipeline import Platform


def _json_block(payload: dict[str, Any]) -> str:
    return html.escape(json.dumps(payload, indent=2, ensure_ascii=False))


class DashboardBuilder:
    def __init__(self, platform: Platform) -> None:
        self.platform = platform

    def render(self) -> str:
        validation = self.platform.validate()
        status = "PASS" if not any(validation.values()) else "FAIL"
        data_summary = self.platform.data.summary()
        eval_weights = self.platform.evaluation.dimension_weights()
        arch_rows = self.platform.architectures.comparison_table()[:5]
        production_summary = {
            "task_count": len(self.platform.production.tasks),
            "areas": sorted({task.area for task in self.platform.production.tasks}),
            "release_gate_count": len(self.platform.production.release_gates),
            "execute_guard": self.platform.production.execution.get("production_execute_env", "FMOPS_ALLOW_PRODUCTION_EXECUTE"),
        }
        return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Foundation Model Ops Dashboard</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2937; }}
    h1, h2 {{ margin: 0 0 12px; }}
    section {{ margin: 24px 0; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    code, pre {{ background: #f9fafb; border: 1px solid #e5e7eb; padding: 12px; display: block; overflow: auto; }}
    .pass {{ color: #047857; }}
    .fail {{ color: #b91c1c; }}
  </style>
</head>
<body>
  <h1>Foundation Model Ops Dashboard</h1>
  <p>Status: <strong class="{status.lower()}">{status}</strong></p>
  <section>
    <h2>Data Summary</h2>
    <pre>{_json_block(data_summary)}</pre>
  </section>
  <section>
    <h2>Top Architecture Candidates</h2>
    <table>
      <tr><th>Name</th><th>Family</th><th>Utility</th><th>Loss</th><th>Reasoning</th></tr>
      {''.join(f"<tr><td>{html.escape(row['name'])}</td><td>{html.escape(row['family'])}</td><td>{row['utility_score']}</td><td>{row['validation_loss']}</td><td>{row['reasoning_score']}</td></tr>" for row in arch_rows)}
    </table>
  </section>
  <section>
    <h2>Evaluation Weights</h2>
    <pre>{_json_block(eval_weights)}</pre>
  </section>
  <section>
    <h2>Production Integration</h2>
    <pre>{_json_block(production_summary)}</pre>
  </section>
  <section>
    <h2>Validation</h2>
    <pre>{_json_block(validation)}</pre>
  </section>
</body>
</html>
"""

    def write(self, output: str | Path) -> Path:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(), encoding="utf-8")
        return path
