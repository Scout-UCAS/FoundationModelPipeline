from __future__ import annotations

import argparse

from fmops.native_training import add_native_training_arguments, native_options_from_args
from fmops.training_entrypoint import run_stage_entrypoint


def main() -> int:
    parser = argparse.ArgumentParser(description="SFT launcher")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--mixture", default="reasoning_final")
    parser.add_argument("--world-size", type=int, default=400)
    parser.add_argument("--output", default="artifacts/runs/sft_plan.json")
    parser.add_argument("--mode", choices=["dry-run", "native", "external"], default="dry-run")
    parser.add_argument("--backend-command")
    add_native_training_arguments(parser)
    args = parser.parse_args()
    return run_stage_entrypoint(
        config_dir=args.config_dir,
        stage_name="SFT",
        mixture=args.mixture,
        world_size=args.world_size,
        output=args.output,
        mode=args.mode,
        backend_command=args.backend_command,
        native_options=native_options_from_args(args),
    )


if __name__ == "__main__":
    raise SystemExit(main())
