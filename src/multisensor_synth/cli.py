from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml
from pydantic import ValidationError

from multisensor_synth.config.loader import ConfigLoadError, load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="multisensor-synth")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_command = commands.add_parser("inspect-config")
    inspect_command.add_argument("--config", required=True, type=Path)

    estimate_command = commands.add_parser("estimate")
    estimate_command.add_argument("--config", required=True, type=Path)

    generate_command = commands.add_parser("generate")
    generate_command.add_argument("--config", required=True, type=Path)
    generate_command.add_argument("--truth-only", action="store_true")

    validate_command = commands.add_parser("validate")
    validate_command.add_argument("--run", required=True, type=Path)
    validate_command.add_argument("--scope", choices=("truth",), required=True)
    return parser


def estimate_config(config_path: str | Path) -> dict[str, object]:
    config = load_config(config_path).config
    days = math.ceil(config.run.duration_sec / 86_400)
    target_min = (
        config.run.participant_count
        * days
        * config.events.target_events_per_day.min
    )
    target_max = (
        config.run.participant_count
        * days
        * config.events.target_events_per_day.max
    )
    return {
        "profile": config.profile,
        "participants": config.run.participant_count,
        "duration_sec": config.run.duration_sec,
        "canonical_rate_hz": config.run.canonical_rate_hz,
        "canonical_truth_rows": (
            config.run.participant_count
            * config.run.duration_sec
            * config.run.canonical_rate_hz
        ),
        "target_event_range": {"min": target_min, "max": target_max},
        "native_clips": "deferred_to_goal_2",
        "estimated_target_clip_sec": {
            "min": target_min
            * (config.events.clip_window.pre_sec + config.events.clip_window.post_sec),
            "max": target_max
            * (config.events.clip_window.pre_sec + config.events.clip_window.post_sec),
        },
    }


def run_cli(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "inspect-config":
            loaded = load_config(args.config)
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "config_sha256": loaded.config_sha256,
                        "resolved": loaded.resolved_data,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "estimate":
            print(json.dumps(estimate_config(args.config), indent=2, sort_keys=True))
            return 0
        if args.command == "generate":
            if not args.truth_only:
                print(
                    "configuration error: Goal 1 generate requires --truth-only",
                    file=sys.stderr,
                )
                return 2
            estimate = estimate_config(args.config)
            print(json.dumps({"estimate": estimate}, sort_keys=True))
            from multisensor_synth.orchestration.pipeline import generate_truth_run

            generated = generate_truth_run(args.config)
            print(
                json.dumps(
                    {
                        "run_id": generated.run_id,
                        "run_path": str(generated.run_dir),
                        "validation": generated.report.status,
                    },
                    sort_keys=True,
                )
            )
            return 0 if generated.report.status == "PASS" else 2
        if args.command == "validate":
            from multisensor_synth.validation.invariants import validate_truth_run
            from multisensor_synth.validation.report import write_validation_report

            report = validate_truth_run(args.run)
            write_validation_report(
                args.run / "reports" / "truth_validation_report.json", report
            )
            print(
                json.dumps(
                    {
                        "run_id": report.run_id,
                        "scope": report.scope,
                        "status": report.status,
                        "checks_run": report.checks_run,
                        "finding_count": len(report.findings),
                    },
                    sort_keys=True,
                )
            )
            return 0 if report.status == "PASS" else 2
    except (ConfigLoadError, ValidationError, yaml.YAMLError, FileNotFoundError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    raise RuntimeError(f"unhandled command: {args.command}")


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
