"""CLI handlers for local-only SYNTHETIC_BETA preparation, fit, and evaluation."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any


def run_beta_command(args: Namespace) -> int:
    command = args.beta_loop_command
    if command == "prepare":
        from .dataset import prepare_dataset

        prepared = prepare_dataset(args.observed, args.output)
        payload: dict[str, object] = {
            "status": "SYNTHETIC_BETA_PREPARED",
            "manifest": str(prepared.manifest_path.resolve()),
            "manifest_sha256": prepared.manifest_sha256,
        }
    elif command == "train":
        from .pipeline import BetaTrainConfig
        from .training_run import train_beta

        manifest = train_beta(
            args.dataset,
            args.output,
            wire_root=args.wire_root,
            runtime_wheel=args.runtime_wheel,
            config=BetaTrainConfig.load(args.config),
        )
        payload = {"status": manifest["status"],
                   "manifest": str((args.output / "manifest.json").resolve()),
                   "fit_attempt_count": manifest["m2_fit_attempt_count"]}
    else:
        from .evaluation_run import evaluate_beta

        receipt = evaluate_beta(args.bundle, args.dataset, args.output)
        locked_use = receipt["locked_use"]
        if not isinstance(locked_use, dict):
            raise ValueError("locked_use_receipt_invalid")
        payload = {"status": receipt["status"],
                   "evaluation": str((args.output / "evaluation.json").resolve()),
                   "locked_use_count": locked_use["use_count"]}
    print(json.dumps(payload, sort_keys=True))
    return 0


def configure_beta_parser(subparsers: Any) -> None:
    beta = subparsers.add_parser("beta-loop")
    commands = beta.add_subparsers(dest="beta_loop_command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--observed", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    train = commands.add_parser("train")
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--wire-root", type=Path, required=True)
    train.add_argument("--runtime-wheel", type=Path, required=True)
    train.add_argument("--config", type=Path, required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--bundle", type=Path, required=True)
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
