"""CLI for the isolated local SYNTHETIC_BETA observed exporter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import BetaConfig
from .exporter import generate_observed


def main() -> int:
    parser = argparse.ArgumentParser(prog="multisensor-synth-beta")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = generate_observed(BetaConfig.load(args.config), args.output)
    print(json.dumps({"status": "SYNTHETIC_BETA_OBSERVED_GENERATED",
                      "manifest": str(manifest.manifest_path.resolve()),
                      "manifest_sha256": manifest.manifest_sha256}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
