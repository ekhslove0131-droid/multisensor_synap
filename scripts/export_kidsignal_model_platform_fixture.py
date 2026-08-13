"""Export the non-deployable Kidsignal model-platform contract fixture."""

from __future__ import annotations

import argparse
from pathlib import Path

from multisensor_ml.platform_handoff import export_contract_fixture_handoff


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export_contract_fixture_handoff(args.output)
    print(result.manifest_path)
    print(result.checksums_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
