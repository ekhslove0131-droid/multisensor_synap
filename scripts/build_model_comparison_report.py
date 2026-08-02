#!/usr/bin/env python3
"""Generate the Korean model comparison HTML from verified validation artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from multisensor_ml.model_comparison import (
    build_comparison_artifact,
    collect_comparison_metrics,
    render_comparison_html,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font-family", default="NanumGothic")
    args = parser.parse_args()
    frame = collect_comparison_metrics(args.project_root)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    artifact = build_comparison_artifact(frame, generated_at=generated_at)
    render_comparison_html(artifact, args.output, font_family=args.font_family)
    print(
        {
            "status": "READY",
            "rows": len(frame),
            "output": str(args.output.resolve()),
            "artifact": str(args.output.with_suffix(".artifact.json").resolve()),
            "data_status": "oracle/sanity",
            "real_data_status": "NOT VERIFIED",
            "locked_test_read": False,
            "font_family": args.font_family,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

