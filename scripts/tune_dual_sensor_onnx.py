#!/usr/bin/env python3
"""Tune and export the two final synthetic sensor-view ONNX bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multisensor_ml.optuna_dual_sensor import (
    DualSensorTuningConfig,
    tune_dual_sensor_models,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--series", default="mvp3-oracle-v1")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("services/onnx_api/models/goal15-final-v1"),
    )
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--max-train-rows", type=int, default=600_000)
    parser.add_argument("--max-validation-rows", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()
    config = DualSensorTuningConfig(
        project_root=args.project_root,
        series_id=args.series,
        output_root=args.output,
        n_trials=args.trials,
        max_train_rows=args.max_train_rows,
        max_validation_rows=args.max_validation_rows,
        seed=args.seed,
    )
    result = tune_dual_sensor_models(config)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output.resolve()),
                "variants": list(result["variants"]),
                "primary_metric": result["primary_metric"],
                "data_status": result["model_scope"],
                "real_data_status": result["real_data_status"],
                "locked_test_read": result["locked_test_read"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
