from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import cast

import pandas as pd
import pyarrow.parquet as pq

from multisensor_ml.pipeline import _write_parquet


def export_knime_artifacts(
    bundle_root: Path,
    prepared_root: Path,
    output_root: Path,
) -> Path:
    """Export bounded Parquet/JSON tables consumed by the KNIME workflow."""

    if output_root.exists():
        raise FileExistsError(f"KNIME export already exists: {output_root}")
    output_root.mkdir(parents=True)
    prepared = json.loads(
        (prepared_root / "manifest.json").read_text(encoding="utf-8")
    )
    people = cast(list[dict[str, object]], prepared["people"])
    split = pd.DataFrame(
        [
            {
                "dataset_id": row["dataset_id"],
                "run_id": row["run_id"],
                "person_id": row["person_id"],
                "person_key": row["person_key"],
                "split_role": row["split_role"],
                "row_count": row["row_count"],
                "source_domain": "synthetic_truth_oracle",
                "status": "oracle/sanity",
            }
            for row in people
        ]
    )
    _write_parquet(split, output_root / "dataset_split.parquet")

    metrics = pq.read_table(bundle_root / "metrics.parquet").to_pandas()
    _write_parquet(
        metrics.loc[metrics["role"].isin(["validation", "locked_test"])],
        output_root / "pattern_performance.parquet",
    )
    _write_parquet(
        metrics.loc[metrics["role"] == "stress"],
        output_root / "noise_stress.parquet",
    )
    for name in (
        "personal_baseline.parquet",
        "predictions.parquet",
        "pr_curve.parquet",
        "synchronization.parquet",
    ):
        shutil.copy2(bundle_root / name, output_root / name)
    status = pd.DataFrame(
        [
            {
                "source_domain": "synthetic_truth_oracle",
                "accuracy_status": "oracle/sanity",
                "synchronization_status": "NOT_AVAILABLE_TRUTH_ONLY",
            },
            {
                "source_domain": "real_observed",
                "accuracy_status": "NOT VERIFIED",
                "synchronization_status": "NOT VERIFIED",
            },
        ]
    )
    _write_parquet(status, output_root / "real_status.parquet")
    dashboard = {
        "schema": "goal1.5/knime-dashboard/v1",
        "status": "oracle/sanity",
        "sections": [
            {"name": "Dataset & Split", "table": "dataset_split.parquet"},
            {"name": "Personal Baseline", "table": "personal_baseline.parquet"},
            {
                "name": "Pattern Performance",
                "table": "pattern_performance.parquet",
                "pr_curve": "pr_curve.parquet",
                "timeline": "predictions.parquet",
            },
            {"name": "Noise Stress", "table": "noise_stress.parquet"},
            {
                "name": "Synchronization & Real Audit",
                "table": "synchronization.parquet",
                "status_table": "real_status.parquet",
            },
        ],
    }
    (output_root / "dashboard_manifest.json").write_text(
        json.dumps(dashboard, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_root
