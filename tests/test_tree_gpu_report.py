from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def test_kaggle_gpu_report_is_hash_closed_and_locked_test_safe() -> None:
    root = Path("reports")
    artifact = json.loads(
        (root / "tree_model_gpu_kaggle_ko.artifact.json").read_text(encoding="utf-8")
    )

    assert artifact["status"] == "VERIFIED_ORACLE_GPU_RUN"
    assert artifact["data_status"] == "oracle/sanity"
    assert artifact["real_data_status"] == "NOT VERIFIED"
    assert artifact["locked_test_read"] is False
    assert artifact["dataset"]["split_counts"] == {
        "train": 24,
        "validation": 6,
        "locked_test": 0,
    }
    assert set(artifact["candidate_model_ids"]) == {
        "xgboost",
        "lightgbm",
        "extra_trees",
    }
    for filename, expected_hash in artifact["file_sha256"].items():
        path = root / filename
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash

    metrics = pd.read_parquet(root / "tree_model_gpu_kaggle_ko.metrics.parquet")
    assert set(metrics["model_id"]) == {
        "xgboost",
        "lightgbm",
        "extra_trees",
        "soft_ensemble",
    }
    assert set(metrics.loc[metrics["gpu_requested"], "execution_device"]) == {"cuda", "mixed"}
