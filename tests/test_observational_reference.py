from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from multisensor_ml.observational_contract import build_observational_frame
from multisensor_ml.observational_reference import (
    export_training_reference_bundle,
    reference_prediction,
    verify_training_reference_bundle,
)
from multisensor_ml.observational_standard import train_observational_standard

GENERATED_AT = "2026-08-06T09:00:00Z"


def _observed_frame(rows: int = 6_000) -> pd.DataFrame:
    seconds = np.arange(rows, dtype=np.float64)
    cycle = np.sin(seconds / 180.0)
    return pd.DataFrame(
        {
            "person_key": "reference-person",
            "session_id": "reference-session",
            "corrected_utc": pd.date_range(
                "2026-08-06T00:00:00Z", periods=rows, freq="s", tz="UTC"
            ),
            "observed__eda_us": 2.0 + 0.5 * np.sin(seconds / 240.0),
            "observed__heart_rate_bpm": 70.0 + 4.0 * cycle,
            "observed__motion_magnitude": 0.2
            + 0.08 * np.maximum(cycle, 0.0),
            "quality_confidence": np.ones(rows, dtype=np.float64),
        }
    )


def _candidate_bundle(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "observed.parquet"
    _observed_frame().to_parquet(source, index=False)
    candidate = tmp_path / "candidate"
    train_observational_standard(
        source,
        candidate,
        source_domain="synthetic_truth_oracle",
        candidate_name="ridge",
        hgb_max_iter=8,
    )
    return source, candidate


def test_reference_uses_training_rows_only_and_keeps_persistence_mase_denominator(
    tmp_path: Path,
) -> None:
    source, candidate = _candidate_bundle(tmp_path)
    contract = export_training_reference_bundle(
        source,
        candidate,
        tmp_path / "reference",
        generated_at=GENERATED_AT,
    )

    observed = build_observational_frame(pd.read_parquet(source))
    split = pd.read_parquet(candidate / "split_assignments.parquet")
    train_rows = split["split_role"].eq("train")
    target = split.loc[train_rows, "standard_target"].to_numpy(dtype="float64")
    persistence = observed.loc[train_rows, "watch_load_raw"].to_numpy(
        dtype="float64"
    )
    expected_mae = float(np.mean(np.abs(persistence - target)))

    assert contract["persistence_training_mae"] == expected_mae
    assert contract["mase_denominator"] == {
        "method": "persistence_training_mae",
        "value": expected_mae,
        "split_role": "train",
    }
    assert contract["persistence_training_mae"] != 0.345370995662263
    assert contract["selection_lineage"] == {
        "roles_read": ["train"],
        "validation_read": False,
        "locked_test_read": False,
    }
    assert contract["thresholds"] is None


def test_reference_selection_is_unchanged_when_non_training_targets_are_poisoned(
    tmp_path: Path,
) -> None:
    source, candidate = _candidate_bundle(tmp_path)
    original = export_training_reference_bundle(
        source,
        candidate,
        tmp_path / "reference-original",
        generated_at=GENERATED_AT,
    )
    poisoned_candidate = tmp_path / "candidate-poisoned"
    poisoned_candidate.mkdir()
    for name in ("manifest.json", "feature_schema.json"):
        (poisoned_candidate / name).write_bytes((candidate / name).read_bytes())
    split = pd.read_parquet(candidate / "split_assignments.parquet")
    split.loc[split["split_role"] != "train", "standard_target"] = 999_999.0
    split.to_parquet(poisoned_candidate / "split_assignments.parquet", index=False)

    poisoned = export_training_reference_bundle(
        source,
        poisoned_candidate,
        tmp_path / "reference-poisoned",
        generated_at=GENERATED_AT,
    )

    for field in (
        "training_split_digest",
        "persistence_training_mae",
        "rolling_median_300_training_mae",
        "reference_method",
        "reference_artifact_sha256",
    ):
        assert poisoned[field] == original[field]


def test_reference_export_is_byte_identical_and_verifies_missing_ood_fail_closed(
    tmp_path: Path,
) -> None:
    source, candidate = _candidate_bundle(tmp_path)
    first = tmp_path / "reference-first"
    second = tmp_path / "reference-second"
    export_training_reference_bundle(
        source, candidate, first, generated_at=GENERATED_AT
    )
    export_training_reference_bundle(
        source, candidate, second, generated_at=GENERATED_AT
    )

    first_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in first.iterdir()
        if path.is_file()
    }
    second_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in second.iterdir()
        if path.is_file()
    }
    assert first_hashes == second_hashes

    verified = verify_training_reference_bundle(first)
    contract = json.loads((first / "reference_contract.json").read_text())
    assert verified["status"] == "VERIFIED"
    assert contract["golden_verified"] is True
    assert contract["idempotency_verified"] is True
    assert contract["missing_ood_verified"] is True

    vector = np.arange(16, dtype=np.float32)
    assert reference_prediction(
        vector, feature_decisionable=True, ood_status="IN_DISTRIBUTION"
    )["status"] == "PREDICTED"
    vector[11] = np.nan
    assert reference_prediction(
        vector, feature_decisionable=True, ood_status="IN_DISTRIBUTION"
    )["status"] == "NOT_DECISIONABLE"
    assert reference_prediction(
        np.arange(16, dtype=np.float32),
        feature_decisionable=True,
        ood_status="OOD_MONITOR",
    )["status"] == "NOT_DECISIONABLE"
