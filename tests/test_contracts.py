from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from multisensor_ml.contracts import (
    FORBIDDEN_ORACLE_EXACT,
    ORACLE_LATENT_FACTORS,
    DatasetRecord,
    SynchronizationRecord,
    ValidationCandidateManifest,
    assert_oracle_columns,
    assign_person_splits,
    build_labels,
)


def test_split_is_person_grouped_deterministic_and_24_6_6() -> None:
    people = [(f"run-{run}", f"person-{person:02d}") for run in range(3) for person in range(12)]

    first = assign_person_splits(people)
    second = assign_person_splits(list(reversed(people)))

    assert first == second
    assert list(first.values()).count("train") == 24
    assert list(first.values()).count("validation") == 6
    assert list(first.values()).count("locked_test") == 6
    assert len(first) == len(set(first)) == 36


def test_oracle_contract_accepts_whitelist_and_rejects_truth_leakage() -> None:
    allowed = [
        "timestamp_utc",
        "person_id",
        *ORACLE_LATENT_FACTORS,
        "is_awake",
        "context",
    ]
    assert_oracle_columns(allowed)

    for forbidden in sorted(FORBIDDEN_ORACLE_EXACT):
        with pytest.raises(ValueError, match="truth leakage"):
            assert_oracle_columns([*allowed, forbidden])

    with pytest.raises(ValueError, match="truth leakage"):
        assert_oracle_columns([*allowed, "active_target_event_binary"])


def test_event_and_forecast_label_boundaries_are_independent_of_latent_rows() -> None:
    timeline = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range(
                datetime(2026, 1, 1, tzinfo=UTC), periods=130, freq="s"
            )
        }
    )
    events = pd.DataFrame(
        {
            "event_id": ["event-1"],
            "onset_utc": [datetime(2026, 1, 1, 0, 1, 0, tzinfo=UTC)],
            "end_utc": [datetime(2026, 1, 1, 0, 1, 10, tzinfo=UTC)],
        }
    )

    labels = build_labels(timeline, events)

    assert labels.loc[0:59, "forecast_60s"].eq(1).all()
    assert labels.loc[60, "forecast_60s"] == 0
    assert labels.loc[59, "event_binary"] == 0
    assert labels.loc[60:70, "event_binary"].eq(1).all()
    assert labels.loc[71, "event_binary"] == 0
    assert labels.loc[60, "phase"] == "event"
    assert labels.loc[0, "phase"] == "forecast"


def test_dataset_record_requires_lineage_and_split_contract() -> None:
    record = DatasetRecord(
        dataset_id="dataset-abc",
        source_domain="synthetic_truth_oracle",
        generator_commit="a" * 40,
        config_sha256="b" * 64,
        run_id="run-1",
        person_key="run-1/person-01",
        time_column="timestamp_utc",
        schema_version="goal1.5/v1",
        logical_hash="c" * 64,
        split_role="train",
    )

    assert record.split_role == "train"
    with pytest.raises(ValueError):
        DatasetRecord.model_validate({**record.model_dump(), "split_role": "test"})


def test_oracle_synchronization_cannot_invent_device_measurements() -> None:
    record = SynchronizationRecord(
        dataset_id="synthetic-oracle",
        session_id=None,
        person_key=None,
        device_pair=None,
        reference_device="Polar H10",
        offset_ms=None,
        drift_ppm=None,
        jitter_ms=None,
        physiological_lag_ms=None,
        overlap_sec=None,
        correlation=None,
        corrected_time_axis="UTC",
        watch_ecg_policy="calibration_only",
        status="NOT_AVAILABLE_TRUTH_ONLY",
    )
    assert record.status == "NOT_AVAILABLE_TRUTH_ONLY"

    with pytest.raises(ValueError, match="must not invent"):
        SynchronizationRecord.model_validate(
            {**record.model_dump(), "offset_ms": 3.5}
        )


def test_validation_candidate_manifest_requires_both_candidates() -> None:
    payload = {
        "schema_version": "goal1.5/ml-validation-candidates/v1",
        "split_role": "validation",
        "locked_test_used": False,
        "candidate_models": [
            "hist_gradient_boosting",
            "logistic_regression",
        ],
        "source_dataset_hash": "a" * 64,
        "split_hash": "b" * 64,
        "feature_schema_hash": "c" * 64,
        "row_count": 64,
        "file": "validation_candidate_metrics.parquet",
        "file_sha256": "d" * 64,
    }

    manifest = ValidationCandidateManifest.model_validate(payload)

    assert manifest.locked_test_used is False
    with pytest.raises(ValueError, match="exactly Logistic and HGB"):
        ValidationCandidateManifest.model_validate(
            {
                **payload,
                "candidate_models": ["logistic_regression"],
            }
        )
