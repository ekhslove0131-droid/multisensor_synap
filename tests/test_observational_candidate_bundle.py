from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import multisensor_ml.observational_standard as observational


def _observed_frame(rows: int = 6_000) -> pd.DataFrame:
    seconds = np.arange(rows, dtype=np.float64)
    cycle = np.sin(seconds / 180.0)
    return pd.DataFrame(
        {
            "person_key": "candidate-person",
            "session_id": "candidate-session",
            "corrected_utc": pd.date_range(
                "2026-08-06T00:00:00Z", periods=rows, freq="s", tz="UTC"
            ),
            "observed__eda_us": 2.0 + 0.5 * np.sin(seconds / 240.0),
            "observed__heart_rate_bpm": 70.0 + 4.0 * cycle,
            "observed__motion_magnitude": 0.2 + 0.08 * np.maximum(cycle, 0.0),
            "quality_confidence": np.ones(rows, dtype=np.float64),
        }
    )


def test_candidate_training_can_pin_a_learned_cpu_candidate_without_locked_test(
    tmp_path: Path,
) -> None:
    source = tmp_path / "observed.parquet"
    _observed_frame().to_parquet(source, index=False)

    result = observational.train_observational_standard(
        source,
        tmp_path / "trained",
        source_domain="synthetic_truth_oracle",
        candidate_name="ridge",
        hgb_max_iter=8,
    )

    assert result["selected_candidate"] == "ridge"
    assert result["locked_test_read"] is False
    assert result["candidate_selection_mode"] == "PINNED_CANDIDATE"


def test_candidate_bundle_is_separate_continuous_shadow_with_null_thresholds(
    tmp_path: Path,
) -> None:
    source = tmp_path / "observed.parquet"
    _observed_frame(6_000).to_parquet(source, index=False)
    active = tmp_path / "active"
    candidate = tmp_path / "candidate"
    observational.export_baseline_shadow_bundle(active)

    manifest = observational.export_candidate_shadow_bundle(
        source,
        candidate,
        active_bundle=active,
        candidate_name="ridge",
        hgb_max_iter=8,
    )

    assert manifest["model_release"] == (
        "observational_standard_30m_watch_baseline_shadow_candidate_v1"
    )
    assert manifest["release_status"] == "candidate"
    assert manifest["delivery_eligible"] is False
    assert manifest["real_data_status"] == "NOT VERIFIED"
    assert manifest["stage"] is None
    assert manifest["locked_test_read"] is False
    assert manifest["thresholds"] is None
    assert manifest["calibration"]["t2"] is None
    assert manifest["calibration"]["blocker"] == (
        "NO_SEPARATE_CALIBRATION_TRAINING_EVIDENCE"
    )
    assert manifest["active_reference"]["model_artifact_sha256"] == hashlib.sha256(
        (active / "model.onnx").read_bytes()
    ).hexdigest()
    assert observational.verify_candidate_shadow_bundle(candidate)["status"] == (
        "VERIFIED"
    )

    golden = json.loads((candidate / "golden_fixture.json").read_text())
    assert len(golden["input_rows"]) == 1_800
    assert golden["ordered_feature_names"] == list(observational.FEATURE_NAMES)
    assert len(golden["expected_feature_vector"]) == 16
    assert np.isfinite(golden["expected_candidate_prediction"])
    assert golden["expected_stage"] is None


def test_candidate_bundle_refuses_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "observed.parquet"
    _observed_frame(6_000).to_parquet(source, index=False)
    output = tmp_path / "candidate"
    observational.export_baseline_shadow_bundle(tmp_path / "active")
    observational.export_candidate_shadow_bundle(
        source,
        output,
        active_bundle=tmp_path / "active",
        candidate_name="ridge",
        hgb_max_iter=8,
    )
    with pytest.raises(FileExistsError):
        observational.export_candidate_shadow_bundle(
            source,
            output,
            active_bundle=tmp_path / "active",
            candidate_name="ridge",
            hgb_max_iter=8,
        )
