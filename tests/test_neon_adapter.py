from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from multisensor_ml.neon_adapter import (
    derive_watch_features,
    fit_personal_baseline_adapter,
    personal_adapter_manifest,
    transform_with_personal_adapter,
)


def _mapped_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_id": ["s1", "s1", "s1", "s1"],
            "sensor_kind": ["heart_rate_ibi", "eda", "accelerometer", "skin_temperature"],
            "sensor_timestamp_utc": pd.to_datetime(
                [
                    "2026-08-02T00:00:00Z",
                    "2026-08-02T00:00:00Z",
                    "2026-08-02T00:00:00Z",
                    "2026-08-02T00:00:01Z",
                ],
                utc=True,
            ),
            "heart_rate_bpm": [80.0, None, None, None],
            "heart_rate_status": [0, None, None, None],
            "skin_conductance_us": [None, 1.2, None, None],
            "status": [None, 0, None, 0],
            "raw_x": [None, None, 3.0, None],
            "raw_y": [None, None, 4.0, None],
            "raw_z": [None, None, 0.0, None],
            "object_temperature_c": [None, None, None, 31.0],
        }
    )


def test_derive_watch_features_requires_corrected_utc_and_preserves_clock_boundary() -> None:
    with pytest.raises(ValueError, match="corrected_utc"):
        derive_watch_features(_mapped_rows())

    derived = derive_watch_features(_mapped_rows(), clock_offset_ms=500.0)
    assert derived.loc[0, "corrected_utc"] == pd.Timestamp("2026-08-02T00:00:00Z")
    assert derived.loc[0, "observed__heart_rate_bpm"] == 80.0
    assert derived.loc[0, "observed__motion_magnitude"] == 5.0
    assert derived.loc[0, "clock_offset_ms"] == 500.0
    assert "event_probability" not in derived.columns


def test_personal_adapter_is_unlabeled_weighted_and_deterministic(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "person_id": ["p1"] * 4,
            "corrected_utc": pd.date_range("2026-08-02", periods=4, freq="s", tz="UTC"),
            "observed__heart_rate_bpm": [80.0, 82.0, 81.0, 120.0],
            "quality_confidence": [1.0, 1.0, 0.5, 1.0],
        }
    )
    adapter = fit_personal_baseline_adapter(
        frame,
        person_id="p1",
        value_columns=("observed__heart_rate_bpm",),
        warmup_seconds=10,
        weight_cap=0.6,
    )
    assert adapter.status == "WARMUP_INCOMPLETE"
    assert 0.0 <= adapter.personal_weight <= 0.6
    assert adapter.model_weights_changed is False
    transformed = transform_with_personal_adapter(frame, adapter)
    assert "adapter__observed__heart_rate_bpm__robust_z" in transformed
    assert "personal_weight" in transformed
    manifest = personal_adapter_manifest(adapter, model_version="hierarchical-v5")
    assert manifest["promotable"] is False
    assert manifest["promotion_block_reason"] == "NO_OBSERVED_LABELS"
    assert manifest["source"] == "neon-observed"
