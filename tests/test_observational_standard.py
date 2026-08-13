from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from multisensor_ml.observational_standard import (
    DECISION_WARMUP_SEC,
    FORECAST_HORIZON_SEC,
    PURGE_SEC,
    build_observational_frame,
    make_purged_splits,
    train_observational_standard,
)


def _observed_frame(rows: int = 12_000) -> pd.DataFrame:
    timestamp = pd.date_range(
        "2026-08-04T00:00:00Z", periods=rows, freq="s", tz="UTC"
    )
    seconds = np.arange(rows, dtype=np.float64)
    # Slowly changing observed signals make persistence and a causal regressor
    # meaningfully comparable without using any synthetic truth label.
    cycle = np.sin(seconds / 180.0)
    return pd.DataFrame(
        {
            "person_key": "person-1",
            "session_id": "session-1",
            "corrected_utc": timestamp,
            "observed__heart_rate_bpm": 70.0 + 4.0 * cycle + seconds / 3_600.0,
            "observed__eda_us": 2.0 + 0.5 * np.sin(seconds / 240.0 + 0.4),
            "observed__motion_magnitude": 0.20 + 0.08 * np.maximum(cycle, 0.0),
            "quality_confidence": np.ones(rows),
            "missing_seconds": np.zeros(rows),
        }
    )


def test_observational_target_is_future_and_quality_is_not_load_multiplier() -> None:
    frame = _observed_frame(3_900)
    frame.loc[1_900, "quality_confidence"] = 0.2
    built = build_observational_frame(frame)

    assert built["standard_target"].notna().sum() > 0
    assert built["standard_target"].isna().iloc[-1]
    assert built.loc[1_900, "watch_load_raw"] == pytest.approx(
        built.loc[1_899, "watch_load_raw"], abs=5.0
    )
    assert "quality_confidence" in built
    assert built.loc[1_900, "quality_ok"] == 0
    assert built.loc[1_900, "watch_load_raw"] >= 0
    assert built.loc[DECISION_WARMUP_SEC, "decisionable"] in (0, 1)


def test_purged_temporal_split_removes_boundary_and_target_crossings() -> None:
    built = build_observational_frame(_observed_frame())
    split = make_purged_splits(built, purge_sec=PURGE_SEC)
    assert set(split["split_role"].dropna().unique()) <= {
        "train",
        "validation",
        "locked_test",
        "purged",
    }
    for role in ("train", "validation", "locked_test"):
        selected = split.loc[split["split_role"] == role]
        assert selected["standard_valid"].astype(bool).all()
    assert (split["split_role"] == "purged").sum() > 0


def test_observational_training_exports_manifest_and_onnx_without_locked_test(
    tmp_path: Path,
) -> None:
    source = tmp_path / "observed.parquet"
    _observed_frame().to_parquet(source, index=False)
    output = tmp_path / "artifact"

    result = train_observational_standard(
        source,
        output,
        source_domain="synthetic_truth_oracle",
        random_state=20260804,
        hgb_max_iter=20,
    )

    assert result["status"] == "TRAINED"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_version"] == "observational_standard_30m_watch_candidate_v2"
    assert manifest["feature_schema_version"] == "watch_standard_16_v2"
    assert manifest["legacy_15_feature_status"] == "legacy_not_for_serving"
    assert manifest["locked_test_read"] is False
    assert manifest["target_contract"]["horizon_sec"] == FORECAST_HORIZON_SEC
    assert manifest["split_contract"]["purge_sec"] == PURGE_SEC
    assert manifest["real_data_status"] == "NOT VERIFIED"
    assert manifest["stage"] is None
    assert (output / "predictions_validation.parquet").is_file()
    for name, digest in manifest["artifacts"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    assert not list(output.glob("*.pkl"))
    assert not list(output.glob("*.joblib"))
    if manifest["selected_candidate"] != "none":
        assert (output / "model.onnx").is_file()
    if manifest["selected_candidate"] not in {"persistence", "rolling_median_300", "none"}:
        assert (output / "model.skops").is_file()


def test_observational_training_fails_closed_without_corrected_utc(tmp_path: Path) -> None:
    source = tmp_path / "bad.parquet"
    frame = _observed_frame(3_000).drop(columns=["corrected_utc"])
    frame.to_parquet(source, index=False)
    with pytest.raises(ValueError, match="corrected_utc"):
        build_observational_frame(frame)


def test_short_observed_data_is_insufficient_support(tmp_path: Path) -> None:
    source = tmp_path / "short.parquet"
    _observed_frame(1_000).to_parquet(source, index=False)
    result = train_observational_standard(
        source,
        tmp_path / "short-artifact",
        source_domain="real_observed",
    )
    assert result["status"] == "INSUFFICIENT_SUPPORT"
    assert result["real_data_status"] == "NOT VERIFIED"
