from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import pytest

import multisensor_ml.observational_standard as observational

EXPECTED_FEATURE_NAMES = (
    "watch_eda_z",
    "watch_hr_z",
    "watch_motion_z",
    "watch_load_raw",
    "watch_eda_mean_30",
    "watch_hr_mean_30",
    "watch_motion_mean_30",
    "watch_eda_slope_60",
    "watch_hr_slope_60",
    "watch_motion_slope_60",
    "watch_load_std_60",
    "watch_load_median_300",
    "watch_load_ema_1800",
    "watch_load_ema_21600",
    "quality_confidence",
    "watch_ineligible_fraction_60",
)


def _canonical_frame(rows: int = 3_700) -> pd.DataFrame:
    seconds = np.arange(rows, dtype=np.float64)
    phase = np.mod(seconds, 4.0)
    after_baseline = np.maximum(seconds - 899.0, 0.0)
    eda = 2.0 + phase * 0.1
    hr = 60.0 + phase * 2.0
    motion = 0.50 + phase * 0.05
    drift = seconds >= 900.0
    eda[drift] = 2.15 + after_baseline[drift] * 0.0001
    hr[drift] = 63.0 + after_baseline[drift] * 0.001
    motion[drift] = 0.575 + after_baseline[drift] * 0.00002
    return pd.DataFrame(
        {
            "person_key": "person-v2",
            "session_id": "session-v2",
            "corrected_utc": pd.date_range(
                "2026-08-04T00:00:00Z", periods=rows, freq="s", tz="UTC"
            ),
            "observed__eda_us": eda,
            "observed__heart_rate_bpm": hr,
            "observed__motion_magnitude": motion,
            "quality_confidence": np.ones(rows, dtype=np.float64),
        }
    )


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_canonical_schema_has_exact_order_shape_and_self_hash() -> None:
    schema = observational.canonical_feature_schema()

    assert observational.FEATURE_NAMES == EXPECTED_FEATURE_NAMES
    assert schema["schema_version"] == "watch_standard_16_v2"
    assert schema["input"] == {
        "name": "features",
        "dtype": "float32",
        "shape": [None, 16],
    }
    assert tuple(item["name"] for item in schema["features"]) == EXPECTED_FEATURE_NAMES
    assert _canonical_json_sha256(schema) == observational.FEATURE_SCHEMA_SHA256


@pytest.mark.parametrize(
    "missing_column",
    [
        "quality_confidence",
        "observed__eda_us",
        "observed__heart_rate_bpm",
        "observed__motion_magnitude",
    ],
)
def test_observed_input_fails_closed_when_quality_or_core_signal_is_missing(
    missing_column: str,
) -> None:
    frame = _canonical_frame(2_000).drop(columns=[missing_column])

    with pytest.raises(ValueError, match=missing_column):
        observational.build_observational_frame(frame)


def test_baseline_and_decision_use_eligible_seconds_not_row_defaults() -> None:
    frame = _canonical_frame()
    frame.loc[np.arange(9, 1_000, 10), "quality_confidence"] = 0.2

    built = observational.build_observational_frame(frame)

    assert built.loc[997, "baseline_ready"] == 0
    assert built.loc[998, "baseline_ready"] == 1
    assert built.loc[1_898, "feature_decisionable"] == 0
    assert built.loc[1_899, "feature_decisionable"] == 1
    assert built.loc[1_899, "target_valid"] == 1
    assert built.loc[len(built) - 1, "feature_decisionable"] == 1
    assert built.loc[len(built) - 1, "target_valid"] == 0


def test_zero_mad_and_iqr_fail_closed_instead_of_using_unit_scale() -> None:
    frame = _canonical_frame(2_000)
    frame["observed__eda_us"] = 2.0

    built = observational.build_observational_frame(frame)

    assert set(built["baseline_status"]) == {
        "BASELINE_SCALE_ZERO:observed__eda_us"
    }
    assert not built["baseline_ready"].astype(bool).any()
    assert not built["feature_decisionable"].astype(bool).any()


def test_rolling_features_are_strict_past_and_slopes_use_corrected_seconds() -> None:
    frame = _canonical_frame()
    changed_current = frame.copy()
    changed_current.loc[1_799, [
        "observed__eda_us",
        "observed__heart_rate_bpm",
        "observed__motion_magnitude",
    ]] = [20.0, 180.0, 4.0]

    built = observational.build_observational_frame(frame)
    changed = observational.build_observational_frame(changed_current)

    strict_past = EXPECTED_FEATURE_NAMES[4:12]
    np.testing.assert_allclose(
        built.loc[1_799, list(strict_past)].to_numpy(dtype=np.float64),
        changed.loc[1_799, list(strict_past)].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    )
    assert changed.loc[1_799, "watch_load_raw"] != pytest.approx(
        built.loc[1_799, "watch_load_raw"]
    )
    assert built.loc[1_799, "watch_eda_slope_60"] == pytest.approx(
        0.0001 / 0.14826, rel=1e-9
    )
    assert built.loc[1_799, "watch_hr_slope_60"] == pytest.approx(
        0.001 / 2.9652, rel=1e-9
    )
    assert built.loc[1_799, "watch_motion_slope_60"] == pytest.approx(
        0.00002 / 0.07413, rel=1e-9
    )


def test_bounded_ema_updates_only_on_eligible_seconds() -> None:
    actual = observational.bounded_ema_eligible(
        np.array([1.0, np.nan, 3.0], dtype=np.float64),
        np.array([True, False, True]),
        half_life_seconds=1.0,
        initial=1.0,
    )

    np.testing.assert_allclose(actual, np.array([1.0, 1.0, 2.0]), atol=1e-12)
    assert pytest.approx(0.000385007632510, rel=1e-12) == observational.EMA_ALPHA_1800
    assert pytest.approx(0.000032089632365, rel=1e-12) == observational.EMA_ALPHA_21600


def test_future_target_requires_ninety_percent_and_rejects_six_second_gap() -> None:
    frame = _canonical_frame()
    frame.loc[2_000:2_005, "quality_confidence"] = 0.2

    built = observational.build_observational_frame(frame)

    assert built.loc[1_799, "feature_decisionable"] == 1
    assert built.loc[1_799, "future_valid_fraction"] > 0.99
    assert built.loc[1_799, "future_max_gap_seconds"] == 6
    assert built.loc[1_799, "target_valid"] == 0


def test_baseline_shadow_bundle_selects_new_median_column_and_carries_boundaries(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    manifest = observational.export_baseline_shadow_bundle(bundle)

    assert manifest["model_release"] == (
        "observational_standard_30m_watch_baseline_shadow_v1"
    )
    assert manifest["selected_candidate"] == "rolling_median_300"
    assert manifest["legacy_15_feature_status"] == "legacy_not_for_serving"
    assert manifest["real_data_status"] == "NOT VERIFIED"
    assert manifest["stage"] is None
    assert manifest["feature_schema_hash"] == observational.FEATURE_SCHEMA_SHA256
    assert observational.verify_baseline_shadow_bundle(bundle)["status"] == "VERIFIED"

    model = onnx.load(bundle / "model.onnx")
    assert model.graph.input[0].type.tensor_type.shape.dim[1].dim_value == 16
    session = ort.InferenceSession(
        str(bundle / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    vector = np.arange(16, dtype=np.float32).reshape(1, 16)
    prediction = session.run(None, {"features": vector})[0]
    np.testing.assert_array_equal(prediction, np.array([[11.0]], dtype=np.float32))

    golden = json.loads((bundle / "golden_fixture.json").read_text(encoding="utf-8"))
    assert len(golden["input_rows"]) == 1_800
    assert len(golden["expected_feature_vector"]) == 16
    assert golden["expected_prediction"] == pytest.approx(
        golden["expected_feature_vector"][11]
    )
    for name, digest in manifest["artifacts"].items():
        assert hashlib.sha256((bundle / name).read_bytes()).hexdigest() == digest


def test_bundle_verifier_replays_golden_input_instead_of_trusting_expected_vector(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    observational.export_baseline_shadow_bundle(bundle)
    golden_path = bundle / "golden_fixture.json"
    manifest_path = bundle / "manifest.json"
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    golden["input_rows"][-1]["observed__eda_us"] = 100.0
    golden_path.write_text(
        json.dumps(golden, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["golden_fixture.json"] = hashlib.sha256(
        golden_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    checksums_path = bundle / "SHA256SUMS.json"
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    checksums["golden_fixture.json"] = hashlib.sha256(golden_path.read_bytes()).hexdigest()
    checksums["manifest.json"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    checksums_path.write_text(
        json.dumps(checksums, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="golden feature vector mismatch"):
        observational.verify_baseline_shadow_bundle(bundle)


def test_bundle_verifier_checks_manifest_checksum_before_loading_contract(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    observational.export_baseline_shadow_bundle(bundle)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["stage"] = "HIGH"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"bundle checksum mismatch: manifest\.json"):
        observational.verify_baseline_shadow_bundle(bundle)
