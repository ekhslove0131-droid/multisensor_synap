import hashlib
import json
import math
from copy import deepcopy
from itertools import pairwise

import pytest

import multisensor_ml.platform_contract as platform_contract

EXPECTED_FEATURE_NAMES = (
    "h10_hr_z",
    "h10_rmssd_inverse_z",
    "h10_load_raw",
    "h10_hr_mean_30",
    "h10_rmssd_inverse_mean_30",
    "h10_hr_slope_60",
    "h10_rmssd_inverse_slope_60",
    "h10_load_std_60",
    "h10_load_median_300",
    "h10_load_ema_1800",
    "h10_load_ema_21600",
    "h10_quality_confidence",
    "h10_ineligible_fraction_60",
)


def _valid_payload() -> dict[str, object]:
    start_ms = 1_786_000_000_000
    ecg_samples = [
        {
            "corrected_utc_ms": start_ms + round(index * 1000 / 130),
            "voltage_uv": float((index % 31) - 15),
        }
        for index in range(1300)
    ]
    return {
        "payload_schema_version": "kidsignal_polar_h10_ecg_v2",
        "source_domain": "real_observed_h10",
        "device_profile": "polar_h10_v1",
        "window_start_corrected_utc_ms": start_ms,
        "sample_rate_hz": 130,
        "expected_sample_count": 1300,
        "observed_sample_count": 1300,
        "gap_count": 0,
        "timestamp_regression_count": 0,
        "is_sufficient": True,
        "hr_stream_state": "observed",
        "ecg_samples": ecg_samples,
        "hr_observations": [
            {
                "phone_utc_ms": start_ms + 1000,
                "hr_bpm": 69,
                "corrected_hr_bpm": 70,
                "rr_ms": [800.0, 810.0],
                "rr_available": True,
                "contact_status": True,
                "contact_status_supported": True,
            },
            {
                "phone_utc_ms": start_ms + 5000,
                "hr_bpm": 71,
                "corrected_hr_bpm": 72,
                "rr_ms": [790.0, 800.0],
                "rr_available": True,
                "contact_status": True,
                "contact_status_supported": True,
            },
            {
                "phone_utc_ms": start_ms + 9000,
                "hr_bpm": 73,
                "corrected_hr_bpm": 74,
                "rr_ms": [805.0],
                "rr_available": True,
                "contact_status": True,
                "contact_status_supported": True,
            },
        ],
    }


def test_h10_runtime_schema_is_canonical_json_with_recomputed_identity() -> None:
    schema = platform_contract.h10_runtime_feature_schema()
    body = {
        key: value
        for key, value in schema.items()
        if key not in {"canonical_json", "sha256"}
    }
    canonical = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    assert schema["schema_version"] == "h10_standard_13_v1"
    assert schema["schema_uuid"] == "8ba3dbff-e5c4-5dd3-be1d-69ea6cea8966"
    assert schema["input_tensor"] == {
        "name": "features",
        "dtype": "float32",
        "shape": [None, 13],
    }
    assert tuple(schema["ordered_feature_names"]) == EXPECTED_FEATURE_NAMES
    assert schema["canonical_json"] == canonical
    assert schema["sha256"] == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert schema["sha256"] == (
        "5ff8e9ceaa534bb8b7fafbdd7f844b06c0a4b21bb206428d5ce9239eec5459d6"
    )
    assert schema["producer_status"] == "PROPOSED_NOT_IMPLEMENTED"
    assert schema["real_data_status"] == "NOT VERIFIED"


def test_h10_runtime_schema_locks_formulas_and_causal_window_rules() -> None:
    schema = platform_contract.h10_runtime_feature_schema()
    formulas = {item["name"]: item["formula"] for item in schema["features"]}

    assert formulas == {
        "h10_hr_z": "clip((HR_t-median(HR_baseline_900))/scale(HR_baseline_900),-20,20)",
        "h10_rmssd_inverse_z": (
            "clip(-(RMSSD_t-median(RMSSD_baseline_900))/"
            "scale(RMSSD_baseline_900),-20,20)"
        ),
        "h10_load_raw": "mean(max(h10_hr_z,0),max(h10_rmssd_inverse_z,0))",
        "h10_hr_mean_30": "time_weighted_mean(h10_hr_z,[t-30,t))",
        "h10_rmssd_inverse_mean_30": "time_weighted_mean(h10_rmssd_inverse_z,[t-30,t))",
        "h10_hr_slope_60": "corrected_utc_ols_slope(h10_hr_z,[t-60,t))",
        "h10_rmssd_inverse_slope_60": "corrected_utc_ols_slope(h10_rmssd_inverse_z,[t-60,t))",
        "h10_load_std_60": "time_weighted_population_std(h10_load_raw,[t-60,t),ddof=0)",
        "h10_load_median_300": "time_weighted_median(h10_load_raw,[t-300,t))",
        "h10_load_ema_1800": (
            "bounded_ema_eligible_second_replay(h10_load_raw,half_life_seconds=1800)"
        ),
        "h10_load_ema_21600": (
            "bounded_ema_eligible_second_replay(h10_load_raw,half_life_seconds=21600)"
        ),
        "h10_quality_confidence": (
            "hard_integrity_gate*min(ecg_coverage,contact_valid_fraction,"
            "rr_valid_fraction)"
        ),
        "h10_ineligible_fraction_60": "1-eligible_seconds([t-60,t))/60",
    }
    assert schema["window_semantics"] == "[t-window,t)"
    assert schema["current_window_summary"] == {
        "HR_t": "median(corrected_hr_bpm from contact-valid observations)",
        "RMSSD_t": "sqrt(mean(diff(flatten(rr_ms in callback order))^2))",
        "emission_time": "max(ecg_samples.corrected_utc_ms)",
        "granularity_seconds": 10,
    }
    assert schema["ecg_voltage_policy"] == (
        "finite raw integrity and provenance only; no R-peak, diagnosis, or model feature in v1"
    )


def test_h10_runtime_schema_locks_quality_baseline_and_coverage() -> None:
    schema = platform_contract.h10_runtime_feature_schema()

    assert schema["quality_contract"] == {
        "expected_payload_schema_version": "kidsignal_polar_h10_ecg_v2",
        "expected_device_profile": "polar_h10_v1",
        "sample_rate_hz": 130,
        "full_window_expected_sample_count": 1300,
        "minimum_ecg_coverage": 0.9,
        "maximum_gap_count": 0,
        "maximum_timestamp_regression_count": 0,
        "require_payload_is_sufficient": True,
        "require_observed_count_matches_samples": True,
        "require_strictly_increasing_corrected_utc_ms": True,
        "require_finite_voltage_uv": True,
        "require_hr_stream_state": "observed",
        "require_contact_supported_and_true": True,
        "require_corrected_hr_bpm": "finite_positive",
        "minimum_contact_valid_fraction": 0.9,
        "minimum_rr_valid_fraction": 0.9,
        "minimum_valid_rr_intervals": 2,
        "quality_missing_policy": "FAIL_CLOSED",
    }
    assert schema["baseline_contract"] == {
        "past_only": True,
        "minimum_eligible_seconds": 900,
        "maximum_wall_seconds": 1000,
        "freeze": "first_qualifying_900_eligible_seconds_then_immutable_for_session",
        "center": "median",
        "scale": "1.4826*MAD",
        "scale_fallback": ["IQR/1.349", "BASELINE_SCALE_ZERO"],
    }
    assert schema["decision_contract"] == {
        "minimum_total_eligible_seconds": 1800,
        "required_features_finite": 13,
        "missing_or_ineligible": "NOT_DECISIONABLE",
        "imputation": "NO_INTERPOLATION_NO_HOLD_FORWARD_NO_ZERO_PADDING",
    }
    assert schema["rolling_coverage"] == {
        "30_seconds": 0.9,
        "60_seconds": 0.9,
        "300_seconds": 0.9,
        "ema_missing_update": "FREEZE_STATE",
    }


def test_valid_h10_v2_window_has_deterministic_summary_and_quality() -> None:
    summary = platform_contract.summarize_h10_v2_window(_valid_payload())
    rr = [800.0, 810.0, 790.0, 800.0, 805.0]
    expected_rmssd = math.sqrt(
        sum((right - left) ** 2 for left, right in pairwise(rr))
        / (len(rr) - 1)
    )

    assert summary["eligible"] is True
    assert summary["eligible_seconds"] == 10
    assert summary["hr_bpm_median"] == 72.0
    assert summary["rr_rmssd_ms"] == pytest.approx(expected_rmssd)
    assert summary["ecg_coverage"] == 1.0
    assert summary["contact_valid_fraction"] == 1.0
    assert summary["rr_valid_fraction"] == 1.0
    assert summary["quality_confidence"] == 1.0
    assert summary["reason_codes"] == []


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda payload: payload.update(sample_rate_hz=129), "SAMPLE_RATE_MISMATCH"),
        (lambda payload: payload.update(gap_count=1), "ECG_GAP_PRESENT"),
        (
            lambda payload: payload.update(timestamp_regression_count=1),
            "ECG_TIMESTAMP_REGRESSION",
        ),
        (lambda payload: payload.update(is_sufficient=False), "SOURCE_QUALITY_INSUFFICIENT"),
        (
            lambda payload: payload.update(observed_sample_count=1299),
            "OBSERVED_COUNT_MISMATCH",
        ),
        (
            lambda payload: payload["hr_observations"][0].update(contact_status=False),
            "CONTACT_COVERAGE_LOW",
        ),
        (
            lambda payload: payload["hr_observations"][0].update(
                contact_status_supported=False
            ),
            "CONTACT_COVERAGE_LOW",
        ),
        (
            lambda payload: payload["hr_observations"][0].update(corrected_hr_bpm=0),
            "CONTACT_COVERAGE_LOW",
        ),
        (
            lambda payload: payload["hr_observations"][0].update(rr_available=False),
            "RR_COVERAGE_LOW",
        ),
        (
            lambda payload: payload["ecg_samples"][3].update(voltage_uv=float("nan")),
            "ECG_VOLTAGE_NONFINITE",
        ),
        (
            lambda payload: payload["ecg_samples"][3].update(
                corrected_utc_ms=payload["ecg_samples"][2]["corrected_utc_ms"]
            ),
            "ECG_CORRECTED_TIME_NOT_STRICT",
        ),
    ],
)
def test_h10_v2_window_quality_failures_are_fail_closed(mutation: object, reason: str) -> None:
    payload = deepcopy(_valid_payload())
    mutation(payload)

    summary = platform_contract.summarize_h10_v2_window(payload)

    assert summary["eligible"] is False
    assert summary["eligible_seconds"] == 0
    assert reason in summary["reason_codes"]


def test_h10_runtime_registry_uses_new_schema_but_remains_not_training_ready() -> None:
    runtime = platform_contract.runtime_feature_schema_for_variant("h10_only")
    schema = platform_contract.h10_runtime_feature_schema()

    assert runtime["schema_uuid"] == schema["schema_uuid"]
    assert runtime["schema_version"] == schema["schema_version"]
    assert runtime["schema_hash"] == schema["sha256"]
    assert runtime["runtime_schema"] == schema
    assert runtime["training_status"] == "PROPOSED_NOT_IMPLEMENTED"
    assert runtime["training_ready"] is False
    assert runtime["schema_hash"] != (
        "f2d6256362726747e59f5951af08f0c0dbf491b8f46bc00ed9c2c196c6b36485"
    )


def test_polar_source_schema_names_actual_h10_v2_fields() -> None:
    schema = platform_contract.source_schema("polar_h10")
    fields = {item["name"]: item for item in schema["fields"]}

    assert fields["payload_schema_version"]["type"] == (
        "literal<kidsignal_polar_h10_ecg_v2>"
    )
    assert fields["device_profile"]["type"] == "literal<polar_h10_v1>"
    assert fields["sample_rate_hz"]["required"] is True
    assert fields["expected_sample_count"]["required"] is True
    assert fields["observed_sample_count"]["required"] is True
    assert fields["gap_count"]["required"] is True
    assert fields["timestamp_regression_count"]["required"] is True
    assert fields["is_sufficient"]["required"] is True
    assert fields["ecg_samples"]["sample_rate_hz"] == 130
    assert "corrected_hr_bpm" in fields["hr_observations"]["type"]
    assert "rr_ms" in fields["hr_observations"]["type"]
    assert "contact_status" in fields["hr_observations"]["type"]


def test_watch_h10_remains_unapproved_after_h10_contract_definition() -> None:
    runtime = platform_contract.runtime_feature_schema_for_variant("watch_h10")

    assert runtime["training_status"] == "PROPOSED_NOT_APPROVED"
    assert runtime["training_ready"] is False
    assert runtime["runtime_schema"] is None
