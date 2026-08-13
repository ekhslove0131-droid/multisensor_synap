import hashlib
import json
from copy import deepcopy

import pytest

import multisensor_ml.h10_runtime_contract as h10_contract

CLOUD_COMPARISON_VECTOR = (
    1.3489815189531904,
    -0.3372453797382976,
    0.6744907594765952,
    1.0117361392148927,
    0.0,
    0.03279685548997059,
    -7.249302883770819e-19,
    0.2103094443755307,
    0.3372453797382976,
    0.4204238569363234,
    0.4962003332486074,
    1.0,
    0.0,
)


def _summary_payload() -> dict[str, object]:
    period = 1_000_000_000 // 130
    samples = [
        {
            "device_timestamp_ns": index * period,
            "phone_monotonic_ns": 20_000_000_000 + index // 130,
            "phone_utc_anchor_ms": 10_000 + index // 130,
            "corrected_utc_ms": 10_000 + index * 1_000 // 130,
            "voltage_uv": 120 + index % 3,
        }
        for index in range(1_300)
    ]
    return {
        "payload_schema_version": "kidsignal_polar_h10_ecg_v2",
        "source_domain": "real_observed_h10",
        "device_profile": "polar_h10_v1",
        "session_id": "session-1",
        "batch_id": "batch-1",
        "sequence": 1,
        "window_id": "window-1",
        "person_key": "person-1",
        "device_id": "h10-1",
        "connection_epoch": 2,
        "window_start_corrected_utc_ms": 10_000,
        "sample_rate_hz": 130,
        "expected_sample_count": 1_300,
        "observed_sample_count": 1_300,
        "gap_count": 0,
        "timestamp_regression_count": 0,
        "is_sufficient": True,
        "close_reason": "sample_count",
        "hr_stream_state": "observed",
        "hr_observations": [
            {
                "phone_monotonic_ns": 20_000_000_000,
                "phone_utc_ms": 11_000,
                "hr_bpm": 69,
                "corrected_hr_bpm": 70,
                "ppg_quality": 3,
                "rr_1024": [819, 829],
                "rr_ms": [800, 810],
                "rr_available": True,
                "contact_status": True,
                "contact_status_supported": True,
            },
            {
                "phone_monotonic_ns": 20_000_000_000,
                "phone_utc_ms": 15_000,
                "hr_bpm": 71,
                "corrected_hr_bpm": 72,
                "ppg_quality": 3,
                "rr_1024": [809, 819],
                "rr_ms": [790, 800],
                "rr_available": True,
                "contact_status": True,
                "contact_status_supported": True,
            },
            {
                "phone_monotonic_ns": 20_000_000_000,
                "phone_utc_ms": 19_000,
                "hr_bpm": 73,
                "corrected_hr_bpm": 74,
                "ppg_quality": 3,
                "rr_1024": [824],
                "rr_ms": [805],
                "rr_available": True,
                "contact_status": True,
                "contact_status_supported": True,
            },
        ],
        "battery_observations": [
            {
                "phone_monotonic_ns": 20_000_000_000,
                "phone_utc_ms": 10_000,
                "percent": 91,
            }
        ],
        "ecg_samples": samples,
    }


def _eligible_summary(index: int) -> dict[str, object]:
    return {
        "eligible": True,
        "eligible_seconds": 10,
        "emission_corrected_utc_ms": (index + 1) * 10_000 - 1,
        "hr_bpm_median": float(70 + index % 9),
        "rr_rmssd_ms": float(30 + index % 7),
        "ecg_coverage": 1.0,
        "contact_valid_fraction": 1.0,
        "rr_valid_fraction": 1.0,
        "quality_confidence": 1.0,
        "reason_codes": [],
    }


def test_model_summary_recipe_has_canonical_bytes_and_expected_summary() -> None:
    payload = _summary_payload()
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")

    summary = h10_contract.summarize_h10_v2_window(payload)

    assert len(encoded) == 180_535
    assert hashlib.sha256(encoded).hexdigest() == (
        "f75039ec05b80a72e60b326a5f9ab50a1c9b75c768f982b774b4522a08cd3877"
    )
    assert isinstance(summary["emission_corrected_utc_ms"], int)
    assert summary == {
        "eligible": True,
        "eligible_seconds": 10,
        "emission_corrected_utc_ms": 19_992,
        "hr_bpm_median": 72.0,
        "rr_rmssd_ms": 12.5,
        "ecg_coverage": 1.0,
        "contact_valid_fraction": 1.0,
        "rr_valid_fraction": 1.0,
        "quality_confidence": 1.0,
        "reason_codes": [],
    }


def test_vector_builder_freezes_900_second_baseline_and_waits_for_1800() -> None:
    result = h10_contract.build_h10_standard_13_vector(
        [_eligible_summary(index) for index in range(179)]
    )

    assert result["status"] == "NOT_DECISIONABLE"
    assert result["reason_code"] == "DECISION_WARMUP_INCOMPLETE"
    assert result["baseline_ready"] is True
    assert result["eligible_seconds"] == 1790
    assert result["vector"] is None
    assert result["baseline"] == {
        "eligible_seconds": 900,
        "wall_seconds": 900.0,
        "hr_center": 74.0,
        "hr_scale": 2.9652,
        "rmssd_center": 33.0,
        "rmssd_scale": 2.9652,
    }


def test_vector_builder_matches_contract_recipe_at_1800_eligible_seconds() -> None:
    summaries = [_eligible_summary(index) for index in range(180)]

    result = h10_contract.build_h10_standard_13_vector(summaries)

    assert result["status"] == "DECISIONABLE"
    assert result["reason_code"] == "DECISIONABLE"
    assert result["baseline_ready"] is True
    assert result["eligible_seconds"] == 1800
    assert tuple(result["feature_names"]) == h10_contract.H10_FEATURE_NAMES
    assert len(result["vector"]) == 13
    assert result["vector"] == pytest.approx(CLOUD_COMPARISON_VECTOR, abs=1e-15)


def test_current_ineligible_window_is_not_replaced_by_previous_physiology() -> None:
    summaries = [_eligible_summary(index) for index in range(180)]
    summaries[-1] = {
        **deepcopy(summaries[-1]),
        "eligible": False,
        "eligible_seconds": 0,
        "quality_confidence": 0.0,
        "reason_codes": ["ECG_GAP_PRESENT"],
    }

    result = h10_contract.build_h10_standard_13_vector(summaries)

    assert result["status"] == "NOT_DECISIONABLE"
    assert result["reason_code"] == "CURRENT_WINDOW_INELIGIBLE"
    assert result["vector"] is None


def test_vector_result_identity_is_exact_and_training_remains_blocked() -> None:
    result = h10_contract.build_h10_standard_13_vector(
        [_eligible_summary(index) for index in range(180)]
    )

    assert result["schema_version"] == "h10_standard_13_v1"
    assert result["schema_uuid"] == "8ba3dbff-e5c4-5dd3-be1d-69ea6cea8966"
    assert result["schema_sha256"] == (
        "5ff8e9ceaa534bb8b7fafbdd7f844b06c0a4b21bb206428d5ce9239eec5459d6"
    )
    assert result["producer_status"] == "PROPOSED_NOT_IMPLEMENTED"
    assert result["training_status"] == "BLOCKED_NO_REAL_COHORT"
    assert result["real_data_status"] == "NOT VERIFIED"
