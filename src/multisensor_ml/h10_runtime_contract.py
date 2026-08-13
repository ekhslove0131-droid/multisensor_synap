"""Executable Polar H10 observation-v2 feature contract.

The contract is model-owned but not a claim that Cloud has an H10 decoder or
feature producer.  It turns the observed Android payload shape into an exact,
fail-closed proposal that Cloud can implement and cross-check later.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Final
from uuid import NAMESPACE_URL, uuid5

H10_PAYLOAD_SCHEMA_VERSION: Final[str] = "kidsignal_polar_h10_ecg_v2"
H10_DEVICE_PROFILE: Final[str] = "polar_h10_v1"
H10_FEATURE_SCHEMA_VERSION: Final[str] = "h10_standard_13_v1"
H10_FEATURE_SCHEMA_UUID: Final[str] = str(
    uuid5(
        NAMESPACE_URL,
        f"https://kidsignal.local/runtime-feature-schema/{H10_FEATURE_SCHEMA_VERSION}",
    )
)
H10_FEATURE_NAMES: Final[tuple[str, ...]] = (
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

_FEATURE_FORMULAS: Final[dict[str, tuple[str, str, str]]] = {
    "h10_hr_z": (
        "clip((HR_t-median(HR_baseline_900))/scale(HR_baseline_900),-20,20)",
        "hr_observations.corrected_hr_bpm",
        "robust_z",
    ),
    "h10_rmssd_inverse_z": (
        "clip(-(RMSSD_t-median(RMSSD_baseline_900))/scale(RMSSD_baseline_900),-20,20)",
        "hr_observations.rr_ms",
        "inverse_robust_z",
    ),
    "h10_load_raw": (
        "mean(max(h10_hr_z,0),max(h10_rmssd_inverse_z,0))",
        "DERIVED_H10",
        "positive_robust_z",
    ),
    "h10_hr_mean_30": (
        "time_weighted_mean(h10_hr_z,[t-30,t))",
        "DERIVED_CAUSAL",
        "robust_z",
    ),
    "h10_rmssd_inverse_mean_30": (
        "time_weighted_mean(h10_rmssd_inverse_z,[t-30,t))",
        "DERIVED_CAUSAL",
        "inverse_robust_z",
    ),
    "h10_hr_slope_60": (
        "corrected_utc_ols_slope(h10_hr_z,[t-60,t))",
        "DERIVED_CAUSAL",
        "robust_z_per_second",
    ),
    "h10_rmssd_inverse_slope_60": (
        "corrected_utc_ols_slope(h10_rmssd_inverse_z,[t-60,t))",
        "DERIVED_CAUSAL",
        "inverse_robust_z_per_second",
    ),
    "h10_load_std_60": (
        "time_weighted_population_std(h10_load_raw,[t-60,t),ddof=0)",
        "DERIVED_CAUSAL",
        "positive_robust_z",
    ),
    "h10_load_median_300": (
        "time_weighted_median(h10_load_raw,[t-300,t))",
        "DERIVED_CAUSAL",
        "positive_robust_z",
    ),
    "h10_load_ema_1800": (
        "bounded_ema_eligible_second_replay(h10_load_raw,half_life_seconds=1800)",
        "DERIVED_CAUSAL",
        "positive_robust_z",
    ),
    "h10_load_ema_21600": (
        "bounded_ema_eligible_second_replay(h10_load_raw,half_life_seconds=21600)",
        "DERIVED_CAUSAL",
        "positive_robust_z",
    ),
    "h10_quality_confidence": (
        "hard_integrity_gate*min(ecg_coverage,contact_valid_fraction,rr_valid_fraction)",
        "CAPTURE_METADATA",
        "ratio_0_1",
    ),
    "h10_ineligible_fraction_60": (
        "1-eligible_seconds([t-60,t))/60",
        "CAPTURE_METADATA",
        "ratio_0_1",
    ),
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_h10_feature_schema() -> dict[str, object]:
    """Return the exact model-side H10 proposal and its content identity."""

    body: dict[str, object] = {
        "schema_version": H10_FEATURE_SCHEMA_VERSION,
        "schema_uuid": H10_FEATURE_SCHEMA_UUID,
        "sensor_variant": "h10_only",
        "producer_status": "PROPOSED_NOT_IMPLEMENTED",
        "training_ready": False,
        "real_data_status": "NOT VERIFIED",
        "payload_contract": {
            "payload_schema_version": H10_PAYLOAD_SCHEMA_VERSION,
            "source_domain": "real_observed_h10",
            "device_profile": H10_DEVICE_PROFILE,
            "required_window_fields": [
                "window_start_corrected_utc_ms",
                "sample_rate_hz",
                "expected_sample_count",
                "observed_sample_count",
                "gap_count",
                "timestamp_regression_count",
                "is_sufficient",
                "hr_stream_state",
                "ecg_samples",
                "hr_observations",
            ],
            "required_ecg_sample_fields": ["corrected_utc_ms", "voltage_uv"],
            "required_hr_observation_fields": [
                "phone_utc_ms",
                "hr_bpm",
                "corrected_hr_bpm",
                "rr_ms",
                "rr_available",
                "contact_status",
                "contact_status_supported",
            ],
            "audit_only_fields": [
                "hr_observations.hr_bpm",
                "hr_observations.ppg_quality",
                "hr_observations.rr_1024",
                "battery_observations",
            ],
        },
        "input_tensor": {
            "name": "features",
            "dtype": "float32",
            "shape": [None, len(H10_FEATURE_NAMES)],
        },
        "ordered_feature_names": list(H10_FEATURE_NAMES),
        "features": [
            {
                "name": name,
                "formula": _FEATURE_FORMULAS[name][0],
                "source_metric": _FEATURE_FORMULAS[name][1],
                "unit": _FEATURE_FORMULAS[name][2],
                "causal": True,
            }
            for name in H10_FEATURE_NAMES
        ],
        "current_window_summary": {
            "HR_t": "median(corrected_hr_bpm from contact-valid observations)",
            "RMSSD_t": "sqrt(mean(diff(flatten(rr_ms in callback order))^2))",
            "emission_time": "max(ecg_samples.corrected_utc_ms)",
            "granularity_seconds": 10,
        },
        "hr_time_policy": (
            "phone_utc_ms proves source-window membership only; it is not a corrected "
            "physiological beat timestamp and is never interpolated or re-anchored"
        ),
        "rmssd_boundary_policy": (
            "within one source window only; do not bridge windows without per-RR corrected time"
        ),
        "ecg_voltage_policy": (
            "finite raw integrity and provenance only; no R-peak, diagnosis, or model feature in v1"
        ),
        "quality_contract": {
            "expected_payload_schema_version": H10_PAYLOAD_SCHEMA_VERSION,
            "expected_device_profile": H10_DEVICE_PROFILE,
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
        },
        "baseline_contract": {
            "past_only": True,
            "minimum_eligible_seconds": 900,
            "maximum_wall_seconds": 1000,
            "freeze": "first_qualifying_900_eligible_seconds_then_immutable_for_session",
            "center": "median",
            "scale": "1.4826*MAD",
            "scale_fallback": ["IQR/1.349", "BASELINE_SCALE_ZERO"],
        },
        "decision_contract": {
            "minimum_total_eligible_seconds": 1800,
            "required_features_finite": len(H10_FEATURE_NAMES),
            "missing_or_ineligible": "NOT_DECISIONABLE",
            "imputation": "NO_INTERPOLATION_NO_HOLD_FORWARD_NO_ZERO_PADDING",
        },
        "rolling_coverage": {
            "30_seconds": 0.9,
            "60_seconds": 0.9,
            "300_seconds": 0.9,
            "ema_missing_update": "FREEZE_STATE",
        },
        "window_semantics": "[t-window,t)",
        "ema_contract": {
            "alpha_per_eligible_second": "1-exp(-ln(2)/half_life_seconds)",
            "initialization": "first eligible h10_load_raw",
            "ten_second_window_update": (
                "replay the same eligible window summary for ten one-second updates"
            ),
        },
        "fusion_status": "WATCH_H10_PROPOSED_NOT_APPROVED",
        "clinical_scope": (
            "physiological monitoring research only; no diagnosis, behavior label, "
            "or treatment output"
        ),
    }
    serialized = _canonical_json(body)
    return {
        **body,
        "canonical_json": serialized,
        "sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    }


def _sequence(value: object) -> list[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _integer(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _append_once(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def summarize_h10_v2_window(payload: Mapping[str, object]) -> dict[str, object]:
    """Summarize one full H10 v2 window under the proposed fail-closed gate."""

    reasons: list[str] = []
    if payload.get("payload_schema_version") != H10_PAYLOAD_SCHEMA_VERSION:
        _append_once(reasons, "PAYLOAD_SCHEMA_MISMATCH")
    if payload.get("source_domain") != "real_observed_h10":
        _append_once(reasons, "SOURCE_DOMAIN_MISMATCH")
    if payload.get("device_profile") != H10_DEVICE_PROFILE:
        _append_once(reasons, "DEVICE_PROFILE_MISMATCH")

    sample_rate = _integer(payload.get("sample_rate_hz"))
    expected_count = _integer(payload.get("expected_sample_count"))
    observed_count = _integer(payload.get("observed_sample_count"))
    gap_count = _integer(payload.get("gap_count"))
    regression_count = _integer(payload.get("timestamp_regression_count"))
    if sample_rate != 130:
        _append_once(reasons, "SAMPLE_RATE_MISMATCH")
    if expected_count != 1300:
        _append_once(reasons, "EXPECTED_COUNT_MISMATCH")
    if payload.get("is_sufficient") is not True:
        _append_once(reasons, "SOURCE_QUALITY_INSUFFICIENT")
    if gap_count != 0:
        _append_once(reasons, "ECG_GAP_PRESENT")
    if regression_count != 0:
        _append_once(reasons, "ECG_TIMESTAMP_REGRESSION")

    ecg_samples = _sequence(payload.get("ecg_samples"))
    if observed_count != len(ecg_samples):
        _append_once(reasons, "OBSERVED_COUNT_MISMATCH")
    ecg_coverage = (
        min(1.0, max(0.0, observed_count / expected_count))
        if observed_count is not None and expected_count is not None and expected_count > 0
        else 0.0
    )
    if ecg_coverage < 0.9:
        _append_once(reasons, "ECG_COVERAGE_LOW")

    corrected_times: list[int] = []
    finite_voltage = True
    for item in ecg_samples:
        if not isinstance(item, Mapping):
            finite_voltage = False
            continue
        corrected = _integer(item.get("corrected_utc_ms"))
        voltage = _finite_number(item.get("voltage_uv"))
        if corrected is None:
            _append_once(reasons, "ECG_CORRECTED_TIME_NONFINITE")
        else:
            corrected_times.append(corrected)
        if voltage is None:
            finite_voltage = False
    if not finite_voltage:
        _append_once(reasons, "ECG_VOLTAGE_NONFINITE")
    if len(corrected_times) != len(ecg_samples) or any(
        right <= left for left, right in pairwise(corrected_times)
    ):
        _append_once(reasons, "ECG_CORRECTED_TIME_NOT_STRICT")

    if payload.get("hr_stream_state") != "observed":
        _append_once(reasons, "HR_STREAM_NOT_OBSERVED")
    hr_observations = _sequence(payload.get("hr_observations"))
    if not hr_observations:
        _append_once(reasons, "HR_OBSERVATIONS_MISSING")

    contact_valid_count = 0
    rr_valid_count = 0
    valid_hr_values: list[float] = []
    valid_rr_values: list[float] = []
    for item in hr_observations:
        if not isinstance(item, Mapping):
            continue
        corrected_hr = _finite_number(item.get("corrected_hr_bpm"))
        contact_valid = (
            item.get("contact_status_supported") is True
            and item.get("contact_status") is True
            and corrected_hr is not None
            and corrected_hr > 0
        )
        if contact_valid:
            contact_valid_count += 1
        if contact_valid and corrected_hr is not None:
            valid_hr_values.append(corrected_hr)

        rr_items = _sequence(item.get("rr_ms"))
        parsed_rr = [_finite_number(value) for value in rr_items]
        rr_valid = (
            contact_valid
            and
            item.get("rr_available") is True
            and bool(parsed_rr)
            and all(value is not None and value > 0 for value in parsed_rr)
        )
        if rr_valid:
            rr_valid_count += 1
            if contact_valid:
                valid_rr_values.extend(float(value) for value in parsed_rr if value is not None)

    observation_count = len(hr_observations)
    contact_fraction = contact_valid_count / observation_count if observation_count else 0.0
    rr_fraction = rr_valid_count / observation_count if observation_count else 0.0
    if contact_fraction < 0.9 or not valid_hr_values:
        _append_once(reasons, "CONTACT_COVERAGE_LOW")
    if rr_fraction < 0.9:
        _append_once(reasons, "RR_COVERAGE_LOW")
    if len(valid_rr_values) < 2:
        _append_once(reasons, "RR_SUPPORT_INSUFFICIENT")

    hard_integrity_reasons = {
        "PAYLOAD_SCHEMA_MISMATCH",
        "SOURCE_DOMAIN_MISMATCH",
        "DEVICE_PROFILE_MISMATCH",
        "SAMPLE_RATE_MISMATCH",
        "EXPECTED_COUNT_MISMATCH",
        "SOURCE_QUALITY_INSUFFICIENT",
        "ECG_GAP_PRESENT",
        "ECG_TIMESTAMP_REGRESSION",
        "OBSERVED_COUNT_MISMATCH",
        "ECG_COVERAGE_LOW",
        "ECG_CORRECTED_TIME_NONFINITE",
        "ECG_CORRECTED_TIME_NOT_STRICT",
        "ECG_VOLTAGE_NONFINITE",
        "HR_STREAM_NOT_OBSERVED",
        "HR_OBSERVATIONS_MISSING",
    }
    hard_integrity_gate = not any(reason in hard_integrity_reasons for reason in reasons)
    quality = (
        min(ecg_coverage, contact_fraction, rr_fraction) if hard_integrity_gate else 0.0
    )
    eligible = not reasons and quality >= 0.9
    hr_median = statistics.median(valid_hr_values) if valid_hr_values else None
    rr_rmssd = None
    if len(valid_rr_values) >= 2:
        squared_differences = [
            (right - left) ** 2
            for left, right in pairwise(valid_rr_values)
        ]
        rr_rmssd = math.sqrt(sum(squared_differences) / len(squared_differences))

    return {
        "eligible": eligible,
        "eligible_seconds": 10 if eligible else 0,
        "emission_corrected_utc_ms": max(corrected_times) if corrected_times else None,
        "hr_bpm_median": hr_median,
        "rr_rmssd_ms": rr_rmssd,
        "ecg_coverage": ecg_coverage,
        "contact_valid_fraction": contact_fraction,
        "rr_valid_fraction": rr_fraction,
        "quality_confidence": quality,
        "reason_codes": reasons,
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _robust_center_scale(values: Sequence[float]) -> tuple[float, float] | None:
    center = float(statistics.median(values))
    mad = float(statistics.median(abs(value - center) for value in values))
    scale = 1.4826 * mad
    if scale == 0.0:
        scale = (_quantile(values, 0.75) - _quantile(values, 0.25)) / 1.349
    if not math.isfinite(scale) or scale <= 0.0:
        return None
    return center, scale


def _vector_result(
    *,
    status: str,
    reason_code: str,
    baseline_ready: bool,
    eligible_seconds: int,
    baseline: dict[str, object] | None,
    vector: list[float] | None,
) -> dict[str, object]:
    schema = canonical_h10_feature_schema()
    return {
        "schema_version": H10_FEATURE_SCHEMA_VERSION,
        "schema_uuid": H10_FEATURE_SCHEMA_UUID,
        "schema_sha256": schema["sha256"],
        "producer_status": "PROPOSED_NOT_IMPLEMENTED",
        "training_status": "BLOCKED_NO_REAL_COHORT",
        "real_data_status": "NOT VERIFIED",
        "status": status,
        "reason_code": reason_code,
        "baseline_ready": baseline_ready,
        "eligible_seconds": eligible_seconds,
        "baseline": baseline,
        "feature_names": list(H10_FEATURE_NAMES),
        "vector": vector,
    }


def _summary_value(summary: Mapping[str, object], field: str) -> float | None:
    return _finite_number(summary.get(field))


def _ols_slope(values: Sequence[float | None]) -> float | None:
    pairs = [(float(index), value) for index, value in enumerate(values) if value is not None]
    if len(pairs) / len(values) < 0.9:
        return None
    x_mean = sum(x for x, _ in pairs) / len(pairs)
    y_mean = sum(y for _, y in pairs) / len(pairs)
    denominator = sum((x - x_mean) ** 2 for x, _ in pairs)
    if denominator == 0.0:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in pairs) / denominator


def _covered_values(values: Sequence[float | None]) -> list[float] | None:
    covered = [value for value in values if value is not None]
    if len(covered) / len(values) < 0.9:
        return None
    return covered


def _bounded_ema(values: Sequence[float | None], half_life_seconds: int) -> float | None:
    alpha = 1.0 - math.exp(-math.log(2.0) / half_life_seconds)
    state: float | None = None
    for value in values:
        if value is None:
            continue
        state = value if state is None else alpha * value + (1.0 - alpha) * state
    return state


def build_h10_standard_13_vector(
    summaries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build one decision vector from ordered, completed H10 source windows."""

    if not summaries:
        return _vector_result(
            status="NOT_DECISIONABLE",
            reason_code="NO_SUMMARIES",
            baseline_ready=False,
            eligible_seconds=0,
            baseline=None,
            vector=None,
        )
    emissions = [_summary_value(summary, "emission_corrected_utc_ms") for summary in summaries]
    if any(value is None for value in emissions):
        return _vector_result(
            status="NOT_DECISIONABLE",
            reason_code="SUMMARY_TIME_INVALID",
            baseline_ready=False,
            eligible_seconds=0,
            baseline=None,
            vector=None,
        )
    emission_values = [float(value) for value in emissions if value is not None]
    if any(right <= left for left, right in pairwise(emission_values)):
        return _vector_result(
            status="NOT_DECISIONABLE",
            reason_code="SUMMARY_TIME_NOT_STRICT",
            baseline_ready=False,
            eligible_seconds=0,
            baseline=None,
            vector=None,
        )

    eligible: list[Mapping[str, object]] = []
    for summary in summaries:
        if summary.get("eligible") is True and summary.get("eligible_seconds") == 10:
            hr = _summary_value(summary, "hr_bpm_median")
            rmssd = _summary_value(summary, "rr_rmssd_ms")
            quality = _summary_value(summary, "quality_confidence")
            if hr is not None and rmssd is not None and quality is not None and quality >= 0.9:
                eligible.append(summary)
    eligible_seconds = len(eligible) * 10
    if len(eligible) < 90:
        return _vector_result(
            status="NOT_DECISIONABLE",
            reason_code="BASELINE_WARMUP_INCOMPLETE",
            baseline_ready=False,
            eligible_seconds=eligible_seconds,
            baseline=None,
            vector=None,
        )

    baseline_summaries = eligible[:90]
    first_emission = _summary_value(baseline_summaries[0], "emission_corrected_utc_ms")
    last_emission = _summary_value(baseline_summaries[-1], "emission_corrected_utc_ms")
    if first_emission is None or last_emission is None:  # pragma: no cover - guarded above
        raise AssertionError("eligible summary emission disappeared")
    baseline_wall_seconds = (last_emission - first_emission) / 1000.0 + 10.0
    if baseline_wall_seconds > 1000.0:
        return _vector_result(
            status="NOT_DECISIONABLE",
            reason_code="BASELINE_WALL_TIME_EXCEEDED",
            baseline_ready=False,
            eligible_seconds=eligible_seconds,
            baseline=None,
            vector=None,
        )
    baseline_hr = [
        float(value)
        for summary in baseline_summaries
        if (value := _summary_value(summary, "hr_bpm_median")) is not None
    ]
    baseline_rmssd = [
        float(value)
        for summary in baseline_summaries
        if (value := _summary_value(summary, "rr_rmssd_ms")) is not None
    ]
    hr_baseline = _robust_center_scale(baseline_hr)
    rmssd_baseline = _robust_center_scale(baseline_rmssd)
    if hr_baseline is None or rmssd_baseline is None:
        return _vector_result(
            status="NOT_DECISIONABLE",
            reason_code="BASELINE_SCALE_ZERO",
            baseline_ready=False,
            eligible_seconds=eligible_seconds,
            baseline=None,
            vector=None,
        )
    hr_center, hr_scale = hr_baseline
    rmssd_center, rmssd_scale = rmssd_baseline
    baseline: dict[str, object] = {
        "eligible_seconds": 900,
        "wall_seconds": baseline_wall_seconds,
        "hr_center": hr_center,
        "hr_scale": hr_scale,
        "rmssd_center": rmssd_center,
        "rmssd_scale": rmssd_scale,
    }
    if summaries[-1].get("eligible") is not True:
        return _vector_result(
            status="NOT_DECISIONABLE",
            reason_code="CURRENT_WINDOW_INELIGIBLE",
            baseline_ready=True,
            eligible_seconds=eligible_seconds,
            baseline=baseline,
            vector=None,
        )
    if eligible_seconds < 1800:
        return _vector_result(
            status="NOT_DECISIONABLE",
            reason_code="DECISION_WARMUP_INCOMPLETE",
            baseline_ready=True,
            eligible_seconds=eligible_seconds,
            baseline=baseline,
            vector=None,
        )

    hr_seconds: list[float | None] = []
    rmssd_seconds: list[float | None] = []
    load_seconds: list[float | None] = []
    eligible_flags: list[bool] = []
    quality_by_summary: list[float | None] = []
    for summary in summaries:
        hr = _summary_value(summary, "hr_bpm_median")
        rmssd = _summary_value(summary, "rr_rmssd_ms")
        quality = _summary_value(summary, "quality_confidence")
        is_eligible = (
            summary.get("eligible") is True
            and summary.get("eligible_seconds") == 10
            and hr is not None
            and rmssd is not None
            and quality is not None
            and quality >= 0.9
        )
        if is_eligible and hr is not None and rmssd is not None:
            hr_z = min(20.0, max(-20.0, (hr - hr_center) / hr_scale))
            rmssd_z = min(20.0, max(-20.0, -(rmssd - rmssd_center) / rmssd_scale))
            load = (max(hr_z, 0.0) + max(rmssd_z, 0.0)) / 2.0
        else:
            hr_z = None
            rmssd_z = None
            load = None
        hr_seconds.extend([hr_z] * 10)
        rmssd_seconds.extend([rmssd_z] * 10)
        load_seconds.extend([load] * 10)
        eligible_flags.extend([is_eligible] * 10)
        quality_by_summary.append(quality if is_eligible else None)

    hr_30 = _covered_values(hr_seconds[-30:])
    rmssd_30 = _covered_values(rmssd_seconds[-30:])
    load_60 = _covered_values(load_seconds[-60:])
    load_300 = _covered_values(load_seconds[-300:])
    hr_slope = _ols_slope(hr_seconds[-60:])
    rmssd_slope = _ols_slope(rmssd_seconds[-60:])
    ema_1800 = _bounded_ema(load_seconds, 1800)
    ema_21600 = _bounded_ema(load_seconds, 21600)
    current_hr = hr_seconds[-1]
    current_rmssd = rmssd_seconds[-1]
    current_load = load_seconds[-1]
    current_quality = quality_by_summary[-1]
    required = (
        hr_30,
        rmssd_30,
        load_60,
        load_300,
        hr_slope,
        rmssd_slope,
        ema_1800,
        ema_21600,
        current_hr,
        current_rmssd,
        current_load,
        current_quality,
    )
    if any(value is None for value in required):
        return _vector_result(
            status="NOT_DECISIONABLE",
            reason_code="ROLLING_COVERAGE_INSUFFICIENT",
            baseline_ready=True,
            eligible_seconds=eligible_seconds,
            baseline=baseline,
            vector=None,
        )
    assert hr_30 is not None and rmssd_30 is not None
    assert load_60 is not None and load_300 is not None
    assert hr_slope is not None and rmssd_slope is not None
    assert ema_1800 is not None and ema_21600 is not None
    assert current_hr is not None and current_rmssd is not None
    assert current_load is not None and current_quality is not None
    load_mean = sum(load_60) / len(load_60)
    load_std = math.sqrt(sum((value - load_mean) ** 2 for value in load_60) / len(load_60))
    vector = [
        current_hr,
        current_rmssd,
        current_load,
        sum(hr_30) / len(hr_30),
        sum(rmssd_30) / len(rmssd_30),
        hr_slope,
        rmssd_slope,
        load_std,
        float(statistics.median(load_300)),
        ema_1800,
        ema_21600,
        current_quality,
        1.0 - sum(eligible_flags[-60:]) / 60.0,
    ]
    if not all(math.isfinite(value) for value in vector):
        return _vector_result(
            status="NOT_DECISIONABLE",
            reason_code="FEATURE_NONFINITE",
            baseline_ready=True,
            eligible_seconds=eligible_seconds,
            baseline=baseline,
            vector=None,
        )
    return _vector_result(
        status="DECISIONABLE",
        reason_code="DECISIONABLE",
        baseline_ready=True,
        eligible_seconds=eligible_seconds,
        baseline=baseline,
        vector=vector,
    )
