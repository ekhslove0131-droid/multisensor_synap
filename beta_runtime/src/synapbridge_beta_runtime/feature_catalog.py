"""Versioned beta feature catalogue and schema digests."""

from __future__ import annotations

import hashlib

from .codec import canonical_bytes

LOOKBACKS = (30, 60, 180)
BASE_FEATURES = {
    "watch": ("hr_median", "ibi_rmssd", "eda_median", "eda_slope",
              "ppg_amp", "temp_median", "temp_change"),
    "h10": ("hr_median", "rr_rmssd", "rr_sdnn", "rr_change"),
    "muse": tuple(f"eeg_{channel}_{name}" for channel in range(1, 5)
                  for name in ("theta_rel", "alpha_rel", "beta_rel", "log_power",
                               "power_change")),
}
CONTEXT_FEATURES = {family: ("motion_var",) for family in BASE_FEATURES}

_DESCRIPTORS = {
    "hr_median": ("bpm", "median_from_observed_intervals", 2, "nonnegative"),
    "ibi_rmssd": ("ms", "rmssd_from_observed_intervals", 2, "nonnegative"),
    "rr_rmssd": ("ms", "rmssd_from_observed_intervals", 2, "nonnegative"),
    "rr_sdnn": ("ms", "sample_sd_from_observed_intervals", 2, "nonnegative"),
    "rr_change": ("ms_per_s", "endpoint_delta_over_elapsed_valid_seconds", 2,
                  "signed"),
    "eda_median": ("uS", "median", None, "nonnegative"),
    "eda_slope": ("uS_per_s", "ols_slope_over_sample_time", None, "signed"),
    "ppg_amp": ("a.u.", "p95_minus_p05", None, "nonnegative"),
    "temp_median": ("degC", "median", None, "signed"),
    "temp_change": ("degC_per_s", "ols_slope_over_sample_time", None, "signed"),
    "theta_rel": ("ratio", "welch_band_power_over_1_40hz", None, "nonnegative"),
    "alpha_rel": ("ratio", "welch_band_power_over_1_40hz", None, "nonnegative"),
    "beta_rel": ("ratio", "welch_band_power_over_1_40hz", None, "nonnegative"),
    "log_power": ("log_uV2", "log_welch_power_1_40hz", None, "signed"),
    "power_change": ("log_uV2_per_s", "half_window_delta_over_elapsed_seconds",
                     None, "signed"),
    "motion_var": ("g2", "variance_of_acceleration_magnitude", None, "nonnegative"),
}


def feature_order(family: str) -> tuple[str, ...]:
    if family not in BASE_FEATURES:
        raise ValueError("feature_family_invalid")
    return tuple(f"{name}_{window}" for window in LOOKBACKS
                 for name in (*BASE_FEATURES[family], *CONTEXT_FEATURES[family]))


def standard_feature_order(family: str) -> tuple[str, ...]:
    return tuple(f"{name}_180" for name in BASE_FEATURES[family])


def feature_contract(family: str) -> dict[str, object]:
    order = feature_order(family)
    descriptors: dict[str, object] = {}
    bases = (*BASE_FEATURES[family], *CONTEXT_FEATURES[family])
    for window in LOOKBACKS:
        for base in bases:
            descriptor_key = next(
                (key for key in _DESCRIPTORS if base == key or base.endswith(f"_{key}")),
                None,
            )
            if descriptor_key is None:
                raise ValueError("feature_descriptor_missing")
            unit, transform, minimum_intervals, direction = _DESCRIPTORS[descriptor_key]
            descriptors[f"{base}_{window}"] = {
                "unit": unit,
                "transform": transform,
                "window_seconds": window,
                "minimum_intervals": minimum_intervals,
                "availability": "PAST_HALF_OPEN_AVAILABLE_AT_DECISION",
                "direction": direction,
            }
    body = {"version": "synthetic_beta_features_v1", "family": family,
            "lookbacks_seconds": LOOKBACKS, "order": order, "features": descriptors,
            "support": "PAST_HALF_OPEN",
            "valid_support_kind": "QUALITY_QUALIFIED_FEATURE_STRIDE",
            "coverage_min": 0.8, "max_gap_seconds": 6}
    return {**body, "sha256": hashlib.sha256(canonical_bytes(body)).hexdigest()}


def schema_sha(family: str) -> str:
    return str(feature_contract(family)["sha256"])
