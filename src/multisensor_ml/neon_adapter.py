"""Fail-closed adapter from Neon Watch observations to personal baselines.

The adapter is deliberately *not* a model retrainer.  It consumes decoded
Galaxy Watch rows, requires an externally corrected UTC clock, computes
observed 1 Hz features and a causal personal baseline, and records a bounded
personal weight.  Without independent observer labels it cannot change the
coefficients of the synthetic standard model or produce model-ready event
predictions.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Final

import numpy as np
import pandas as pd

ADAPTER_SCHEMA_VERSION: Final[str] = "goal1.5/neon-observed-adapter/v1"
DEFAULT_PRIOR_SECONDS: Final[float] = 1_800.0
DEFAULT_WEIGHT_CAP: Final[float] = 0.75
_REQUIRED_MAPPED_COLUMNS: Final[frozenset[str]] = frozenset(
    {"session_id", "sensor_kind", "sensor_timestamp_utc"}
)
_DEFAULT_VALUE_COLUMNS: Final[tuple[str, ...]] = (
    "observed__heart_rate_bpm",
    "observed__eda_us",
    "observed__motion_magnitude",
    "observed__ppg_mean",
    "observed__skin_temperature_c",
)


@dataclass(frozen=True, slots=True)
class PersonalBaselineAdapter:
    """A bounded, label-free personal baseline calibration record."""

    person_id: str
    centers: dict[str, float]
    scales: dict[str, float]
    n_eff: int
    quality_confidence: float
    personal_weight: float
    weight_cap: float
    warmup_seconds: int
    status: str
    model_weights_changed: bool = False
    source: str = "neon-observed"
    schema_version: str = ADAPTER_SCHEMA_VERSION


def _require_corrected_time(frame: pd.DataFrame) -> pd.Series:
    if "corrected_utc" not in frame.columns:
        raise ValueError(
            "corrected_utc is required; received sensor time is not a corrected model clock"
        )
    corrected = pd.to_datetime(frame["corrected_utc"], utc=True, errors="coerce")
    if corrected.isna().all():
        raise ValueError("corrected_utc contains no valid timestamps")
    return corrected


def _quality_from_status(frame: pd.DataFrame) -> pd.Series:
    status_columns = [
        column
        for column in ("heart_rate_status", "status", "green_status", "ir_status", "red_status")
        if column in frame.columns
    ]
    if not status_columns:
        return pd.Series(1.0, index=frame.index, dtype="float64")
    values = frame[status_columns].apply(pd.to_numeric, errors="coerce")
    available = values.notna().sum(axis=1)
    valid = (values == 0).sum(axis=1)
    quality = valid / available.replace(0, np.nan)
    return pd.Series(quality.fillna(0.0).clip(0.0, 1.0), index=frame.index)


def derive_watch_features(
    mapped_samples: pd.DataFrame,
    *,
    clock_offset_ms: float | None = None,
    physiological_lag_ms: float | None = None,
) -> pd.DataFrame:
    """Aggregate decoded Watch rows onto a corrected 1 Hz time axis.

    ``clock_offset_ms`` is accepted only as an externally measured offset.  It
    is never estimated from ``received_at`` because network delay and
    physiological lag are different quantities.
    """

    missing = _REQUIRED_MAPPED_COLUMNS.difference(mapped_samples.columns)
    if missing:
        raise ValueError(f"mapped samples missing columns: {sorted(missing)}")
    sensor_time = pd.to_datetime(mapped_samples["sensor_timestamp_utc"], utc=True, errors="coerce")
    if sensor_time.isna().all():
        raise ValueError("sensor_timestamp_utc contains no valid timestamps")
    if clock_offset_ms is None:
        # A corrected_utc column may be supplied by a future SQL view.  The
        # mapper itself intentionally keeps source time separate.
        if "corrected_utc" not in mapped_samples.columns:
            raise ValueError("corrected_utc or externally measured clock_offset_ms is required")
        corrected = _require_corrected_time(mapped_samples)
        applied_offset = pd.Series(0.0, index=mapped_samples.index)
    else:
        corrected = sensor_time + pd.to_timedelta(float(clock_offset_ms), unit="ms")
        applied_offset = pd.Series(float(clock_offset_ms), index=mapped_samples.index)

    work = mapped_samples.copy()
    work["corrected_utc"] = corrected
    work["clock_offset_ms"] = applied_offset
    work["physiological_lag_ms"] = (
        float(physiological_lag_ms) if physiological_lag_ms is not None else np.nan
    )
    work["quality_confidence"] = _quality_from_status(work)
    work["time_bucket"] = work["corrected_utc"].dt.floor("s")

    numeric = {
        "observed__heart_rate_bpm": "heart_rate_bpm",
        "observed__eda_us": "skin_conductance_us",
        "observed__skin_temperature_c": "object_temperature_c",
    }
    for output_column, input_column in numeric.items():
        if input_column in work.columns:
            work[output_column] = pd.to_numeric(work[input_column], errors="coerce")
        else:
            work[output_column] = np.nan
    if {"raw_x", "raw_y", "raw_z"}.issubset(work.columns):
        x = pd.to_numeric(work["raw_x"], errors="coerce")
        y = pd.to_numeric(work["raw_y"], errors="coerce")
        z = pd.to_numeric(work["raw_z"], errors="coerce")
        work["observed__motion_magnitude"] = np.sqrt(x * x + y * y + z * z)
    else:
        work["observed__motion_magnitude"] = np.nan
    ppg_columns = [
        column for column in ("raw_green", "raw_ir", "raw_red") if column in work.columns
    ]
    if ppg_columns:
        work["observed__ppg_mean"] = (
            work[ppg_columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        )
    else:
        work["observed__ppg_mean"] = np.nan

    group_columns = ["session_id", "time_bucket"]
    aggregations: dict[str, str] = {
        column: "mean"
        for column in _DEFAULT_VALUE_COLUMNS
        if column in work.columns
    }
    aggregations["quality_confidence"] = "mean"
    aggregations["clock_offset_ms"] = "mean"
    aggregations["physiological_lag_ms"] = "mean"
    result = work.groupby(group_columns, sort=True, observed=True).agg(aggregations).reset_index()
    result = result.rename(columns={"time_bucket": "corrected_utc"})
    present = (
        work.assign(_present=1)
        .groupby(group_columns, sort=True, observed=True)["_present"]
        .sum()
    )
    result["sample_count"] = present.to_numpy(dtype="int64")
    result["sensor_kind_count"] = (
        work.groupby(group_columns, sort=True, observed=True)["sensor_kind"]
        .nunique()
        .to_numpy(dtype="int64")
    )
    result["missing_sensor_count"] = (
        work["sensor_kind"].nunique() - result["sensor_kind_count"]
    ).clip(lower=0)
    result["adapter_status"] = "OBSERVED_BASELINE_ONLY"
    result["real_data_status"] = "NOT VERIFIED"
    return result.sort_values(["session_id", "corrected_utc"], kind="stable").reset_index(drop=True)


def _robust_scale(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return 1.0
    median = float(numeric.median())
    mad = float((numeric - median).abs().median())
    scale = 1.4826 * mad
    if not math.isfinite(scale) or scale < 1e-6:
        std = float(numeric.std(ddof=0))
        scale = std if math.isfinite(std) and std >= 1e-6 else 1.0
    return scale


def fit_personal_baseline_adapter(
    frame: pd.DataFrame,
    *,
    person_id: str,
    value_columns: Iterable[str] = _DEFAULT_VALUE_COLUMNS,
    warmup_seconds: int = 1_800,
    weight_cap: float = DEFAULT_WEIGHT_CAP,
) -> PersonalBaselineAdapter:
    """Fit a causal personal center/scale using only the initial warm-up window."""

    if warmup_seconds <= 0:
        raise ValueError("warmup_seconds must be positive")
    if not 0.0 <= weight_cap <= 1.0:
        raise ValueError("weight_cap must be between 0 and 1")
    corrected = _require_corrected_time(frame)
    values = tuple(dict.fromkeys(str(column) for column in value_columns))
    missing = [column for column in values if column not in frame.columns]
    if missing:
        raise ValueError(f"baseline value columns are missing: {missing}")
    valid_times = corrected.dropna()
    if valid_times.empty:
        raise ValueError("no valid corrected timestamps for personal baseline")
    start = valid_times.min()
    end = start + pd.Timedelta(seconds=warmup_seconds)
    baseline = frame.loc[corrected.between(start, end, inclusive="left")].copy()
    quality_source: object
    if "quality_confidence" in baseline.columns:
        quality_source = baseline["quality_confidence"]
    else:
        quality_source = pd.Series(1.0, index=baseline.index, dtype="float64")
    quality = pd.to_numeric(quality_source, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}
    valid_total = 0
    for column in values:
        numeric = pd.to_numeric(baseline[column], errors="coerce")
        valid = numeric.notna()
        valid_total += int(valid.sum())
        centers[column] = float(numeric[valid].median()) if valid.any() else 0.0
        scales[column] = _robust_scale(numeric)
    n_eff = valid_total
    quality_confidence = float(quality.mean()) if len(quality) else 0.0
    personal_weight = min(
        float(weight_cap),
        float(n_eff) / (float(n_eff) + DEFAULT_PRIOR_SECONDS) * quality_confidence,
    )
    status = "READY_FOR_BASELINE_ONLY" if valid_times.max() >= end else "WARMUP_INCOMPLETE"
    if len(baseline) == 0 or n_eff == 0:
        status = "INSUFFICIENT_OBSERVED_SUPPORT"
    return PersonalBaselineAdapter(
        person_id=str(person_id),
        centers=centers,
        scales=scales,
        n_eff=n_eff,
        quality_confidence=max(0.0, min(1.0, quality_confidence)),
        personal_weight=max(0.0, min(float(weight_cap), personal_weight)),
        weight_cap=float(weight_cap),
        warmup_seconds=int(warmup_seconds),
        status=status,
    )


def transform_with_personal_adapter(
    frame: pd.DataFrame,
    adapter: PersonalBaselineAdapter,
) -> pd.DataFrame:
    """Append robust deviations and bounded weight without changing model coefficients."""

    _require_corrected_time(frame)
    result = frame.copy()
    for column, center in adapter.centers.items():
        if column not in result.columns:
            raise ValueError(f"adapter value column is missing: {column}")
        values = pd.to_numeric(result[column], errors="coerce")
        result[f"adapter__{column}__robust_z"] = (values - center) / adapter.scales[column]
    result["personal_weight"] = adapter.personal_weight
    result["adapter_status"] = adapter.status
    result["model_weights_changed"] = False
    result["real_data_status"] = "NOT VERIFIED"
    return result


def personal_adapter_manifest(
    adapter: PersonalBaselineAdapter,
    *,
    model_version: str,
) -> dict[str, object]:
    """Create an auditable manifest that blocks label-free promotion."""

    result: dict[str, object] = {
        "schema_version": adapter.schema_version,
        "source": adapter.source,
        "person_id": adapter.person_id,
        "model_version": model_version,
        "status": adapter.status,
        "real_data_status": "NOT VERIFIED",
        "model_weights_changed": False,
        "promotable": False,
        "promotion_block_reason": "NO_OBSERVED_LABELS",
        "clock_sync_status": "EXTERNAL_CORRECTED_UTC_REQUIRED",
        "label_support": "NONE",
    }
    result.update(asdict(adapter))
    result["schema_version"] = adapter.schema_version
    result["source"] = adapter.source
    result["promotable"] = False
    result["promotion_block_reason"] = "NO_OBSERVED_LABELS"
    return result
