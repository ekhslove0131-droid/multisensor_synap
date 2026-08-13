"""Canonical real-Watch observational feature and baseline-shadow contract.

This module is intentionally independent from synthetic ``monitoring_v2``.
It accepts corrected-UTC, one-hertz Watch observations and never consumes
event truth, behavior labels, or hidden synthetic factors.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import numpy as np
import pandas as pd

MODEL_RELEASE: Final[str] = "observational_standard_30m_watch_baseline_shadow_v1"
BASELINE_VERSION: Final[str] = "personal_robust_baseline_900s_v2"
LOAD_FORMULA_VERSION: Final[str] = "watch_positive_z_bounded_ema_v2"
FEATURE_SCHEMA_VERSION: Final[str] = "watch_standard_16_v2"
QUALITY_FORMULA_VERSION: Final[str] = "watch_core_quality_min_v1"
TARGET_VERSION: Final[str] = "observational_future_mean_1800_valid90_gap5_v2"
LEGACY_15_FEATURE_STATUS: Final[str] = "legacy_not_for_serving"

BASELINE_WARMUP_SEC: Final[int] = 900
BASELINE_MAX_SPAN_SEC: Final[int] = 1_000
DECISION_WARMUP_SEC: Final[int] = 1_800
FORECAST_HORIZON_SEC: Final[int] = 1_800
PURGE_SEC: Final[int] = 1_800
QUALITY_GATE: Final[float] = 0.70
MIN_VALID_FRACTION: Final[float] = 0.90
MAX_GAP_SEC: Final[int] = 5

SIGNAL_COLUMNS: Final[tuple[str, ...]] = (
    "observed__eda_us",
    "observed__heart_rate_bpm",
    "observed__motion_magnitude",
)
Z_COLUMNS: Final[tuple[str, ...]] = (
    "watch_eda_z",
    "watch_hr_z",
    "watch_motion_z",
)
FEATURE_NAMES: Final[tuple[str, ...]] = (
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

EMA_ALPHA_1800: Final[float] = 1.0 - math.exp(-math.log(2.0) / 1_800.0)
EMA_ALPHA_21600: Final[float] = 1.0 - math.exp(-math.log(2.0) / 21_600.0)


def _feature(
    name: str,
    formula: str,
    history: str,
    *,
    unit: str,
) -> dict[str, str]:
    return {"name": name, "formula": formula, "history": history, "unit": unit}


def canonical_feature_schema() -> dict[str, object]:
    """Return the immutable JSON-compatible Watch feature contract."""

    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "input": {"name": "features", "dtype": "float32", "shape": [None, 16]},
        "ordered_feature_names": list(FEATURE_NAMES),
        "features": [
            _feature(
                "watch_eda_z",
                "clip((eda_t-baseline_median_eda)/baseline_scale_eda,-20,20)",
                "current eligible second",
                unit="robust_z",
            ),
            _feature(
                "watch_hr_z",
                "clip((hr_t-baseline_median_hr)/baseline_scale_hr,-20,20)",
                "current eligible second",
                unit="robust_z",
            ),
            _feature(
                "watch_motion_z",
                "clip((motion_t-baseline_median_motion)/baseline_scale_motion,-20,20)",
                "current eligible second",
                unit="robust_z",
            ),
            _feature(
                "watch_load_raw",
                "mean(max(watch_eda_z,0),max(watch_hr_z,0),max(watch_motion_z,0))",
                "current eligible second; quality is not multiplied",
                unit="positive_robust_z",
            ),
            _feature(
                "watch_eda_mean_30",
                "mean(watch_eda_z[t-30:t])",
                "strict past; at least 27 eligible seconds",
                unit="robust_z",
            ),
            _feature(
                "watch_hr_mean_30",
                "mean(watch_hr_z[t-30:t])",
                "strict past; at least 27 eligible seconds",
                unit="robust_z",
            ),
            _feature(
                "watch_motion_mean_30",
                "mean(watch_motion_z[t-30:t])",
                "strict past; at least 27 eligible seconds",
                unit="robust_z",
            ),
            _feature(
                "watch_eda_slope_60",
                "corrected_utc_ols_slope(watch_eda_z[t-60:t])",
                "strict past; at least 54 eligible seconds",
                unit="robust_z_per_second",
            ),
            _feature(
                "watch_hr_slope_60",
                "corrected_utc_ols_slope(watch_hr_z[t-60:t])",
                "strict past; at least 54 eligible seconds",
                unit="robust_z_per_second",
            ),
            _feature(
                "watch_motion_slope_60",
                "corrected_utc_ols_slope(watch_motion_z[t-60:t])",
                "strict past; at least 54 eligible seconds",
                unit="robust_z_per_second",
            ),
            _feature(
                "watch_load_std_60",
                "population_std(watch_load_raw[t-60:t],ddof=0)",
                "strict past; at least 54 eligible seconds",
                unit="positive_robust_z",
            ),
            _feature(
                "watch_load_median_300",
                "median(watch_load_raw[t-300:t])",
                "strict past; at least 270 eligible seconds",
                unit="positive_robust_z",
            ),
            _feature(
                "watch_load_ema_1800",
                "ema(load_t,alpha=1-2**(-1/1800))",
                "eligible-second replay including current second",
                unit="positive_robust_z",
            ),
            _feature(
                "watch_load_ema_21600",
                "ema(load_t,alpha=1-2**(-1/21600))",
                "eligible-second replay including current second",
                unit="positive_robust_z",
            ),
            _feature(
                "quality_confidence",
                "min(core_sensor_coverage*core_sensor_validity)",
                "current HR+EDA+ACC SensorSummary; missing is invalid",
                unit="fraction",
            ),
            _feature(
                "watch_ineligible_fraction_60",
                "1-sum(eligible[t-60:t])/60",
                "strict past 60 corrected-UTC seconds",
                unit="fraction",
            ),
        ],
        "versions": {
            "baseline_version": BASELINE_VERSION,
            "load_formula_version": LOAD_FORMULA_VERSION,
            "quality_formula_version": QUALITY_FORMULA_VERSION,
            "target_version": TARGET_VERSION,
        },
        "quality_contract": {
            "required_core": ["HR", "EDA", "ACC"],
            "gate": QUALITY_GATE,
            "missing_quality": "NOT_DECISIONABLE",
            "quality_multiplies_load": False,
            "initial_expected_samples_per_second": {"HR": 1, "EDA": 1, "ACC": 25},
        },
        "baseline_contract": {
            "eligible_seconds": BASELINE_WARMUP_SEC,
            "max_wall_seconds": BASELINE_MAX_SPAN_SEC,
            "max_consecutive_gap_seconds": MAX_GAP_SEC,
            "scale": "1.4826*MAD; fallback IQR/1.349; otherwise BASELINE_SCALE_ZERO",
            "freeze": True,
        },
        "decision_contract": {
            "total_eligible_seconds": DECISION_WARMUP_SEC,
            "baseline_seconds_are_included": True,
            "non_finite_feature": "NOT_DECISIONABLE",
        },
        "legacy_15_feature_status": LEGACY_15_FEATURE_STATUS,
    }


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


FEATURE_SCHEMA_SHA256: Final[str] = hashlib.sha256(
    _canonical_json(canonical_feature_schema())
).hexdigest()


@dataclass(frozen=True, slots=True)
class ObservationalStandardConfig:
    random_state: int = 20260804
    baseline_warmup_sec: int = BASELINE_WARMUP_SEC
    baseline_max_span_sec: int = BASELINE_MAX_SPAN_SEC
    decision_warmup_sec: int = DECISION_WARMUP_SEC
    forecast_horizon_sec: int = FORECAST_HORIZON_SEC
    purge_sec: int = PURGE_SEC
    quality_gate: float = QUALITY_GATE
    min_valid_fraction: float = MIN_VALID_FRACTION
    max_gap_sec: int = MAX_GAP_SEC
    min_train_rows: int = 100
    min_validation_rows: int = 100
    min_relative_improvement: float = 0.02
    hgb_max_iter: int = 120

    def validate(self) -> None:
        if self.baseline_warmup_sec <= 0:
            raise ValueError("baseline_warmup_sec must be positive")
        if self.baseline_max_span_sec < self.baseline_warmup_sec:
            raise ValueError("baseline_max_span_sec must cover baseline eligible seconds")
        if self.decision_warmup_sec < self.baseline_warmup_sec:
            raise ValueError("decision_warmup_sec must include baseline warm-up")
        if self.forecast_horizon_sec <= 0:
            raise ValueError("forecast_horizon_sec must be positive")
        if self.purge_sec < self.forecast_horizon_sec:
            raise ValueError("purge_sec must cover the forecast horizon")
        if not 0.0 < self.quality_gate <= 1.0:
            raise ValueError("quality_gate must be in (0, 1]")
        if not 0.0 < self.min_valid_fraction <= 1.0:
            raise ValueError("min_valid_fraction must be in (0, 1]")
        if self.max_gap_sec < 0:
            raise ValueError("max_gap_sec must be non-negative")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _require_input_columns(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"corrected_utc", "quality_confidence", *SIGNAL_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("missing required observational columns: " + ", ".join(missing))
    output = frame.copy()
    if "person_key" not in output.columns:
        if "session_id" not in output.columns:
            raise ValueError("observational input requires person_key or session_id")
        output["person_key"] = output["session_id"].astype(str)
    if "session_id" not in output.columns:
        output["session_id"] = output["person_key"].astype(str)
    output["corrected_utc"] = pd.to_datetime(
        output["corrected_utc"], utc=True, errors="coerce"
    )
    if output["corrected_utc"].isna().any():
        raise ValueError("corrected_utc contains invalid or missing timestamps")
    for column in (*SIGNAL_COLUMNS, "quality_confidence"):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output["quality_confidence"] = output["quality_confidence"].clip(0.0, 1.0)
    return output.sort_values(
        ["person_key", "session_id", "corrected_utc"], kind="stable"
    ).reset_index(drop=True)


def _resample_one_group(group: pd.DataFrame) -> pd.DataFrame:
    person_key = str(group["person_key"].iloc[0])
    session_id = str(group["session_id"].iloc[0])
    indexed = group.set_index("corrected_utc").sort_index()
    start = cast(pd.Timestamp, indexed.index.min()).floor("s")
    end = cast(pd.Timestamp, indexed.index.max()).floor("s")
    grid = pd.date_range(start, end, freq="1s", tz="UTC")
    values = indexed[[*SIGNAL_COLUMNS, "quality_confidence"]].resample("1s").mean()
    values = values.reindex(grid).reset_index(names="corrected_utc")
    values.insert(0, "session_id", session_id)
    values.insert(0, "person_key", person_key)
    return values


def _find_baseline_indices(
    eligible: np.ndarray,
    *,
    required: int,
    max_span: int,
    max_gap: int,
) -> tuple[int, int, np.ndarray] | None:
    start: int | None = None
    count = 0
    gap = 0
    for index, is_eligible in enumerate(eligible):
        if not is_eligible:
            if start is not None:
                gap += 1
                if gap > max_gap:
                    start = None
                    count = 0
                    gap = 0
            continue
        if start is None:
            start = index
            count = 1
            gap = 0
        else:
            count += 1
            gap = 0
        if index - start + 1 > max_span:
            start = index
            count = 1
        if count >= required:
            selected = np.flatnonzero(eligible[start : index + 1]) + start
            return start, index, selected[:required]
    return None


def _robust_center_scale(values: np.ndarray) -> tuple[float, float] | None:
    numeric = values[np.isfinite(values)]
    if numeric.size == 0:
        return None
    center = float(np.median(numeric))
    mad = float(np.median(np.abs(numeric - center)))
    scale = 1.4826 * mad
    epsilon = max(1e-12, abs(center) * 1e-9)
    if not math.isfinite(scale) or scale <= epsilon:
        q1, q3 = np.quantile(numeric, [0.25, 0.75])
        scale = float((q3 - q1) / 1.349)
    if not math.isfinite(scale) or scale <= epsilon:
        return None
    return center, scale


def bounded_ema_eligible(
    values: np.ndarray,
    eligible: np.ndarray,
    *,
    half_life_seconds: float,
    initial: float,
) -> np.ndarray:
    """Update a bounded EMA only on eligible rows and hold state otherwise."""

    if half_life_seconds <= 0:
        raise ValueError("half_life_seconds must be positive")
    if values.shape != eligible.shape:
        raise ValueError("values and eligible must have identical shape")
    alpha = 1.0 - math.exp(-math.log(2.0) / half_life_seconds)
    output = np.empty(values.shape, dtype="float64")
    state = float(initial)
    for index, value in enumerate(values):
        if bool(eligible[index]) and math.isfinite(float(value)):
            state = (1.0 - alpha) * state + alpha * float(value)
        output[index] = state
    return output


def _rolling_ols_slope(values: pd.Series, window: int, minimum: int) -> pd.Series:
    shifted = pd.to_numeric(values, errors="coerce").shift(1)

    def slope(window_values: np.ndarray) -> float:
        valid = np.isfinite(window_values)
        if int(valid.sum()) < minimum:
            return math.nan
        x = np.arange(window_values.size, dtype="float64")[valid]
        y = window_values[valid]
        x_centered = x - x.mean()
        denominator = float(np.sum(x_centered**2))
        if denominator <= 0.0:
            return math.nan
        return float(np.sum(x_centered * (y - y.mean())) / denominator)

    return shifted.rolling(window, min_periods=minimum).apply(slope, raw=True)


def _longest_false_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in values:
        if bool(value):
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _future_target(
    load: np.ndarray,
    eligible: np.ndarray,
    *,
    horizon: int,
    minimum_fraction: float,
    max_gap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = eligible & np.isfinite(load)
    sums = np.where(valid, load, 0.0)
    cumulative_sum = np.concatenate(([0.0], np.cumsum(sums, dtype="float64")))
    cumulative_count = np.concatenate(([0], np.cumsum(valid, dtype="int64")))
    target = np.full(load.size, np.nan, dtype="float64")
    fraction = np.zeros(load.size, dtype="float64")
    maximum_gap = np.zeros(load.size, dtype="int64")
    target_valid = np.zeros(load.size, dtype="int8")
    minimum_count = math.ceil(horizon * minimum_fraction)
    for index in range(load.size):
        start = index + 1
        end = start + horizon
        if end > load.size:
            continue
        count = int(cumulative_count[end] - cumulative_count[start])
        fraction[index] = count / horizon
        gap = _longest_false_run(valid[start:end])
        maximum_gap[index] = gap
        if count >= minimum_count and gap <= max_gap:
            target[index] = float(
                (cumulative_sum[end] - cumulative_sum[start]) / count
            )
            target_valid[index] = 1
    return target, fraction, maximum_gap, target_valid


def _dynamic_ema_and_count(
    load: np.ndarray,
    eligible: np.ndarray,
    *,
    start: int,
    initial: float,
    max_gap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ema_1800 = np.full(load.size, np.nan, dtype="float64")
    ema_21600 = np.full(load.size, np.nan, dtype="float64")
    count = np.zeros(load.size, dtype="int64")
    state_1800 = initial
    state_21600 = initial
    eligible_count = 0
    gap = 0
    for index in range(start, load.size):
        if bool(eligible[index]) and math.isfinite(float(load[index])):
            gap = 0
            eligible_count += 1
            state_1800 = (
                (1.0 - EMA_ALPHA_1800) * state_1800
                + EMA_ALPHA_1800 * float(load[index])
            )
            state_21600 = (
                (1.0 - EMA_ALPHA_21600) * state_21600
                + EMA_ALPHA_21600 * float(load[index])
            )
        else:
            gap += 1
            if gap > max_gap:
                eligible_count = 0
                state_1800 = initial
                state_21600 = initial
        ema_1800[index] = state_1800
        ema_21600[index] = state_21600
        count[index] = eligible_count
    return ema_1800, ema_21600, count


def _build_one_group(
    group: pd.DataFrame,
    *,
    config: ObservationalStandardConfig,
) -> pd.DataFrame:
    result = group.sort_values("corrected_utc", kind="stable").reset_index(drop=True).copy()
    core_present = result[list(SIGNAL_COLUMNS)].notna().all(axis=1).to_numpy(dtype=bool)
    quality_known = result["quality_confidence"].notna().to_numpy(dtype=bool)
    quality_ok = (
        result["quality_confidence"].fillna(-1.0).to_numpy(dtype="float64")
        >= config.quality_gate
    )
    eligible = core_present & quality_known & quality_ok
    result["quality_ok"] = quality_ok.astype("int8")
    result["core_present"] = core_present.astype("int8")
    result["eligible"] = eligible.astype("int8")

    baseline = _find_baseline_indices(
        eligible,
        required=config.baseline_warmup_sec,
        max_span=config.baseline_max_span_sec,
        max_gap=config.max_gap_sec,
    )
    baseline_status = "INSUFFICIENT_BASELINE"
    baseline_ready = np.zeros(len(result), dtype="int8")
    centers: list[float] = []
    scales: list[float] = []
    baseline_start = 0
    baseline_freeze = len(result)
    baseline_indices = np.array([], dtype="int64")
    if baseline is not None:
        baseline_start, baseline_freeze, baseline_indices = baseline
        for signal in SIGNAL_COLUMNS:
            estimate = _robust_center_scale(
                result.loc[baseline_indices, signal].to_numpy(dtype="float64")
            )
            if estimate is None:
                baseline_status = f"BASELINE_SCALE_ZERO:{signal}"
                centers = []
                scales = []
                break
            center, scale = estimate
            centers.append(center)
            scales.append(scale)
        if centers:
            baseline_status = "READY"
            baseline_ready[baseline_freeze:] = 1
    result["baseline_status"] = baseline_status
    result["baseline_ready"] = baseline_ready
    result["baseline_start_index"] = baseline_start if centers else -1
    result["baseline_freeze_index"] = baseline_freeze if centers else -1

    for z_name in Z_COLUMNS:
        result[z_name] = np.nan
    result["watch_load_raw"] = np.nan
    for feature in FEATURE_NAMES[4:14]:
        result[feature] = np.nan
    result["watch_ineligible_fraction_60"] = np.nan
    result["eligible_count_since_reset"] = 0
    result["decision_warmup_complete"] = 0
    result["feature_decisionable"] = 0

    if centers:
        for source, z_name, center, scale in zip(
            SIGNAL_COLUMNS, Z_COLUMNS, centers, scales, strict=True
        ):
            result[z_name] = ((result[source] - center) / scale).clip(-20.0, 20.0)
            safe_name = source.replace("observed__", "")
            result[f"baseline_center__{safe_name}"] = center
            result[f"baseline_scale__{safe_name}"] = scale
        positive = result[list(Z_COLUMNS)].clip(lower=0.0)
        result["watch_load_raw"] = positive.mean(axis=1).where(core_present)
        gated_load = result["watch_load_raw"].where(eligible)
        for z_name, mean_name, slope_name in zip(
            Z_COLUMNS,
            ("watch_eda_mean_30", "watch_hr_mean_30", "watch_motion_mean_30"),
            ("watch_eda_slope_60", "watch_hr_slope_60", "watch_motion_slope_60"),
            strict=True,
        ):
            gated_signal = result[z_name].where(eligible)
            result[mean_name] = gated_signal.shift(1).rolling(30, min_periods=27).mean()
            result[slope_name] = _rolling_ols_slope(gated_signal, 60, 54)
        result["watch_load_std_60"] = (
            gated_load.shift(1).rolling(60, min_periods=54).std(ddof=0)
        )
        result["watch_load_median_300"] = (
            gated_load.shift(1).rolling(300, min_periods=270).median()
        )
        baseline_load = gated_load.iloc[baseline_indices].dropna().to_numpy(dtype="float64")
        initial_load = float(np.median(baseline_load))
        ema_1800, ema_21600, dynamic_count = _dynamic_ema_and_count(
            result["watch_load_raw"].to_numpy(dtype="float64"),
            eligible,
            start=baseline_start,
            initial=initial_load,
            max_gap=config.max_gap_sec,
        )
        result["watch_load_ema_1800"] = ema_1800
        result["watch_load_ema_21600"] = ema_21600
        result["eligible_count_since_reset"] = dynamic_count
        result["decision_warmup_complete"] = (
            dynamic_count >= config.decision_warmup_sec
        ).astype("int8")
        eligible_series = pd.Series(eligible, index=result.index, dtype="float64")
        result["watch_ineligible_fraction_60"] = (
            1.0 - eligible_series.shift(1).rolling(60, min_periods=60).mean()
        )
        current_history_1800 = eligible_series.rolling(
            config.decision_warmup_sec, min_periods=config.decision_warmup_sec
        ).sum()
        features_finite = np.isfinite(
            result[list(FEATURE_NAMES)].to_numpy(dtype="float64")
        ).all(axis=1)
        result["feature_decisionable"] = (
            (result["baseline_ready"].to_numpy(dtype="int8") == 1)
            & eligible
            & (dynamic_count >= config.decision_warmup_sec)
            & (
                current_history_1800.to_numpy(dtype="float64")
                >= math.ceil(config.decision_warmup_sec * config.min_valid_fraction)
            )
            & features_finite
        ).astype("int8")
    else:
        for source in SIGNAL_COLUMNS:
            safe_name = source.replace("observed__", "")
            result[f"baseline_center__{safe_name}"] = np.nan
            result[f"baseline_scale__{safe_name}"] = np.nan

    target, future_fraction, future_gap, target_valid = _future_target(
        result["watch_load_raw"].to_numpy(dtype="float64"),
        eligible,
        horizon=config.forecast_horizon_sec,
        minimum_fraction=config.min_valid_fraction,
        max_gap=config.max_gap_sec,
    )
    result["standard_target"] = target
    result["future_valid_fraction"] = future_fraction
    result["future_max_gap_seconds"] = future_gap
    result["target_valid"] = target_valid
    result["standard_valid"] = target_valid
    result["decisionable"] = result["feature_decisionable"]
    return result


def build_observational_frame(
    observed: pd.DataFrame,
    *,
    config: ObservationalStandardConfig | None = None,
) -> pd.DataFrame:
    """Normalize actual Watch rows and create causal features plus separate labels."""

    active = config or ObservationalStandardConfig()
    active.validate()
    normalized = _require_input_columns(observed)
    groups = [
        _resample_one_group(group)
        for _, group in normalized.groupby(["person_key", "session_id"], sort=True)
    ]
    if not groups:
        raise ValueError("observational input is empty")
    processed = [
        _build_one_group(group, config=active)
        for group in groups
    ]
    return pd.concat(processed, ignore_index=True).sort_values(
        ["person_key", "session_id", "corrected_utc"], kind="stable"
    ).reset_index(drop=True)


def ordered_feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    values = frame[list(FEATURE_NAMES)].apply(pd.to_numeric, errors="coerce")
    matrix = values.to_numpy(dtype="float32")
    if not np.isfinite(matrix).all():
        raise ValueError("canonical feature matrix contains non-finite values")
    return cast(np.ndarray, matrix)


def _dump_baseline_onnx(path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper

    feature_index = FEATURE_NAMES.index("watch_load_median_300")
    input_info = helper.make_tensor_value_info("features", TensorProto.FLOAT, ["N", 16])
    output_info = helper.make_tensor_value_info(
        "prediction", TensorProto.FLOAT, ["N", 1]
    )
    index = helper.make_tensor("feature_index", TensorProto.INT64, [1], [feature_index])
    node = helper.make_node(
        "Gather", inputs=["features", "feature_index"], outputs=["prediction"], axis=1
    )
    graph = helper.make_graph(
        [node], "observational_standard_baseline_v2", [input_info], [output_info], [index]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    path.write_bytes(model.SerializeToString())


def _golden_input_frame() -> pd.DataFrame:
    rows = 1_800
    seconds = np.arange(rows, dtype="float64")
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
            "person_key": "golden-person",
            "session_id": "golden-session",
            "corrected_utc": pd.date_range(
                "2026-08-04T00:00:00Z", periods=rows, freq="s", tz="UTC"
            ),
            "observed__eda_us": eda,
            "observed__heart_rate_bpm": hr,
            "observed__motion_magnitude": motion,
            "quality_confidence": np.ones(rows, dtype="float64"),
        }
    )


def _golden_fixture() -> dict[str, object]:
    input_frame = _golden_input_frame()
    built = build_observational_frame(input_frame)
    last = built.iloc[-1]
    vector = np.asarray([last[name] for name in FEATURE_NAMES], dtype="float32")
    if int(last["feature_decisionable"]) != 1 or not np.isfinite(vector).all():
        raise AssertionError("golden fixture did not reach a decisionable canonical vector")
    rows: list[dict[str, object]] = []
    for item in input_frame.to_dict(orient="records"):
        timestamp = cast(pd.Timestamp, item["corrected_utc"])
        rows.append(
            {
                "person_key": str(item["person_key"]),
                "session_id": str(item["session_id"]),
                "corrected_utc": timestamp.isoformat().replace("+00:00", "Z"),
                "observed__eda_us": float(cast(float, item["observed__eda_us"])),
                "observed__heart_rate_bpm": float(
                    cast(float, item["observed__heart_rate_bpm"])
                ),
                "observed__motion_magnitude": float(
                    cast(float, item["observed__motion_magnitude"])
                ),
                "quality_confidence": float(
                    cast(float, item["quality_confidence"])
                ),
            }
        )
    return {
        "schema_version": "watch_standard_16_golden_fixture_v1",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "input_rows": rows,
        "decision_timestamp": rows[-1]["corrected_utc"],
        "ordered_feature_names": list(FEATURE_NAMES),
        "expected_feature_vector": vector.tolist(),
        "expected_prediction": float(vector[FEATURE_NAMES.index("watch_load_median_300")]),
        "expected_feature_decisionable": True,
        "expected_stage": None,
    }


def export_baseline_shadow_bundle(
    output_root: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Export the deterministic rolling-median shadow bundle, not a learned model."""

    root = output_root.resolve()
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FileExistsError(f"baseline shadow bundle already exists: {root}")
    root.mkdir(parents=True, exist_ok=True)
    schema = canonical_feature_schema()
    _write_json(root / "feature_schema.json", schema)
    _dump_baseline_onnx(root / "model.onnx")
    _write_json(root / "golden_fixture.json", _golden_fixture())
    runtime = {
        "python": "3.12",
        "numpy": np.__version__,
        "onnx": importlib.metadata.version("onnx"),
        "onnxruntime": importlib.metadata.version("onnxruntime"),
        "onnx_providers": ["CPUExecutionProvider"],
        "onnx_intra_op_threads": 1,
    }
    _write_json(root / "runtime.json", runtime)
    (root / "runtime_requirements.txt").write_text(
        f"numpy=={runtime['numpy']}\n"
        f"onnxruntime=={runtime['onnxruntime']}\n",
        encoding="utf-8",
    )
    artifacts = {
        path.name: _sha256_file(path)
        for path in sorted(root.iterdir())
        if path.is_file()
    }
    manifest: dict[str, object] = {
        "schema_version": "observational_baseline_shadow_bundle_v1",
        "model_release": MODEL_RELEASE,
        "model_kind": "deterministic_baseline_selector",
        "selected_candidate": "rolling_median_300",
        "input_name": "features",
        "input_dtype": "float32",
        "input_shape": [None, 16],
        "output_name": "prediction",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_hash": FEATURE_SCHEMA_SHA256,
        "baseline_version": BASELINE_VERSION,
        "load_formula_version": LOAD_FORMULA_VERSION,
        "quality_formula_version": QUALITY_FORMULA_VERSION,
        "target_version": TARGET_VERSION,
        "legacy_15_feature_status": LEGACY_15_FEATURE_STATUS,
        "real_data_status": "NOT VERIFIED",
        "stage": None,
        "pattern": None,
        "behavior": None,
        "locked_test_read": False,
        "artifacts": artifacts,
    }
    _write_json(root / "manifest.json", manifest)
    checksums = {
        path.name: _sha256_file(path)
        for path in sorted(root.iterdir())
        if path.is_file()
    }
    _write_json(root / "SHA256SUMS.json", checksums)
    return manifest


def verify_baseline_shadow_bundle(bundle: Path) -> dict[str, object]:
    """Fail closed on hash, schema, ONNX shape, or golden-output mismatch."""

    import onnx
    import onnxruntime as ort  # type: ignore[import-untyped]

    root = bundle.resolve()
    checksums = json.loads((root / "SHA256SUMS.json").read_text(encoding="utf-8"))
    for name, digest in checksums.items():
        if _sha256_file(root / name) != digest:
            raise ValueError(f"bundle checksum mismatch: {name}")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    schema = json.loads((root / "feature_schema.json").read_text(encoding="utf-8"))
    if schema != canonical_feature_schema():
        raise ValueError("feature schema differs from canonical watch_standard_16_v2")
    if hashlib.sha256(_canonical_json(schema)).hexdigest() != FEATURE_SCHEMA_SHA256:
        raise ValueError("feature schema SHA-256 mismatch")
    for name, digest in manifest["artifacts"].items():
        if _sha256_file(root / name) != digest:
            raise ValueError(f"artifact SHA-256 mismatch: {name}")
    model = onnx.load(root / "model.onnx")
    if model.graph.input[0].type.tensor_type.shape.dim[1].dim_value != 16:
        raise ValueError("ONNX input is not float32[N,16]")
    golden = json.loads((root / "golden_fixture.json").read_text(encoding="utf-8"))
    expected_vector = np.asarray(
        golden["expected_feature_vector"], dtype="float32"
    ).reshape(1, 16)
    replayed = build_observational_frame(pd.DataFrame(golden["input_rows"]))
    if int(replayed.iloc[-1]["feature_decisionable"]) != 1:
        raise ValueError("golden replay is not feature-decisionable")
    replayed_vector = ordered_feature_matrix(replayed.iloc[[-1]])
    if not np.allclose(replayed_vector, expected_vector, rtol=1e-6, atol=1e-7):
        raise ValueError("golden feature vector mismatch")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(root / "model.onnx"),
        providers=["CPUExecutionProvider"],
        sess_options=options,
    )
    prediction = float(session.run(None, {"features": replayed_vector})[0][0, 0])
    if not math.isclose(prediction, float(golden["expected_prediction"]), abs_tol=1e-7):
        raise ValueError("golden ONNX prediction mismatch")
    return {
        "status": "VERIFIED",
        "model_release": manifest["model_release"],
        "feature_schema_hash": FEATURE_SCHEMA_SHA256,
    }
