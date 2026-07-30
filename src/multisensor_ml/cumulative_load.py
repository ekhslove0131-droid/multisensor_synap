from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import pandas as pd

SCALAR_EXPOSURE_FACTORS: tuple[str, ...] = (
    "autonomic_arousal",
    "cognitive_load",
    "sensory_context",
    "sleep_pressure",
)
RECOVERY_FACTOR = "recovery_capacity"
INDEPENDENT_FACTORS: tuple[str, ...] = (
    "motor_activation",
    "social_context",
)
REQUIRED_FACTORS = (
    *SCALAR_EXPOSURE_FACTORS,
    RECOVERY_FACTOR,
    *INDEPENDENT_FACTORS,
)


def _horizon_label(horizon_sec: int) -> str:
    labels = {
        1800: "30m",
        21600: "6h",
        86400: "24h",
        259200: "72h",
    }
    return labels.get(horizon_sec, f"{horizon_sec}s")


def _validate_robust_z(robust_z: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_FACTORS).difference(robust_z.columns))
    if missing:
        raise ValueError(f"cumulative load missing robust-z factors: {missing}")
    numeric = robust_z.loc[:, REQUIRED_FACTORS].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype="float64")).all():
        raise ValueError("cumulative load robust-z factors must be finite")
    return cast(pd.DataFrame, numeric.astype("float64"))


def instantaneous_load(robust_z: pd.DataFrame) -> pd.Series:
    """Return label-free net exposure; positive recovery supplies a bounded credit."""

    numeric = _validate_robust_z(robust_z)
    exposure_parts = [
        numeric[factor].clip(lower=0.0) for factor in SCALAR_EXPOSURE_FACTORS
    ]
    exposure_parts.append((-numeric[RECOVERY_FACTOR]).clip(lower=0.0))
    exposure = pd.concat(exposure_parts, axis=1).mean(axis=1)
    recovery_credit = numeric[RECOVERY_FACTOR].clip(lower=0.0) / len(
        exposure_parts
    )
    return (exposure - recovery_credit).rename("instantaneous_load")


def _decayed_state(
    timestamps_ns: np.ndarray,
    values: np.ndarray,
    horizon_sec: int,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.zeros(len(values), dtype="float64")
    mature = np.zeros(len(values), dtype=bool)
    if len(values) == 0:
        return states, mature
    first_ns = int(timestamps_ns[0])
    previous_ns = first_ns
    for index in range(1, len(values)):
        current_ns = int(timestamps_ns[index])
        elapsed_sec = max(0.0, (current_ns - previous_ns) / 1_000_000_000)
        decay = float(np.exp(-np.log(2.0) * elapsed_sec / horizon_sec))
        states[index] = max(
            0.0,
            decay * states[index - 1] + (1.0 - decay) * values[index],
        )
        mature[index] = (current_ns - first_ns) / 1_000_000_000 >= horizon_sec
        previous_ns = current_ns
    return states, mature


def build_cumulative_load(
    frame: pd.DataFrame,
    robust_z: pd.DataFrame,
    *,
    horizons_sec: Sequence[int],
) -> pd.DataFrame:
    """Build causal load states independently within each person and run."""

    required = {"person_key", "run_id", "timestamp_utc"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"cumulative load input missing columns: {missing}")
    if frame.empty or len(frame) != len(robust_z):
        raise ValueError("frame and robust_z must be non-empty and row-aligned")
    horizons = tuple(int(value) for value in horizons_sec)
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("cumulative load horizons must be positive")

    numeric = _validate_robust_z(robust_z).reset_index(drop=True)
    work = frame.loc[:, ["person_key", "run_id", "timestamp_utc"]].copy()
    work["_source_order"] = np.arange(len(work), dtype=np.int64)
    work["timestamp_utc"] = pd.to_datetime(work["timestamp_utc"], utc=True, errors="raise")
    work = work.sort_values(
        ["person_key", "run_id", "timestamp_utc", "_source_order"],
        kind="mergesort",
    ).reset_index(drop=True)
    numeric = numeric.iloc[work["_source_order"].to_numpy(dtype=np.int64)].reset_index(
        drop=True
    )
    scalar_values = instantaneous_load(numeric).to_numpy(dtype="float64")
    factor_values: dict[str, np.ndarray] = {}
    for factor in REQUIRED_FACTORS:
        direction = -numeric[factor] if factor == RECOVERY_FACTOR else numeric[factor]
        factor_values[factor] = direction.clip(lower=0.0).to_numpy(dtype="float64")

    output = work.copy()
    grouped = work.groupby(["person_key", "run_id"], sort=False, observed=True)
    for horizon in horizons:
        label = _horizon_label(horizon)
        scalar_state = np.zeros(len(work), dtype="float64")
        maturity = np.zeros(len(work), dtype=bool)
        factor_states = {
            factor: np.zeros(len(work), dtype="float64") for factor in REQUIRED_FACTORS
        }
        for index in grouped.indices.values():
            positions = np.asarray(index, dtype=np.int64)
            timestamps = (
                work.iloc[positions]["timestamp_utc"]
                .dt.as_unit("ns")
                .astype("int64")
                .to_numpy()
            )
            state, group_maturity = _decayed_state(
                timestamps,
                scalar_values[positions],
                horizon,
            )
            scalar_state[positions] = state
            maturity[positions] = group_maturity
            for factor in REQUIRED_FACTORS:
                factor_state, _ = _decayed_state(
                    timestamps,
                    factor_values[factor][positions],
                    horizon,
                )
                factor_states[factor][positions] = factor_state
        output[f"cumulative_load_{label}"] = scalar_state
        output[f"cumulative_load_{label}_mature"] = maturity
        for factor in REQUIRED_FACTORS:
            output[f"{factor}_load_{label}"] = factor_states[factor]

    if 1800 in horizons:
        grouped_load = output.groupby(
            ["person_key", "run_id"], sort=False, observed=True
        )["cumulative_load_30m"]
        output["load_slope_30m"] = grouped_load.diff().fillna(0.0)

    return (
        output.sort_values("_source_order", kind="mergesort")
        .drop(columns=["_source_order"])
        .reset_index(drop=True)
    )


def fit_train_load_quantiles(
    load_frame: pd.DataFrame,
    *,
    column: str = "cumulative_load_30m",
) -> dict[str, float]:
    if "split_role" not in load_frame or column not in load_frame:
        raise ValueError("load quantiles require split_role and load column")
    train = pd.to_numeric(
        load_frame.loc[load_frame["split_role"].eq("train"), column],
        errors="raise",
    )
    if train.empty:
        raise ValueError("load quantiles require train rows")
    return {
        "q33": float(train.quantile(1 / 3)),
        "q66": float(train.quantile(2 / 3)),
    }


def classify_load_state(
    load_frame: pd.DataFrame,
    train_quantiles: Mapping[str, float],
    *,
    column: str = "cumulative_load_30m",
) -> pd.Series:
    if column not in load_frame:
        raise ValueError(f"load state missing column: {column}")
    if set(train_quantiles) != {"q33", "q66"}:
        raise ValueError("load state requires q33 and q66 train thresholds")
    q33 = float(train_quantiles["q33"])
    q66 = float(train_quantiles["q66"])
    if q33 > q66:
        raise ValueError("load-state quantiles are not ordered")
    values = pd.to_numeric(load_frame[column], errors="raise")
    states = np.select(
        [values <= q33, values <= q66],
        ["LOW", "MEDIUM"],
        default="HIGH",
    )
    return pd.Series(states, index=load_frame.index, name="load_state", dtype="string")
