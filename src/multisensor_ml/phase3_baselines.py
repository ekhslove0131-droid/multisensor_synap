from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from multisensor_ml.baseline import MAD_FLOOR
from multisensor_ml.contracts import ORACLE_LATENT_FACTORS

BaselineMode = Literal["global_context", "personal_pooled", "personal_context"]


def fit_train_global_context_baseline(
    frame: pd.DataFrame,
    factors: Sequence[str] = ORACLE_LATENT_FACTORS,
) -> pd.DataFrame:
    """Fit deterministic context baselines from an already train-filtered frame."""

    if "split_role" in frame and not frame["split_role"].eq("train").all():
        raise ValueError("global baseline input must contain train rows only")
    required = {"context", *factors}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"global baseline missing columns: {missing}")
    if frame.empty:
        raise ValueError("global baseline requires train rows")
    rows: list[dict[str, object]] = []
    for context, group in frame.groupby("context", observed=True, sort=True):
        row: dict[str, object] = {"context": str(context)}
        for factor in factors:
            values = pd.to_numeric(group[factor], errors="raise")
            median = float(values.median())
            mad = float((values - median).abs().median())
            row[f"{factor}__global_median"] = median
            row[f"{factor}__global_mad"] = max(mad, MAD_FLOOR)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("context", kind="mergesort").reset_index(drop=True)


def baseline_policy_hash(mode: BaselineMode, weight_cap: float) -> str:
    payload = json.dumps(
        {"mode": mode, "weight_cap": float(weight_cap)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _history_groups(work: pd.DataFrame, mode: BaselineMode) -> list[str]:
    keys = ["person_key", "run_id"]
    if mode == "personal_context":
        keys.append("context")
    return keys


def _refresh_mask(
    work: pd.DataFrame,
    *,
    refresh_sec: int,
) -> pd.Series:
    start = work.groupby(["person_key", "run_id"], sort=False, observed=True)[
        "timestamp_utc"
    ].transform("min")
    elapsed = (work["timestamp_utc"] - start).dt.total_seconds().astype("int64")
    return elapsed.mod(refresh_sec).eq(0)


def build_causal_baseline(
    frame: pd.DataFrame,
    global_baseline: pd.DataFrame,
    *,
    mode: BaselineMode,
    weight_cap: float,
    warmup_sec: int = 1800,
    lookback_sec: int = 21600,
    refresh_sec: int = 60,
    factors: Sequence[str] = ORACLE_LATENT_FACTORS,
) -> pd.DataFrame:
    """Build a trailing-only global, pooled-personal, or context-personal baseline."""

    if mode not in {"global_context", "personal_pooled", "personal_context"}:
        raise ValueError(f"unknown baseline mode: {mode}")
    if not 0 <= weight_cap <= 1:
        raise ValueError("weight_cap must be in [0, 1]")
    if warmup_sec <= 0 or lookback_sec < warmup_sec or refresh_sec <= 0:
        raise ValueError("invalid baseline timing contract")
    required = {"person_key", "run_id", "timestamp_utc", "context", *factors}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"baseline input missing columns: {missing}")
    if frame.empty:
        raise ValueError("baseline input is empty")

    work = frame.copy()
    work["_source_order"] = np.arange(len(work), dtype=np.int64)
    work["timestamp_utc"] = pd.to_datetime(work["timestamp_utc"], utc=True, errors="raise")
    work = work.sort_values(
        ["person_key", "run_id", "timestamp_utc", "_source_order"],
        kind="mergesort",
    ).reset_index(drop=True)
    context_lookup = global_baseline.set_index("context")
    missing_contexts = sorted(set(work["context"]) - set(context_lookup.index))
    if missing_contexts:
        raise ValueError(f"global baseline lacks context: {missing_contexts}")

    groups = _history_groups(work, mode)
    grouped = work.groupby(groups, sort=False, observed=True)
    n_eff = grouped.cumcount().clip(upper=lookback_sec).astype("int64")
    if mode == "global_context":
        weight = pd.Series(0.0, index=work.index, dtype="float64")
    else:
        raw_weight = np.where(
            n_eff >= warmup_sec,
            n_eff / (n_eff + warmup_sec),
            0.0,
        )
        weight = pd.Series(np.minimum(raw_weight, weight_cap), index=work.index)
    refresh_mask = _refresh_mask(work, refresh_sec=refresh_sec)
    output = pd.DataFrame(
        {
            "person_key": work["person_key"],
            "run_id": work["run_id"],
            "timestamp_utc": work["timestamp_utc"],
            "context": work["context"],
            "n_eff": n_eff,
            "personal_weight": weight,
            "is_refresh": refresh_mask,
        }
    )

    for factor in factors:
        global_median = work["context"].map(
            context_lookup[f"{factor}__global_median"]
        ).astype("float64")
        global_mad = work["context"].map(
            context_lookup[f"{factor}__global_mad"]
        ).astype("float64")
        if mode == "global_context":
            personal_median = global_median.copy()
            personal_mad = global_mad.copy()
        else:
            values = pd.to_numeric(work[factor], errors="raise").astype("float64")
            medians = pd.Series(np.nan, index=work.index, dtype="float64")
            mads = pd.Series(np.nan, index=work.index, dtype="float64")
            for _, index in grouped.indices.items():
                positions = pd.Index(index, dtype="int64")
                group_values = values.iloc[positions]
                prior = group_values.shift(1)
                median = prior.rolling(
                    window=lookback_sec,
                    min_periods=warmup_sec,
                ).median()
                deviation = (prior - median).abs()
                mad = deviation.rolling(
                    window=lookback_sec,
                    min_periods=1,
                ).median()
                local_refresh = refresh_mask.iloc[positions].to_numpy(dtype=bool)
                medians.iloc[positions] = median.where(local_refresh).ffill().array
                mads.iloc[positions] = mad.where(local_refresh).ffill().array
            personal_median = medians.fillna(global_median)
            personal_mad = mads.fillna(global_mad).clip(lower=MAD_FLOOR)
        output[f"{factor}__personal_median"] = personal_median
        output[f"{factor}__personal_mad"] = personal_mad
        output[f"{factor}__center"] = (
            (1.0 - weight) * global_median + weight * personal_median
        )
        output[f"{factor}__mad"] = (
            (1.0 - weight) * global_mad + weight * personal_mad
        ).clip(lower=MAD_FLOOR)

    output["_source_order"] = work["_source_order"]
    return (
        output.sort_values("_source_order", kind="mergesort")
        .drop(columns="_source_order")
        .reset_index(drop=True)
    )
