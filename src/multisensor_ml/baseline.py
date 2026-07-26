from __future__ import annotations

import numpy as np
import pandas as pd

from multisensor_ml.contracts import ORACLE_LATENT_FACTORS

MAD_FLOOR = 1e-4


def fit_global_baseline(
    frame: pd.DataFrame,
    factors: tuple[str, ...] = ORACLE_LATENT_FACTORS,
) -> pd.DataFrame:
    """Fit context-specific robust train-person baselines."""

    rows: list[dict[str, object]] = []
    for context, group in frame.groupby("context", observed=True):
        row: dict[str, object] = {"context": str(context)}
        for factor in factors:
            median = float(group[factor].median())
            mad = float((group[factor] - median).abs().median())
            row[f"{factor}__global_median"] = median
            row[f"{factor}__global_mad"] = max(mad, MAD_FLOOR)
        rows.append(row)
    if not rows:
        raise ValueError("global baseline requires at least one train row")
    return pd.DataFrame(rows).sort_values("context").reset_index(drop=True)


def personalize_baseline(
    frame: pd.DataFrame,
    global_baseline: pd.DataFrame,
    *,
    warmup_sec: int = 1800,
    lookback_sec: int = 21600,
    refresh_sec: int = 60,
    quality_confidence: float = 1.0,
    factors: tuple[str, ...] = ORACLE_LATENT_FACTORS,
) -> pd.DataFrame:
    """Blend causal personal robust baselines with context-specific global baselines."""

    if not 0 <= quality_confidence <= 1:
        raise ValueError("quality_confidence must be in [0, 1]")
    if warmup_sec <= 0 or lookback_sec < warmup_sec or refresh_sec <= 0:
        raise ValueError("invalid warmup/lookback/refresh contract")
    if not frame["timestamp_utc"].is_monotonic_increasing:
        raise ValueError("personal baseline input must be time sorted")

    context_lookup = global_baseline.set_index("context")
    missing_contexts = sorted(set(frame["context"]) - set(context_lookup.index))
    if missing_contexts:
        raise ValueError(f"global baseline lacks contexts: {missing_contexts}")

    count = len(frame)
    index = np.arange(count, dtype=np.int64)
    n_eff = np.minimum(index, lookback_sec)
    weight = np.where(
        index >= warmup_sec,
        n_eff / (n_eff + warmup_sec) * quality_confidence,
        0.0,
    )
    refresh_mask = index % refresh_sec == 0
    output = pd.DataFrame(
        {
            "timestamp_utc": frame["timestamp_utc"].to_numpy(),
            "context": frame["context"].to_numpy(),
            "n_eff": n_eff,
            "quality_confidence": np.full(count, quality_confidence),
            "personal_weight": weight,
            "is_refresh": refresh_mask,
        },
        index=frame.index,
    )

    for factor in factors:
        values = frame[factor].astype("float64")
        prior = values.shift(1)
        personal_median = prior.rolling(
            window=lookback_sec,
            min_periods=warmup_sec,
        ).median()
        absolute_deviation = (prior - personal_median).abs()
        personal_mad = absolute_deviation.rolling(
            window=lookback_sec,
            min_periods=1,
        ).median()
        personal_median = personal_median.where(refresh_mask).ffill()
        personal_mad = personal_mad.where(refresh_mask).ffill()

        global_median = frame["context"].map(
            context_lookup[f"{factor}__global_median"]
        ).astype("float64")
        global_mad = frame["context"].map(
            context_lookup[f"{factor}__global_mad"]
        ).astype("float64")
        personal_median = personal_median.fillna(global_median)
        personal_mad = personal_mad.fillna(global_mad).clip(lower=MAD_FLOOR)
        output[f"{factor}__personal_median"] = personal_median
        output[f"{factor}__personal_mad"] = personal_mad
        output[f"{factor}__center"] = (
            (1 - weight) * global_median + weight * personal_median
        )
        output[f"{factor}__mad"] = (
            (1 - weight) * global_mad + weight * personal_mad
        ).clip(lower=MAD_FLOOR)
    return output
