from __future__ import annotations

import numpy as np
import pandas as pd

from multisensor_ml.contracts import ORACLE_LATENT_FACTORS, assert_oracle_columns

DEFAULT_WINDOWS: tuple[int, ...] = (5, 15, 30, 60, 180, 300)
CONTEXT_CATEGORIES: tuple[str, ...] = (
    "sleep",
    "transition",
    "meal_context",
    "focused_task",
    "moderate_activity",
    "light_activity",
    "wake_rest",
    "sedentary_activity",
)
ROBUST_SCALE = 1.4826


def build_causal_features(
    frame: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    factors: tuple[str, ...] = ORACLE_LATENT_FACTORS,
) -> tuple[pd.DataFrame, list[str]]:
    """Build current and trailing-only robust-z features."""

    assert_oracle_columns(list(frame.columns))
    if len(frame) != len(baseline):
        raise ValueError("frame and baseline must have equal length")
    columns: dict[str, object] = {}
    names: list[str] = []
    for factor in factors:
        z_name = f"{factor}__robust_z"
        robust_z = (
            (frame[factor].astype("float64") - baseline[f"{factor}__center"])
            / (ROBUST_SCALE * baseline[f"{factor}__mad"])
        ).clip(-20, 20)
        columns[z_name] = robust_z.astype("float32").to_numpy()
        names.append(z_name)
        for window in windows:
            rolling = robust_z.rolling(window=window, min_periods=1)
            mean_name = f"{factor}__mean_{window}s"
            std_name = f"{factor}__std_{window}s"
            slope_name = f"{factor}__slope_{window}s"
            columns[mean_name] = rolling.mean().astype("float32").to_numpy()
            columns[std_name] = (
                rolling.std(ddof=0).fillna(0).astype("float32").to_numpy()
            )
            denominator = max(window - 1, 1)
            columns[slope_name] = (
                (robust_z - robust_z.shift(window - 1).fillna(robust_z.iloc[0]))
                / denominator
            ).astype("float32").to_numpy()
            names.extend((mean_name, std_name, slope_name))

    timestamp = pd.to_datetime(frame["timestamp_utc"], utc=True)
    seconds_of_day = timestamp.dt.hour * 3600 + timestamp.dt.minute * 60 + timestamp.dt.second
    radians = 2 * np.pi * seconds_of_day / 86_400
    for name, values in (
        ("time_sin", np.sin(radians)),
        ("time_cos", np.cos(radians)),
        ("weekday_sin", np.sin(2 * np.pi * timestamp.dt.dayofweek / 7)),
        ("weekday_cos", np.cos(2 * np.pi * timestamp.dt.dayofweek / 7)),
    ):
        columns[name] = np.asarray(values, dtype=np.float32)
        names.append(name)
    columns["is_awake"] = frame["is_awake"].astype("int8").to_numpy()
    names.append("is_awake")
    for context in CONTEXT_CATEGORIES:
        name = f"context__{context}"
        columns[name] = (frame["context"] == context).astype("int8").to_numpy()
        names.append(name)
    return pd.DataFrame(columns, index=frame.index), names
