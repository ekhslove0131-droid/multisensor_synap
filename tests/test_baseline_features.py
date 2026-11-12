from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

from multisensor_ml.baseline import fit_global_baseline, personalize_baseline
from multisensor_ml.contracts import ORACLE_LATENT_FACTORS
from multisensor_ml.features import build_causal_features


def _frame(rows: int = 1900) -> pd.DataFrame:
    data: dict[str, object] = {
        "timestamp_utc": pd.date_range("2026-01-01", periods=rows, freq="s", tz="UTC"),
        "context": ["ordinary_activity"] * rows,
        "is_awake": [True] * rows,
    }
    for index, factor in enumerate(ORACLE_LATENT_FACTORS):
        data[factor] = np.linspace(index, index + 1, rows, dtype=np.float32)
    return pd.DataFrame(data)


def test_personal_baseline_is_causal_warm_and_bounded() -> None:
    frame = _frame()
    global_baseline = fit_global_baseline(frame.iloc[:1800])

    baseline = personalize_baseline(
        frame,
        global_baseline,
        warmup_sec=1800,
        lookback_sec=21600,
        refresh_sec=60,
        quality_confidence=1.0,
    )

    assert baseline.loc[:1799, "personal_weight"].eq(0).all()
    assert baseline.loc[1800, "personal_weight"] == 0.5
    assert baseline["personal_weight"].between(0, 1).all()

    changed = frame.copy()
    changed.loc[1850:, ORACLE_LATENT_FACTORS[0]] = 10_000
    changed_baseline = personalize_baseline(changed, global_baseline)
    pd.testing.assert_frame_equal(
        baseline.loc[:1849],
        changed_baseline.loc[:1849],
        check_exact=True,
    )


@given(st.floats(min_value=0, max_value=1, allow_nan=False))
def test_quality_confidence_keeps_personal_weight_in_unit_interval(quality: float) -> None:
    frame = _frame(1810)
    global_baseline = fit_global_baseline(frame.iloc[:1800])

    baseline = personalize_baseline(frame, global_baseline, quality_confidence=quality)

    assert baseline["personal_weight"].between(0, 1).all()


def test_causal_feature_windows_never_observe_future_rows() -> None:
    frame = _frame(1900)
    global_baseline = fit_global_baseline(frame.iloc[:1800])
    baseline = personalize_baseline(frame, global_baseline)

    first, names = build_causal_features(frame, baseline, windows=(5, 60))
    changed = frame.copy()
    changed.loc[1850:, ORACLE_LATENT_FACTORS[0]] = 99_999
    second, _ = build_causal_features(changed, baseline, windows=(5, 60))

    pd.testing.assert_frame_equal(first.loc[:1849], second.loc[:1849])
    assert f"{ORACLE_LATENT_FACTORS[0]}__mean_5s" in names
    assert f"{ORACLE_LATENT_FACTORS[0]}__std_60s" in names
    assert f"{ORACLE_LATENT_FACTORS[0]}__slope_60s" in names


def test_personal_baseline_never_exceeds_selected_adaptation_cap() -> None:
    frame = _frame(1900)
    global_baseline = fit_global_baseline(frame.iloc[:1800])

    baseline = personalize_baseline(frame, global_baseline, weight_cap=0.20)

    assert baseline["personal_weight"].max() == 0.20
