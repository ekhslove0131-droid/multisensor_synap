import numpy as np
import pandas as pd
import pytest

from multisensor_ml.phase3_baselines import (
    baseline_policy_hash,
    build_causal_baseline,
    fit_train_global_context_baseline,
)

FACTORS = ("autonomic_arousal", "recovery_capacity")


def source_frame() -> pd.DataFrame:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    return pd.DataFrame(
        {
            "person_key": ["P1"] * 8,
            "run_id": ["R1"] * 8,
            "timestamp_utc": pd.date_range(start, periods=8, freq="s"),
            "context": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "autonomic_arousal": [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0],
            "recovery_capacity": [8.0, 80.0, 7.0, 70.0, 6.0, 60.0, 5.0, 50.0],
        }
    )


def global_baseline() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "context": ["A", "B"],
            "autonomic_arousal__global_median": [0.0, 0.0],
            "autonomic_arousal__global_mad": [2.0, 2.0],
            "recovery_capacity__global_median": [9.0, 90.0],
            "recovery_capacity__global_mad": [2.0, 2.0],
        }
    )


def test_personal_context_baseline_uses_only_past_same_context() -> None:
    result = build_causal_baseline(
        source_frame(),
        global_baseline(),
        mode="personal_context",
        weight_cap=0.5,
        warmup_sec=2,
        lookback_sec=4,
        refresh_sec=1,
        factors=FACTORS,
    )

    assert result.loc[0, "personal_weight"] == 0.0
    assert result.loc[2, "personal_weight"] == 0.0
    assert result.loc[4, "personal_weight"] == pytest.approx(0.5)
    assert result.loc[4, "autonomic_arousal__personal_median"] == 1.5
    assert result.loc[5, "autonomic_arousal__personal_median"] == 15.0
    assert result["personal_weight"].between(0.0, 0.5).all()


def test_personal_context_baseline_is_prefix_invariant() -> None:
    frame = source_frame()
    changed = frame.copy()
    changed.loc[7, "autonomic_arousal"] = 1_000_000.0

    original_result = build_causal_baseline(
        frame,
        global_baseline(),
        mode="personal_context",
        weight_cap=0.5,
        warmup_sec=2,
        lookback_sec=4,
        refresh_sec=1,
        factors=FACTORS,
    )
    changed_result = build_causal_baseline(
        changed,
        global_baseline(),
        mode="personal_context",
        weight_cap=0.5,
        warmup_sec=2,
        lookback_sec=4,
        refresh_sec=1,
        factors=FACTORS,
    )

    pd.testing.assert_frame_equal(original_result.iloc[:7], changed_result.iloc[:7])


def test_global_context_mode_uses_exact_train_baseline() -> None:
    result = build_causal_baseline(
        source_frame(),
        global_baseline(),
        mode="global_context",
        weight_cap=0.0,
        warmup_sec=2,
        lookback_sec=4,
        refresh_sec=1,
        factors=FACTORS,
    )

    assert not result["personal_weight"].any()
    assert result.loc[result["context"] == "A", "autonomic_arousal__center"].eq(0.0).all()
    assert result.loc[result["context"] == "B", "recovery_capacity__mad"].eq(2.0).all()


def test_global_context_fit_is_train_only_and_deterministic() -> None:
    frame = source_frame()
    first = fit_train_global_context_baseline(frame, factors=FACTORS)
    changed = frame.copy()
    changed["split_role"] = "validation"

    with pytest.raises(ValueError, match="train"):
        fit_train_global_context_baseline(changed, factors=FACTORS)

    second = fit_train_global_context_baseline(frame, factors=FACTORS)
    pd.testing.assert_frame_equal(first, second)
    assert baseline_policy_hash("personal_context", 0.5) == baseline_policy_hash(
        "personal_context",
        0.5,
    )


def test_unknown_context_fails_closed() -> None:
    frame = source_frame()
    frame.loc[0, "context"] = "UNKNOWN"

    with pytest.raises(ValueError, match="context"):
        build_causal_baseline(
            frame,
            global_baseline(),
            mode="personal_context",
            weight_cap=0.5,
            warmup_sec=2,
            lookback_sec=4,
            refresh_sec=1,
            factors=FACTORS,
        )


def test_current_row_is_excluded_from_personal_center() -> None:
    frame = source_frame()
    result = build_causal_baseline(
        frame,
        global_baseline(),
        mode="personal_pooled",
        weight_cap=1.0,
        warmup_sec=2,
        lookback_sec=4,
        refresh_sec=1,
        factors=FACTORS,
    )

    assert np.isclose(result.loc[2, "autonomic_arousal__personal_median"], 5.5)
