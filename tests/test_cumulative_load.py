import numpy as np
import pandas as pd
import pytest

from multisensor_ml.cumulative_load import (
    build_cumulative_load,
    classify_load_state,
    fit_train_load_quantiles,
    instantaneous_load,
)

FACTORS = (
    "autonomic_arousal",
    "motor_activation",
    "cognitive_load",
    "sleep_pressure",
    "sensory_context",
    "recovery_capacity",
    "social_context",
)


def frame(periods: int = 8, *, seconds: list[int] | None = None) -> pd.DataFrame:
    offsets = seconds if seconds is not None else list(range(periods))
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    return pd.DataFrame(
        {
            "person_key": ["P1"] * len(offsets),
            "run_id": ["R1"] * len(offsets),
            "timestamp_utc": [start + pd.Timedelta(seconds=value) for value in offsets],
        }
    )


def z_frame(periods: int = 8, value: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame({factor: [value] * periods for factor in FACTORS})


def test_future_mutation_cannot_change_past_load() -> None:
    source = frame()
    robust_z = z_frame()
    changed = robust_z.copy()
    changed.loc[7, "autonomic_arousal"] = 1_000_000.0

    original = build_cumulative_load(source, robust_z, horizons_sec=(4,))
    mutated = build_cumulative_load(source, changed, horizons_sec=(4,))

    pd.testing.assert_frame_equal(original.iloc[:7], mutated.iloc[:7])


def test_constant_exposure_is_monotone_and_bounded() -> None:
    result = build_cumulative_load(frame(), z_frame(), horizons_sec=(4,))
    values = result["cumulative_load_4s"]

    assert values.is_monotonic_increasing
    assert values.between(0.0, 1.0).all()
    assert values.iloc[-1] < 1.0


def test_sustained_recovery_decreases_load() -> None:
    robust_z = z_frame()
    robust_z.loc[4:, :] = 0.0
    robust_z.loc[4:, "recovery_capacity"] = 4.0

    result = build_cumulative_load(frame(), robust_z, horizons_sec=(2,))

    assert result.loc[7, "cumulative_load_2s"] < result.loc[3, "cumulative_load_2s"]
    assert result["cumulative_load_2s"].ge(0.0).all()


def test_person_or_run_change_resets_state() -> None:
    source = frame()
    source.loc[4:, "person_key"] = "P2"
    result = build_cumulative_load(source, z_frame(), horizons_sec=(4,))

    assert result.loc[4, "cumulative_load_4s"] == pytest.approx(0.0)


def test_elapsed_gap_uses_exponential_decay() -> None:
    source = frame(periods=3, seconds=[0, 1, 5])
    robust_z = z_frame(periods=3, value=0.0)
    robust_z.loc[1, "autonomic_arousal"] = 5.0

    result = build_cumulative_load(source, robust_z, horizons_sec=(4,))

    expected = result.loc[1, "cumulative_load_4s"] * 0.5
    assert result.loc[2, "cumulative_load_4s"] == pytest.approx(expected)


def test_labels_and_phase_do_not_affect_results() -> None:
    source = frame()
    source["event_binary"] = [0, 1] * 4
    source["phase"] = ["peak"] * 8
    changed = source.copy()
    changed["event_binary"] = 1 - changed["event_binary"]
    changed["phase"] = "NO_EVENT"

    original = build_cumulative_load(source, z_frame(), horizons_sec=(4,))
    mutated = build_cumulative_load(changed, z_frame(), horizons_sec=(4,))

    pd.testing.assert_frame_equal(original, mutated)


def test_motor_and_social_are_not_in_scalar_but_have_independent_state() -> None:
    base = z_frame(value=0.0)
    changed = base.copy()
    changed["motor_activation"] = 10.0
    changed["social_context"] = 10.0

    original = build_cumulative_load(frame(), base, horizons_sec=(4,))
    mutated = build_cumulative_load(frame(), changed, horizons_sec=(4,))

    pd.testing.assert_series_equal(
        original["cumulative_load_4s"],
        mutated["cumulative_load_4s"],
    )
    assert mutated["motor_activation_load_4s"].iloc[-1] > 0.0
    assert mutated["social_context_load_4s"].iloc[-1] > 0.0


def test_72h_maturity_requires_259200_valid_seconds() -> None:
    source = frame(periods=3, seconds=[0, 259_199, 259_200])
    result = build_cumulative_load(source, z_frame(periods=3), horizons_sec=(259_200,))

    assert not bool(result.loc[1, "cumulative_load_72h_mature"])
    assert bool(result.loc[2, "cumulative_load_72h_mature"])


def test_load_state_quantiles_are_fit_on_train_only() -> None:
    load = pd.DataFrame(
        {
            "split_role": ["train"] * 4 + ["validation"] * 2,
            "cumulative_load_30m": [0.0, 1.0, 2.0, 3.0, 100.0, 200.0],
        }
    )
    quantiles = fit_train_load_quantiles(load)
    states = classify_load_state(
        pd.DataFrame({"cumulative_load_30m": [0.0, 1.5, 3.0]}),
        quantiles,
    )

    assert quantiles["q33"] < 2.0
    assert quantiles["q66"] < 3.0
    assert states.tolist() == ["LOW", "MEDIUM", "HIGH"]


def test_instantaneous_load_uses_expected_direction() -> None:
    robust_z = z_frame(periods=2, value=0.0)
    robust_z.loc[0, "autonomic_arousal"] = 5.0
    robust_z.loc[1, "recovery_capacity"] = -5.0

    load = instantaneous_load(robust_z)

    assert np.isclose(load.iloc[0], 1.0)
    assert np.isclose(load.iloc[1], 1.0)
