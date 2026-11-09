from __future__ import annotations

import numpy as np
import pytest

from multisensor_synth.beta_observed import cardiac
from multisensor_synth.beta_observed.cardiac import beat_phase, heart_rate
from multisensor_synth.beta_observed.generator import generate_part

from .fixtures import beta_state


def test_consecutive_parts_preserve_sequence_and_time_order() -> None:
    state = beta_state()
    first = generate_part(state, "h10", 0)
    second = generate_part(state, "h10", 10_000)

    assert first.signals["ecg"].sequence[-1] + 1 == second.signals["ecg"].sequence[0]
    assert first.signals["ecg"].time_us[-1] < second.signals["ecg"].time_us[0]
    assert not np.array_equal(first.signals["ecg"].values, second.signals["ecg"].values)


def test_ppg_and_ecg_share_causal_heart_modulation_without_equal_waveforms() -> None:
    state = beta_state()
    watch = generate_part(state, "watch", 20_000)
    h10 = generate_part(state, "h10", 20_000)
    watch_hr = watch.signals["hr"].values[watch.signals["hr"].valid]
    h10_hr = h10.signals["hr"].values[h10.signals["hr"].valid]
    assert abs(float(np.median(watch_hr)) - float(np.median(h10_hr))) < 8
    assert watch.signals["ppg_green"].unit == "a.u."
    assert h10.signals["ecg"].unit == "uV"


def test_observed_peak_intervals_follow_ramp_and_recovery_across_chunks() -> None:
    state = beta_state()
    active = generate_part(state, "h10", 20_000)
    recovery = generate_part(state, "h10", 50_000)
    active_hr = active.signals["hr"].values[active.signals["hr"].valid]
    recovery_hr = recovery.signals["hr"].values[recovery.signals["hr"].valid]
    assert len(active.signals["rr"].values) >= 8
    assert len(recovery.signals["rr"].values) >= 8
    assert float(np.median(active_hr)) > float(np.median(recovery_hr))
    assert active.signals["rr"].time_us[-1] < recovery.signals["rr"].time_us[0]


def test_integrated_phase_derivative_tracks_time_varying_heart_rate() -> None:
    state = beta_state()
    time = np.linspace(0, 80, 8001)
    instantaneous = np.diff(beat_phase(state, time)) * 60 / np.diff(time)
    expected = heart_rate(state, (time[:-1] + time[1:]) / 2)
    assert np.max(np.abs(instantaneous - expected)) < 0.1


def test_ppg_uses_fixed_200ms_delay_not_point_two_cycles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ZeroNoise:
        @staticmethod
        def normal(_mean: float, _sd: float, length: int) -> np.ndarray:
            return np.zeros(length, dtype=np.float64)

    monkeypatch.setattr(cardiac, "beat_phase", lambda _state, time: 2.0 * time)
    monkeypatch.setattr(cardiac, "_rng", lambda *_args: ZeroNoise())
    observed = cardiac.ppg(beta_state(), 1_000, fs=10)
    time = 1.0 + np.arange(100, dtype=np.float64) / 10
    delayed_phase = np.mod(2.0 * (time - 0.2), 1.0)
    expected = 0.2 + np.exp(
        -0.5 * (np.minimum(delayed_phase, 1 - delayed_phase) / 0.08) ** 2
    )
    assert observed == pytest.approx(expected.astype(np.float32), abs=1e-7)
