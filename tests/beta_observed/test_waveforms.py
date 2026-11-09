from __future__ import annotations

import numpy as np

from multisensor_synth.beta_observed.generator import generate_part

from .fixtures import beta_state


def test_native_family_cadence_and_dtype() -> None:
    state = beta_state()
    watch = generate_part(state, "watch", start_ms=0)
    h10 = generate_part(state, "h10", start_ms=0)
    muse = generate_part(state, "muse", start_ms=0)

    assert len(watch.signals["ppg_green"].values) == 1000
    assert len(h10.signals["ecg"].values) == 1300
    assert len(muse.signals["eeg_1"].values) == 2560
    assert all(signal.values.dtype == np.float32 for part in (watch, h10, muse)
               for signal in part.signals.values())
    assert set(muse.signals) == {"eeg_1", "eeg_2", "eeg_3", "eeg_4"}
    assert set(muse.auxiliary) == {"acc_x", "acc_y", "acc_z", "contact"}


def test_same_seed_is_byte_reproducible_and_different_seed_changes_waveform() -> None:
    first = generate_part(beta_state(seed=7), "h10", 20_000)
    again = generate_part(beta_state(seed=7), "h10", 20_000)
    changed = generate_part(beta_state(seed=8), "h10", 20_000)

    assert np.array_equal(first.signals["ecg"].values, again.signals["ecg"].values)
    assert not np.array_equal(first.signals["ecg"].values, changed.signals["ecg"].values)


def test_rr_is_observed_after_beats_and_not_prefilled() -> None:
    part = generate_part(beta_state(), "h10", 0)
    rr = part.signals["rr"]
    assert len(rr.values) >= 1
    assert rr.time_us[0] > part.start_ms * 1000
    assert np.all(rr.available_ms * 1000 >= rr.time_us)
