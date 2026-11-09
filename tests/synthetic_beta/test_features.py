from __future__ import annotations

import pytest

pytest.importorskip("neurokit2", reason="generator dependency is owned by beta-observed extra")

from multisensor_synth.beta_observed.config import BetaSignalState
from multisensor_synth.beta_observed.generator import generate_part
from synapbridge_beta_runtime.features import derive_features
from synapbridge_beta_runtime.types import NativePart, SignalSamples


def _parts(family: str, seconds: int = 180):
    state = BetaSignalState.fixture()
    return [generate_part(state, family, start) for start in range(0, seconds * 1000, 10_000)]


def test_h10_features_are_recomputed_from_past_native_samples() -> None:
    frames = derive_features(_parts("h10"), 180_000)
    frame = frames[0]
    assert frame.family == "h10"
    assert frame.decision_ms == 180_000
    assert frame.values["hr_median_180"] > 0
    assert frame.values["rr_rmssd_180"] >= 0
    assert frame.values["rr_sdnn_180"] >= 0
    assert frame.valid_support_ms == (170_000, 180_000)


def test_future_native_part_does_not_change_current_features() -> None:
    history = _parts("watch")
    before = derive_features(history, 180_000)
    future = generate_part(BetaSignalState.fixture(seed=999), "watch", 180_000)
    after = derive_features([*history, future], 180_000)
    assert before == after


def test_samples_not_available_at_decision_are_excluded() -> None:
    parts = _parts("watch", seconds=190)
    last = parts[-1]
    eda = last.signals["eda"]
    values = eda.values.copy()
    values[:] = 999.0
    delayed = SignalSamples(
        values,
        eda.time_us,
        eda.available_ms + 20_000,
        eda.sequence,
        eda.valid,
        eda.unit,
        eda.nominal_fs_hz,
    )
    signals = dict(last.signals)
    signals["eda"] = delayed
    parts[-1] = NativePart(
        last.scope,
        last.source_uuid,
        last.family,
        NativePart.freeze_signals(signals),
        last.auxiliary,
        last.start_ms,
        last.end_ms,
    )
    assert derive_features(parts, 190_000) == ()


def test_feature_history_cannot_mix_two_source_uuids() -> None:
    parts = _parts("watch")
    last = parts[-1]
    parts[-1] = NativePart(
        last.scope,
        "30000000-0000-4000-8000-000000000099",
        last.family,
        last.signals,
        last.auxiliary,
        last.start_ms,
        last.end_ms,
    )
    with pytest.raises(ValueError, match="feature_scope_mismatch"):
        derive_features(parts, 180_000)


def test_partial_but_eligible_stride_preserves_quality_and_feature_support() -> None:
    parts = _parts("watch", seconds=190)
    last = parts[-1]
    eda = last.signals["eda"]
    valid = eda.valid.copy()
    valid[-1] = False
    signals = dict(last.signals)
    signals["eda"] = SignalSamples(
        eda.values, eda.time_us, eda.available_ms, eda.sequence, valid,
        eda.unit, eda.nominal_fs_hz,
    )
    parts[-1] = NativePart(
        last.scope, last.source_uuid, last.family, NativePart.freeze_signals(signals),
        last.auxiliary, last.start_ms, last.end_ms,
    )
    frame = derive_features(parts, 190_000)[0]
    assert frame.quality == pytest.approx(0.9)
    assert frame.valid_support_ms == (180_000, 190_000)


def test_missing_motion_keeps_biological_m0_features_but_omits_m2_context() -> None:
    stripped = []
    for part in _parts("watch"):
        signals = {name: sample for name, sample in part.signals.items()
                   if not name.startswith("acc_")}
        stripped.append(NativePart(
            part.scope, part.source_uuid, part.family,
            NativePart.freeze_signals(signals), part.auxiliary,
            part.start_ms, part.end_ms,
        ))
    frame = derive_features(stripped, 180_000)[0]
    assert "hr_median_180" in frame.values
    assert "motion_var_180" not in frame.values


def test_watch_model_features_ignore_vendor_hr_and_ibi_values() -> None:
    original = _parts("watch")
    changed = []
    for part in original:
        signals = dict(part.signals)
        for name, replacement in (("hr", 179.0), ("ibi", 333.0)):
            sample = signals[name]
            values = sample.values.copy()
            values[:] = replacement
            signals[name] = SignalSamples(
                values, sample.time_us, sample.available_ms, sample.sequence,
                sample.valid, sample.unit, sample.nominal_fs_hz,
            )
        changed.append(NativePart(
            part.scope, part.source_uuid, part.family,
            NativePart.freeze_signals(signals), part.auxiliary,
            part.start_ms, part.end_ms,
        ))
    assert derive_features(original, 180_000) == derive_features(changed, 180_000)


def test_muse_features_are_computed_from_waveform_not_teacher_power() -> None:
    frame = derive_features(_parts("muse"), 180_000)[0]
    for channel in range(1, 5):
        total = sum(frame.values[f"eeg_{channel}_{band}_rel_180"]
                    for band in ("theta", "alpha", "beta"))
        assert 0 < total < 1
        assert frame.values[f"eeg_{channel}_log_power_180"] > 0

