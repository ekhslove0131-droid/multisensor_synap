"""Streaming observed waveform generation and immutable local manifests."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from synapbridge_beta_runtime.types import BetaScope, NativePart, SignalSamples

from .cardiac import ecg, observed_intervals, ppg
from .config import FAMILIES, BetaSignalState
from .eda import eda_values, temperature_values
from .eeg import eeg_channel

_WAVE_CACHE: OrderedDict[tuple[str, str, str], tuple[int, np.ndarray]] = OrderedDict()


def generate_part(state: BetaSignalState, family: str, start_ms: int) -> NativePart:
    if family not in FAMILIES or start_ms < state.timeline_origin_ms or start_ms % 10_000:
        raise ValueError("beta_part_request_invalid")
    auxiliary: dict[str, SignalSamples]
    if family == "watch":
        signals, auxiliary = _watch(state, start_ms), {}
    elif family == "h10":
        signals, auxiliary = _h10(state, start_ms), {}
    else:
        signals, auxiliary = _muse(state, start_ms)
    return NativePart(
        BetaScope(state.run_uuid, state.epoch, state.subject_uuid),
        state.source_uuids[family], family, NativePart.freeze_signals(signals),
        NativePart.freeze_auxiliary(auxiliary), start_ms, start_ms + 10_000,
    )


def _watch(state: BetaSignalState, start: int) -> dict[str, SignalSamples]:
    base_ppg = ppg(state, start)
    values = {"ppg_green": base_ppg, "ppg_ir": base_ppg * 0.85,
              "ppg_red": base_ppg * 0.72, "eda": eda_values(state, start),
              "skin_temp": temperature_values(state, start)}
    values.update(_motion(state, start))
    signals = {name: _regular(state, name, value, start, _fs(name), _unit("watch", name))
               for name, value in values.items()}
    hr, intervals = _observed_cardiac(state, start, base_ppg, 100, "ppg")
    signals["hr"], signals["ibi"] = hr, intervals
    return signals


def _h10(state: BetaSignalState, start: int) -> dict[str, SignalSamples]:
    base_ecg = ecg(state, start)
    values = {"ecg": base_ecg, **_motion(state, start)}
    signals = {name: _regular(state, name, value, start, _fs(name), _unit("h10", name))
               for name, value in values.items()}
    hr, intervals = _observed_cardiac(state, start, base_ecg, 130, "ecg")
    signals["hr"], signals["rr"] = hr, intervals
    return signals


def _muse(state: BetaSignalState, start: int) -> tuple[dict[str, SignalSamples],
                                                       dict[str, SignalSamples]]:
    signals = {f"eeg_{index}": _regular(state, f"eeg_{index}",
               eeg_channel(state, start, index), start, 256, "uV") for index in range(1, 5)}
    auxiliary = {name: _regular(state, name, value, start, 25, "g")
                 for name, value in _motion(state, start).items()}
    contact = np.full(10, 0.0 if _dropout(state, start) else 1.0, dtype=np.float32)
    auxiliary["contact"] = _regular(state, "contact", contact, start, 1, "binary")
    return signals, auxiliary


def _regular(state: BetaSignalState, name: str, values: np.ndarray, start: int,
             fs: int, unit: str) -> SignalSamples:
    count = len(values)
    time = start * 1000 + np.floor(np.arange(count) * 1_000_000 / fs).astype(np.int64)
    sequence = ((start - state.timeline_origin_ms) * fs // 1000
                + np.arange(count)).astype(np.int64)
    valid = np.full(count, not _dropout(state, start), dtype=np.bool_)
    return _samples(values, time, sequence, valid, unit, float(fs), delay_ms=50)


def _observed_cardiac(
    state: BetaSignalState, start: int, current: np.ndarray, fs: int, kind: str,
) -> tuple[SignalSamples, SignalSamples]:
    previous_start = start - 10_000
    chunks = [current]
    starts = [start]
    if previous_start >= state.timeline_origin_ms:
        key = (state.run_uuid, state.subject_uuid, kind)
        cached = _WAVE_CACHE.get(key)
        previous = (cached[1] if cached is not None and cached[0] == previous_start
                    else ecg(state, previous_start) if kind == "ecg"
                    else ppg(state, previous_start))
        chunks.insert(0, previous)
        starts.insert(0, previous_start)
    key = (state.run_uuid, state.subject_uuid, kind)
    _WAVE_CACHE[key] = (start, current.copy())
    _WAVE_CACHE.move_to_end(key)
    while len(_WAVE_CACHE) > 8:
        _WAVE_CACHE.popitem(last=False)
    values = np.concatenate(chunks)
    times = np.concatenate([
        value * 1000 + np.floor(np.arange(len(chunk)) * 1_000_000 / fs).astype(np.int64)
        for value, chunk in zip(starts, chunks, strict=True)
    ])
    intervals, beat_times = observed_intervals(values, times, float(fs), kind)
    in_part = (beat_times >= start * 1000) & (beat_times < (start + 10_000) * 1000)
    intervals, beat_times = intervals[in_part], beat_times[in_part]
    available = (beat_times + 999) // 1000 + 80
    part_valid = not _dropout(state, start)
    if previous_start >= state.timeline_origin_ms:
        part_valid = part_valid and not _dropout(state, previous_start)
    interval_signal = _samples_with_available(
        intervals, beat_times, available, beat_times.copy(),
        np.full(len(intervals), part_valid, dtype=np.bool_), "ms", None,
    )
    boundaries = start * 1000 + np.arange(1, 11, dtype=np.int64) * 1_000_000 - 1
    hr_values = np.zeros(10, dtype=np.float32)
    hr_valid = np.zeros(10, dtype=np.bool_)
    for index, boundary in enumerate(boundaries):
        eligible = np.flatnonzero((beat_times <= boundary) & (available * 1000 <= boundary))
        if len(eligible) and part_valid:
            hr_values[index] = 60_000 / intervals[eligible[-1]]
            hr_valid[index] = True
    hr_signal = _samples_with_available(
        hr_values, boundaries, (boundaries + 999) // 1000,
        ((start - state.timeline_origin_ms) // 1000 + np.arange(10)).astype(np.int64),
        hr_valid, "bpm", 1.0,
    )
    return hr_signal, interval_signal


def _samples(values: np.ndarray, time: np.ndarray, sequence: np.ndarray,
             valid: np.ndarray, unit: str, fs: float | None,
             delay_ms: int) -> SignalSamples:
    time_array = np.asarray(time, dtype=np.int64)
    available = (time_array + 999) // 1000 + delay_ms
    return _samples_with_available(values, time_array, available, sequence, valid, unit, fs)


def _samples_with_available(
    values: np.ndarray, time: np.ndarray, available: np.ndarray, sequence: np.ndarray,
    valid: np.ndarray, unit: str, fs: float | None,
) -> SignalSamples:
    arrays = [np.asarray(values, dtype=np.float32), np.asarray(time, dtype=np.int64),
              np.asarray(available, dtype=np.int64), np.asarray(sequence, dtype=np.int64),
              np.asarray(valid, dtype=np.bool_)]
    for array in arrays:
        array.flags.writeable = False
    return SignalSamples(*arrays, unit, fs)


def _motion(state: BetaSignalState, start: int) -> dict[str, np.ndarray]:
    logical = (start - state.timeline_origin_ms) / 1000 + np.arange(250) / 25
    amplitude = np.where((logical >= 2700) & (logical < 2820), 0.35, 0.02)
    return {axis: (amplitude * np.sin(2 * np.pi * (index + 1) * logical)).astype(np.float32)
            for index, axis in enumerate(("acc_x", "acc_y", "acc_z"))}


def _dropout(state: BetaSignalState, start: int) -> bool:
    logical = (start - state.timeline_origin_ms) // 1000
    return state.subject_index % 6 == 5 and 3300 <= logical < 3360


def _fs(name: str) -> int:
    if name.startswith("ppg"):
        return 100
    if name.startswith("acc"):
        return 25
    return 130 if name == "ecg" else 1


def _unit(_family: str, name: str) -> str:
    return {"ecg": "uV", "hr": "bpm", "eda": "uS", "skin_temp": "degC",
            "ppg_green": "a.u.", "ppg_ir": "a.u.", "ppg_red": "a.u.",
            "acc_x": "g", "acc_y": "g", "acc_z": "g"}[name]
