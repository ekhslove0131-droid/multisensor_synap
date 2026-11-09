"""Continuous phase-integrated cardiac and delayed PPG observations."""

from __future__ import annotations

import hashlib
from functools import lru_cache

import neurokit2 as nk
import numpy as np
from scipy.signal import find_peaks

from .config import BetaSignalState


def heart_rate(state: BetaSignalState, time_seconds: np.ndarray) -> np.ndarray:
    rate = np.full_like(time_seconds, state.baseline_hr, dtype=np.float64)
    response = _response_gain(state)
    for onset_ms, offset_ms in state.events_ms:
        onset, offset = onset_ms / 1000, offset_ms / 1000
        pre = np.clip((time_seconds - (onset - 60)) / 60, 0, 1)
        post = np.exp(-np.maximum(time_seconds - offset, 0) / 45)
        active = (time_seconds >= onset) & (time_seconds < offset)
        rate += response * np.where(time_seconds < onset, pre, np.where(active, 1.0, post))
    logical = time_seconds - state.timeline_origin_ms / 1000
    rate += 25.0 * ((logical >= 2700) & (logical < 2820))
    return np.clip(rate, 40, 180)


def beat_phase(state: BetaSignalState, time_seconds: np.ndarray) -> np.ndarray:
    origin = state.timeline_origin_ms / 1000
    cycles = state.baseline_hr * np.maximum(time_seconds - origin, 0) / 60
    gain = _response_gain(state)
    for onset_ms, offset_ms in state.events_ms:
        onset, offset = onset_ms / 1000, offset_ms / 1000
        cycles += gain * _event_integral(time_seconds, onset, offset) / 60
    exercise_start, exercise_end = origin + 2700, origin + 2820
    overlap = np.clip(np.minimum(time_seconds, exercise_end) - exercise_start,
                      0, exercise_end - exercise_start)
    cycles += 25.0 * overlap / 60
    return cycles


def ecg(state: BetaSignalState, start_ms: int, fs: int = 130) -> np.ndarray:
    time = _time(start_ms, fs)
    phase = np.mod(beat_phase(state, time), 1.0)
    grid, template = _neurokit_ecg_template()
    morphology = np.interp(phase, grid, template)
    noise = _rng(state, "ecg", start_ms).normal(0, 12, len(time))
    return np.asarray(950 * morphology + noise, dtype=np.float32)


def ppg(state: BetaSignalState, start_ms: int, fs: int = 100) -> np.ndarray:
    time = _time(start_ms, fs)
    phase = np.mod(beat_phase(state, time - 0.2), 1.0)
    pulse = np.exp(-0.5 * (np.minimum(phase, 1 - phase) / 0.08) ** 2)
    noise = _rng(state, "ppg", start_ms).normal(0, 0.015, len(time))
    return np.asarray(0.2 + pulse + noise, dtype=np.float32)


def observed_intervals(
    values: np.ndarray, time_us: np.ndarray, fs: float, kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    prominence = 150.0 if kind == "ecg" else 0.25
    peaks, _ = find_peaks(values, distance=max(1, int(fs * 0.3)), prominence=prominence)
    if len(peaks) < 2:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
    peak_times = time_us[peaks]
    intervals = np.diff(peak_times).astype(np.float64) / 1000
    valid = (intervals > 250) & (intervals < 2000)
    return intervals[valid].astype(np.float32), peak_times[1:][valid]


def _event_integral(time: np.ndarray, onset: float, offset: float) -> np.ndarray:
    start = onset - 60
    result = np.zeros_like(time, dtype=np.float64)
    pre = (time > start) & (time < onset)
    result[pre] = (time[pre] - start) ** 2 / 120
    active = (time >= onset) & (time < offset)
    result[active] = 30 + time[active] - onset
    post = time >= offset
    result[post] = 30 + offset - onset + 45 * (1 - np.exp(-(time[post] - offset) / 45))
    return result


@lru_cache(maxsize=1)
def _neurokit_ecg_template() -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(nk.ecg_simulate(
        duration=20, sampling_rate=130, heart_rate=60, heart_rate_std=0,
        noise=0, random_state=20261006, method="ecgsyn",
    ), dtype=np.float64)
    peaks, _ = find_peaks(raw, distance=78, prominence=0.3)
    if len(peaks) < 4:
        raise ValueError("neurokit_ecg_template_invalid")
    segment = raw[peaks[1]:peaks[2] + 1]
    segment -= np.median(segment)
    scale = np.max(np.abs(segment))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("neurokit_ecg_template_invalid")
    return np.linspace(0, 1, len(segment)), segment / scale


def _response_gain(state: BetaSignalState) -> float:
    return {"RESPONSIVE": 18.0, "WEAK": 7.0, "INVERSE": -9.0,
            "NO_RESPONSE": 0.0}[state.response_kind]


def _time(start_ms: int, fs: int) -> np.ndarray:
    return start_ms / 1000 + np.arange(fs * 10, dtype=np.float64) / fs


def _rng(state: BetaSignalState, name: str, start_ms: int) -> np.random.Generator:
    value = hashlib.sha256(f"{state.seed}|{state.subject_uuid}|{name}|{start_ms}".encode()).digest()
    return np.random.default_rng(int.from_bytes(value[:8], "big"))
