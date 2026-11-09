"""Causal cardiac interval summaries."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def ecg_rr_ms(values: np.ndarray, time_us: np.ndarray, fs: float) -> np.ndarray:
    return wave_rr_ms(values, time_us, fs, prominence=150.0)


def wave_rr_ms(
    values: np.ndarray, time_us: np.ndarray, fs: float, *, prominence: float,
) -> np.ndarray:
    if len(values) < fs * 3:
        return np.empty(0, dtype=np.float64)
    peaks, _ = find_peaks(values, distance=max(1, int(fs * 0.3)), prominence=prominence)
    return np.asarray(np.diff(time_us[peaks]), dtype=np.float64) / 1000.0


def interval_features(intervals_ms: np.ndarray) -> tuple[float, float, float]:
    valid = intervals_ms[np.isfinite(intervals_ms) & (intervals_ms > 250) &
                         (intervals_ms < 2000)]
    if len(valid) < 2:
        raise ValueError("intervals_insufficient")
    differences = np.diff(valid)
    rmssd = float(np.sqrt(np.mean(differences ** 2))) if len(differences) else 0.0
    sdnn = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
    elapsed_seconds = float(np.sum(valid[1:]) / 1000)
    if elapsed_seconds <= 0:
        raise ValueError("intervals_insufficient")
    change = float((valid[-1] - valid[0]) / elapsed_seconds)
    return rmssd, sdnn, change
