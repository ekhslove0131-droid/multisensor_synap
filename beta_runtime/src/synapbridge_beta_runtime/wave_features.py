"""Observed-only waveform statistics."""

from __future__ import annotations

import numpy as np
from scipy.signal import welch


def slope(values: np.ndarray, time_us: np.ndarray) -> float:
    if len(values) < 2:
        raise ValueError("slope_insufficient")
    time = (time_us - time_us[0]).astype(np.float64) / 1_000_000
    centered = time - np.mean(time)
    denominator = float(np.dot(centered, centered))
    if denominator <= 0:
        raise ValueError("slope_insufficient")
    return float(np.dot(centered, values - np.mean(values)) / denominator)


def amplitude(values: np.ndarray) -> float:
    if len(values) < 3:
        raise ValueError("amplitude_insufficient")
    return float(np.quantile(values, 0.95) - np.quantile(values, 0.05))


def change_rate(current: float, prior: float, elapsed_seconds: float) -> float:
    if elapsed_seconds <= 0:
        raise ValueError("change_rate_elapsed_invalid")
    return float((current - prior) / elapsed_seconds)


def eeg_features(values: np.ndarray, fs: float) -> tuple[float, float, float, float]:
    if len(values) < fs * 4:
        raise ValueError("eeg_insufficient")
    frequency, power = welch(values, fs=fs, nperseg=int(fs * 4), noverlap=int(fs * 2))
    denominator = float(power[(frequency >= 1) & (frequency < 40)].sum())
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError("eeg_power_invalid")
    bands = [float(power[(frequency >= lo) & (frequency < hi)].sum() / denominator)
             for lo, hi in ((4, 8), (8, 13), (13, 30))]
    return bands[0], bands[1], bands[2], float(np.log(max(denominator, 1e-12)))
