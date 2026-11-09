"""Four-channel stochastic band mixtures for the synthetic beta realm."""

from __future__ import annotations

import hashlib

import numpy as np

from .cardiac import heart_rate
from .config import BetaSignalState


def eeg_channel(state: BetaSignalState, start_ms: int, channel: int, fs: int = 256) -> np.ndarray:
    time = start_ms / 1000 + np.arange(fs * 10, dtype=np.float64) / fs
    response = np.clip((heart_rate(state, time) - state.baseline_hr) / 18.0, -1, 1)
    rng = _rng(state, channel, start_ms)
    alpha = (13.0 - 3.0 * response) * np.sin(2 * np.pi * (9.5 + channel * 0.2) * time)
    beta = (5.0 + 4.0 * response) * np.sin(2 * np.pi * (18 + channel) * time + channel)
    theta = 7.0 * np.sin(2 * np.pi * (5.5 + channel * 0.1) * time)
    noise = rng.normal(0, 2.5, len(time))
    return (alpha + beta + theta + noise).astype(np.float32)


def _rng(state: BetaSignalState, channel: int, start_ms: int) -> np.random.Generator:
    digest = hashlib.sha256(
        f"{state.seed}|{state.subject_uuid}|eeg|{channel}|{start_ms}".encode()
    ).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))
