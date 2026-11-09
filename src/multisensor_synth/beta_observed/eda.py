"""Low-rate EDA and skin-temperature observations."""

from __future__ import annotations

from functools import lru_cache

import neurokit2 as nk
import numpy as np

from .cardiac import heart_rate
from .config import BetaSignalState


def eda_values(state: BetaSignalState, start_ms: int) -> np.ndarray:
    time = start_ms / 1000 + np.arange(10, dtype=np.float64)
    response = (heart_rate(state, time) - state.baseline_hr) / 18.0
    logical = np.floor(time - state.timeline_origin_ms / 1000).astype(int)
    template = _neurokit_eda_template()
    morphology = template[np.mod(logical, len(template))]
    return np.asarray(np.maximum(0, 4.0 + 1.2 * response + morphology),
                      dtype=np.float32)


def temperature_values(state: BetaSignalState, start_ms: int) -> np.ndarray:
    time = start_ms / 1000 + np.arange(10, dtype=np.float64)
    response = (heart_rate(state, time) - state.baseline_hr) / 18.0
    return (33.0 - 0.2 * response + 0.05 * np.sin(time / 120)).astype(np.float32)


@lru_cache(maxsize=1)
def _neurokit_eda_template() -> np.ndarray:
    raw = np.asarray(nk.eda_simulate(
        duration=600, sampling_rate=1, noise=0, scr_number=10, drift=0,
        random_state=20261006,
    ), dtype=np.float64)
    centered = raw - np.median(raw)
    scale = np.max(np.abs(centered))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("neurokit_eda_template_invalid")
    return np.asarray(centered / scale * 0.05, dtype=np.float64)
