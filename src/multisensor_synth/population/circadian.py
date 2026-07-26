from __future__ import annotations

import numpy as np


def circadian_wave(seconds: np.ndarray, phase_hour: float = 15.0) -> np.ndarray:
    phase_seconds = phase_hour * 3600
    return np.sin(2 * np.pi * (seconds - phase_seconds) / 86_400)
