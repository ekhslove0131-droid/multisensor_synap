from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from multisensor_synth.config.models import PhaseDurations

PHASE_ORDER = (
    "pre_early",
    "pre_late",
    "onset",
    "peak",
    "recovery_early",
    "recovery_late",
    "post",
)


def sample_phase_durations(
    config: PhaseDurations, rng: np.random.Generator
) -> dict[str, int]:
    ranges = (
        config.pre_early_sec,
        config.pre_late_sec,
        config.onset_sec,
        config.peak_sec,
        config.recovery_early_sec,
        config.recovery_late_sec,
        config.post_sec,
    )
    return {
        name: int(rng.integers(bounds.min, bounds.max + 1))
        for name, bounds in zip(PHASE_ORDER, ranges, strict=True)
    }


def boundaries_from_start(
    start_time_ns: int, durations: Mapping[str, int]
) -> dict[str, int]:
    cursor = start_time_ns
    boundaries = {"start_time_ns": cursor}
    cursor += durations["pre_early"] * 1_000_000_000
    boundaries["pre_late_start_time_ns"] = cursor
    cursor += durations["pre_late"] * 1_000_000_000
    boundaries["onset_time_ns"] = cursor
    cursor += durations["onset"] * 1_000_000_000
    boundaries["peak_start_time_ns"] = cursor
    cursor += durations["peak"] * 1_000_000_000
    boundaries["peak_end_time_ns"] = cursor
    cursor += durations["recovery_early"] * 1_000_000_000
    boundaries["recovery_early_end_time_ns"] = cursor
    cursor += durations["recovery_late"] * 1_000_000_000
    boundaries["recovery_late_end_time_ns"] = cursor
    cursor += durations["post"] * 1_000_000_000
    boundaries["end_time_ns"] = cursor
    return boundaries


def phase_curve(name: str, x: NDArray[np.float64]) -> NDArray[np.float64]:
    clipped = np.clip(x, 0.0, 1.0)
    if name == "smoothstep":
        values = clipped * clipped * (3 - 2 * clipped)
    elif name == "exponential_rise":
        values = np.expm1(3 * clipped) / np.expm1(3)
    elif name == "sigmoid":
        raw = 1 / (1 + np.exp(-10 * (clipped - 0.5)))
        values = (raw - raw[0]) / (raw[-1] - raw[0])
    elif name == "plateau":
        values = np.ones_like(clipped)
    elif name == "decay":
        values = np.exp(-3 * clipped)
    else:
        raise ValueError(f"unknown phase curve: {name}")
    return np.clip(values, 0.0, 1.0)
