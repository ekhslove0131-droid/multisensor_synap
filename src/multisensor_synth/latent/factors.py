from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from multisensor_synth.domain.seeds import SeedTree
from multisensor_synth.population.circadian import circadian_wave

FACTOR_NAMES = (
    "autonomic_arousal",
    "motor_activation",
    "cognitive_load",
    "sleep_pressure",
    "sensory_context",
    "recovery_capacity",
    "social_context",
)


def generate_slow_factor(
    person_id: str,
    factor_name: str,
    seconds: NDArray[np.int64],
    base: float,
    seeds: SeedTree,
    circadian_amplitude: float,
) -> NDArray[np.float32]:
    rng = seeds.rng(f"person/{person_id}/latent/{factor_name}")
    anchor_seconds = np.arange(0, len(seconds) + 300, 300, dtype=np.int64)
    anchor_noise = rng.normal(0, 0.045, size=len(anchor_seconds))
    slow_noise = np.interp(seconds, anchor_seconds, anchor_noise)
    phase = float(rng.uniform(0, 2 * np.pi))
    circadian = circadian_amplitude * circadian_wave(
        seconds,
        phase_hour=-phase * 24 / (2 * np.pi),
    )
    ultradian = 0.025 * np.sin(2 * np.pi * seconds / 14_400 + phase / 2)
    values = np.clip(base + slow_noise + circadian + ultradian, 0, 1)
    return values.astype(np.float32)
