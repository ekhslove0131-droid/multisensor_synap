from __future__ import annotations

import math

import numpy as np
from numpy.random import Generator

from multisensor_synth.config.models import BetaDistribution, LogNormal, TruncatedNormal


def sample_truncated_normal(config: TruncatedNormal, rng: Generator) -> float:
    for _ in range(10_000):
        value = float(rng.normal(config.mean, config.std))
        if config.min <= value <= config.max:
            return value
    return float(np.clip(config.mean, config.min, config.max))


def sample_lognormal(config: LogNormal, rng: Generator) -> float:
    sigma = math.log(config.geometric_std)
    return float(rng.lognormal(mean=math.log(config.median), sigma=sigma))


def sample_beta(config: BetaDistribution, rng: Generator) -> float:
    return float(rng.beta(config.alpha, config.beta))
