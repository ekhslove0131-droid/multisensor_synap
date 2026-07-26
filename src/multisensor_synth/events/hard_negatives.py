from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from multisensor_synth.domain.contracts import HARD_NEGATIVE_TYPES


def allocate_hard_negative_types(
    total: int,
    probabilities: Mapping[str, float],
    rng: np.random.Generator,
) -> list[str]:
    if total <= 0:
        return []
    required = list(HARD_NEGATIVE_TYPES[: min(total, len(HARD_NEGATIVE_TYPES))])
    remaining = total - len(required)
    if remaining:
        names = tuple(probabilities)
        weights = tuple(probabilities.values())
        required.extend(str(value) for value in rng.choice(names, size=remaining, p=weights))
    return required
