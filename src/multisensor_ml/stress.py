from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

StressKind = Literal["gaussian", "block_missing", "time_shift", "latent_dropout"]


@dataclass(frozen=True, slots=True)
class StressScenario:
    kind: StressKind
    magnitude: float
    seed: int


def apply_stress(frame: pd.DataFrame, scenario: StressScenario) -> pd.DataFrame:
    """Apply a deterministic feature-level Goal 1.5 stress perturbation."""

    result = frame.copy(deep=True)
    numeric = list(result.select_dtypes(include=[np.number]).columns)
    rng = np.random.default_rng(scenario.seed)
    if scenario.kind == "gaussian":
        noise = rng.normal(0, scenario.magnitude, size=(len(result), len(numeric)))
        noisy = result[numeric].to_numpy(dtype=np.float64) + noise
        for index, column in enumerate(numeric):
            result[column] = noisy[:, index].astype(np.float32)
    elif scenario.kind == "block_missing":
        length = min(int(scenario.magnitude), len(result))
        start = int(rng.integers(0, max(len(result) - length + 1, 1)))
        result.loc[result.index[start : start + length], numeric] = 0
    elif scenario.kind == "time_shift":
        shift = int(scenario.magnitude)
        result.loc[:, numeric] = result[numeric].shift(shift, fill_value=0)
    elif scenario.kind == "latent_dropout":
        latent_prefixes = sorted(
            {column.split("__", 1)[0] for column in numeric if "__" in column}
        )
        if scenario.magnitude <= 1:
            count = int(np.ceil(len(latent_prefixes) * scenario.magnitude))
        else:
            count = int(scenario.magnitude)
        count = min(max(count, 0), len(latent_prefixes))
        dropped = set(rng.choice(latent_prefixes, size=count, replace=False).tolist())
        columns = [
            column for column in numeric if column.split("__", 1)[0] in dropped
        ]
        result.loc[:, columns] = 0
    else:
        raise ValueError(f"unsupported stress kind: {scenario.kind}")
    return result
