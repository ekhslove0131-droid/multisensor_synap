"""Validation-only alignment for frozen causal stage-policy selection."""

from __future__ import annotations

import numpy as np

from .samples import SampleSet
from .stage_policy import select_stage_policy

ValidationHead = tuple[SampleSet, np.ndarray, float]


def select_policy_from_validation(
    values: dict[tuple[str, str], ValidationHead],
) -> dict[str, object]:
    source = "watch+h10+muse"
    current = values.get((source, "current"))
    onset = values.get((source, "onset60"))
    if current is None or onset is None:
        raise ValueError("stage_validation_heads_missing")
    current_samples, current_probability, current_threshold = current
    onset_samples, onset_probability, onset_threshold = onset
    current_positive = np.asarray(current_probability) > current_threshold
    onset_by_key = {
        (subject, int(decision)): bool(probability > onset_threshold)
        for subject, decision, probability in zip(
            onset_samples.subjects, onset_samples.decisions_ms,
            onset_probability, strict=True,
        )
    }
    onset_positive = np.asarray([
        onset_by_key.get((subject, int(decision)), False)
        for subject, decision in zip(
            current_samples.subjects, current_samples.decisions_ms, strict=True,
        )
    ], dtype=bool)
    return select_stage_policy(
        current_samples.deviations,
        current_positive,
        onset_positive,
        current_samples.temporal_stages,
        current_samples.subjects,
        current_samples.decisions_ms,
    )
