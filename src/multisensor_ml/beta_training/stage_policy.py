"""Validation-only causal stage replay matching the Apps stage contract."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def select_stage_policy(
    deviations: np.ndarray, current_positive: np.ndarray,
    onset_positive: np.ndarray, reference_stages: np.ndarray,
    subjects: tuple[str, ...], decisions_ms: np.ndarray,
) -> dict[str, object]:
    values = np.asarray(deviations, dtype=float)
    current = np.asarray(current_positive, dtype=bool)
    onset = np.asarray(onset_positive, dtype=bool)
    truth = np.asarray(reference_stages, dtype=np.int64)
    times = np.asarray(decisions_ms, dtype=np.int64)
    if (values.ndim != 1 or any(array.shape != values.shape
                                for array in (current, onset, truth, times))
            or len(subjects) != len(values) or not np.isfinite(values).all()
            or not set(np.unique(truth)) <= set(range(6))):
        raise ValueError("stage_validation_invalid")
    scored = []
    for d_on in (1.0, 1.5, 2.0):
        predicted, reversals = causal_stages(
            values, current, onset, subjects, times, d_on,
        )
        phase_mask = (truth > 0) | (predicted > 0)
        macro_f1 = (float(f1_score(
            truth[phase_mask], predicted[phase_mask], labels=[1, 2, 3, 4, 5],
            average="macro", zero_division=0,
        )) if np.any(phase_mask) else 0.0)
        scored.append((reversals, -macro_f1, -d_on, d_on, macro_f1))
    reversals, _, _, d_on, macro_f1 = min(scored)
    return {"version": "synthetic_beta_stage_policy_v1", "d_on": d_on,
            "d_off": d_on / 2, "selection_role": "VALIDATION",
            "selection_metric": (
                "stage_reversals_then_reference_temporal_phase_macro_f1_then_d_on"
            ),
            "validation_stage_reversals": reversals,
            "validation_phase_macro_f1": macro_f1,
            "candidate_grid": [1.0, 1.5, 2.0]}


def causal_stages(
    deviation: np.ndarray, current: np.ndarray, onset: np.ndarray,
    subjects: tuple[str, ...], times: np.ndarray, d_on: float,
) -> tuple[np.ndarray, int]:
    output = np.zeros(len(deviation), dtype=np.int64)
    reversals = 0
    for subject in dict.fromkeys(subjects):
        indices = [index for index, value in enumerate(subjects) if value == subject]
        ordered = sorted(indices, key=lambda index: int(times[index]))
        stage = 0
        counts: dict[str, int] = {}
        started: int | None = None
        previous_time: int | None = None
        previous_value: float | None = None
        peak: float | None = None
        last_positive_stage = 0
        for index in ordered:
            time, value = int(times[index]), float(deviation[index])
            missing = previous_time is not None and time - previous_time > 10_000
            if missing:
                counts = {}
                assert previous_time is not None
                if time - (previous_time + 10_000) >= 60_000 and stage:
                    output[index] = stage
                    stage, counts, started, peak = 0, {}, None, None
                    previous_value, previous_time, last_positive_stage = None, time, 0
                    continue
            closed = False
            if stage == 0:
                cp = _consecutive(counts, "current", bool(current[index]))
                fp = _consecutive(counts, "forecast", bool(onset[index]))
                if max(cp, fp) >= 2:
                    stage, started, peak, counts = 1, time, value, {}
            elif stage == 1:
                if started is not None and time - started >= 120_000:
                    closed = True
                elif _consecutive(
                    counts, "confirm", bool(current[index]) and value >= d_on,
                ) >= 2:
                    stage, counts = 2, {}
            elif stage == 2:
                plateau = _consecutive(
                    counts, "plateau",
                    value >= d_on and peak is not None and value - peak <= 0.05,
                )
                down = _consecutive(
                    counts, "downturn", value >= d_on and previous_value is not None
                    and value < previous_value,
                )
                if max(plateau, down) >= 2:
                    stage, counts = 3, {}
            elif stage == 3:
                falling = (peak is not None and peak - value >= 0.25
                           and previous_value is not None and value < previous_value)
                if _consecutive(counts, "fall", falling) >= 2:
                    stage, counts = 4, {}
            elif stage == 4:
                if _consecutive(
                    counts, "return", value <= d_on / 2 and not current[index],
                ) >= 3:
                    stage, counts = 5, {}
            elif _consecutive(counts, "post", value <= d_on / 2) >= 3:
                stage, counts, closed = 0, {}, True
            if stage and last_positive_stage and stage < last_positive_stage:
                reversals += 1
            output[index] = stage
            last_positive_stage = stage if stage else 0
            peak = max(peak, value) if peak is not None and stage else (value if stage else None)
            previous_value, previous_time = value, time
            if closed:
                stage, counts, started, peak = 0, {}, None, None
                previous_value, last_positive_stage = None, 0
    return output, reversals


def _consecutive(counts: dict[str, int], name: str, condition: bool) -> int:
    counts[name] = counts.get(name, 0) + 1 if condition else 0
    return counts[name]
