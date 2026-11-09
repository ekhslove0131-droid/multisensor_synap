from __future__ import annotations

import numpy as np

from multisensor_ml.beta_training.selection import (
    _episode_metrics,
    choose_candidate,
    valid_negative_exposure_hours,
)
from multisensor_ml.beta_training.train import BetaCandidate


class _Estimator:
    pass


def test_validation_selection_uses_declared_grid_and_lr_tie_break() -> None:
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    probability = np.array([0.1, 0.2, 0.8, 0.9])
    candidates = [
        BetaCandidate("HGB", "watch", "current", _Estimator(), 1, 2),
        BetaCandidate("LR", "watch", "current", _Estimator(), 1, 2),
    ]
    selected = choose_candidate(candidates, {"HGB": probability, "LR": probability}, y,
                                sensitivity_gate=0.7)
    assert selected.algorithm == "LR"
    assert selected.threshold in np.arange(0.05, 1.0, 0.05)


def test_current_alarm_starting_before_onset_matches_approved_lead_window() -> None:
    predicted = np.array([True, True, True, True, False, False])
    labels = np.array([0, 0, 1, 1, 0, 0], dtype=np.int64)
    sensitivity, false_alarms = _episode_metrics(
        predicted,
        labels,
        None,
        np.arange(6, dtype=np.int64) * 10_000,
        lead_tolerance_ms=60_000,
    )
    assert sensitivity == 1.0
    assert false_alarms == 0


def test_selection_prioritizes_sensitivity_before_far() -> None:
    labels: list[int] = []
    lr: list[float] = []
    hgb: list[float] = []
    subjects: list[str] = []
    decisions: list[int] = []
    for subject_no in range(4):
        subject = f"subject-{subject_no}"
        local = np.zeros(110, dtype=np.int64)
        local[2:4] = 1
        labels.extend(local.tolist())
        lr.extend((np.where(local == 1, 0.99, 0.01)
                   if subject_no < 3 else np.full(110, 0.01)).tolist())
        hgb_values = np.where(local == 1, 0.99, 0.01)
        if subject_no == 0:
            hgb_values[100:102] = 0.99
        hgb.extend(hgb_values.tolist())
        subjects.extend([subject] * 110)
        decisions.extend((np.arange(110) * 10_000).tolist())
    candidates = [
        BetaCandidate("LR", "watch", "current", _Estimator(), 1, 2),
        BetaCandidate("HGB", "watch", "current", _Estimator(), 1, 2),
    ]
    selected = choose_candidate(
        candidates, {"LR": np.asarray(lr), "HGB": np.asarray(hgb)},
        np.asarray(labels), sensitivity_gate=0.7, subjects=tuple(subjects),
        decisions_ms=np.asarray(decisions),
    )
    assert selected.algorithm == "HGB"
    assert selected.sensitivity == 1.0
    assert 0 < selected.far_per_hour <= 1.0


def test_far_exposure_counts_only_valid_negative_windows() -> None:
    labels = np.array([0] * 360 + [1] * 180, dtype=np.int64)
    assert valid_negative_exposure_hours(labels) == 1.0


def test_exact_event_metadata_preserves_offset_and_excluded_event_denominator() -> None:
    times = np.array([0, 10_000, 20_000, 30_000, 40_000], dtype=np.int64)
    labels = np.zeros(5, dtype=np.int64)
    sensitivity, false_alarms = _episode_metrics(
        np.array([False, False, False, True, True]), labels,
        ("subject",) * 5, times, lead_tolerance_ms=60_000,
        event_ranges={"subject": ((20_000, 40_000),)}, head="current",
    )
    assert sensitivity == 1.0
    assert false_alarms == 0
    missed, _ = _episode_metrics(
        np.zeros(5, dtype=bool), labels, ("subject",) * 5, times,
        lead_tolerance_ms=60_000,
        event_ranges={"subject": ((20_000, 40_000),)}, head="current",
    )
    assert missed == 0.0
