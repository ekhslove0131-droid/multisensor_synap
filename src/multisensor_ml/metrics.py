from __future__ import annotations

from collections.abc import Iterable
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    recall_score,
)


def _segments(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _event_counts(
    truth: NDArray[np.int8],
    predicted: NDArray[np.bool_],
) -> tuple[int, int, int]:
    truth_events = _segments(truth.astype(bool))
    detected = sum(bool(predicted[start : end + 1].any()) for start, end in truth_events)
    false_alerts = len(_segments(predicted & ~truth.astype(bool)))
    return detected, len(truth_events) - detected, false_alerts


def forecast_lead_times(
    truth: NDArray[np.int8],
    probability: NDArray[np.float64],
    *,
    threshold: float,
) -> list[int]:
    """Return seconds from the first alert in each forecast block to its onset."""

    predicted = probability >= threshold
    leads: list[int] = []
    for start, end in _segments(truth.astype(bool)):
        alerts = np.flatnonzero(predicted[start : end + 1])
        if len(alerts):
            leads.append(end - (start + int(alerts[0])) + 1)
    return leads


def expected_calibration_error(
    truth: NDArray[np.int8],
    probability: NDArray[np.float64],
    bins: int = 10,
) -> float:
    edges = np.linspace(0, 1, bins + 1)
    result = 0.0
    for lower, upper in pairwise(edges):
        include_upper = upper == 1
        mask = (probability >= lower) & (
            (probability <= upper) if include_upper else (probability < upper)
        )
        if mask.any():
            result += float(mask.mean()) * abs(
                float(truth[mask].mean()) - float(probability[mask].mean())
            )
    return result


def evaluate_probabilities(
    truth: NDArray[np.int8],
    probability: NDArray[np.float64],
    *,
    threshold: float,
    duration_hours: float,
) -> dict[str, float | int | list[list[int]]]:
    predicted = probability >= threshold
    detected, missed, false_alerts = _event_counts(truth, predicted)
    event_recall = detected / (detected + missed) if detected + missed else 0.0
    event_precision = detected / (detected + false_alerts) if detected + false_alerts else 0.0
    event_f1 = (
        2 * event_precision * event_recall / (event_precision + event_recall)
        if event_precision + event_recall
        else 0.0
    )
    matrix = confusion_matrix(truth, predicted, labels=[0, 1]).astype(int).tolist()
    return {
        "aucpr": float(average_precision_score(truth, probability)),
        "row_recall": float(recall_score(truth, predicted, zero_division=0)),
        "row_f1": float(f1_score(truth, predicted, zero_division=0)),
        "event_recall": event_recall,
        "event_precision": event_precision,
        "event_f1": event_f1,
        "detected_events": detected,
        "missed_events": missed,
        "false_alerts": false_alerts,
        "false_alerts_per_hour": false_alerts / duration_hours if duration_hours else 0.0,
        "brier_score": float(brier_score_loss(truth, probability)),
        "calibration_error": expected_calibration_error(truth, probability),
        "confusion_matrix": matrix,
    }


def select_event_threshold(
    truth: NDArray[np.int8],
    probability: NDArray[np.float64],
    *,
    duration_hours: float,
    candidates: Iterable[float] | None = None,
) -> float:
    """Choose validation event-F1 threshold; ties prefer fewer false alerts then higher cutoff."""

    values = (
        np.unique(np.round(probability, 6))
        if candidates is None
        else np.asarray(list(candidates), dtype=np.float64)
    )
    scored: list[tuple[float, float, float]] = []
    for threshold in values:
        predicted = probability >= threshold
        detected, missed, false_alerts = _event_counts(truth, predicted)
        recall = detected / (detected + missed) if detected + missed else 0.0
        precision = detected / (detected + false_alerts) if detected + false_alerts else 0.0
        event_f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        false_alerts_per_hour = false_alerts / duration_hours if duration_hours else 0.0
        scored.append(
            (
                event_f1,
                -false_alerts_per_hour,
                float(threshold),
            )
        )
    if not scored:
        return 0.5
    return max(scored)[2]
