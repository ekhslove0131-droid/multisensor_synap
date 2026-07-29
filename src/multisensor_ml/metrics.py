from __future__ import annotations

from collections.abc import Iterable
from itertools import pairwise

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    recall_score,
)

EVENT_SEGMENT_KEYS = (
    "dataset_id",
    "person_key",
    "day_key",
    "session_id",
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
    truth_mask = truth.astype(bool)
    truth_events = _segments(truth_mask)
    detected = sum(bool(predicted[start : end + 1].any()) for start, end in truth_events)
    false_alerts = sum(
        not bool(truth_mask[start : end + 1].any())
        for start, end in _segments(predicted)
    )
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


def evaluate_segmented_probabilities(
    frame: pd.DataFrame,
    *,
    threshold: float,
    truth_column: str,
    probability_column: str,
) -> dict[str, float | int | list[list[int]]]:
    required_columns = {
        *EVENT_SEGMENT_KEYS,
        "canonical_time",
        truth_column,
        probability_column,
    }
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(
            f"event evaluation requires columns: {missing_columns}"
        )
    if frame.loc[:, list(EVENT_SEGMENT_KEYS)].isna().any().any():
        raise ValueError("event segment identity contains null values")
    truth_values = frame[truth_column]
    if (
        truth_values.isna().any()
        or not set(truth_values.unique()).issubset({0, 1})
    ):
        raise ValueError("truth must contain exact binary values")
    probability_values = frame[probability_column].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(probability_values).all()
        or (probability_values < 0).any()
        or (probability_values > 1).any()
    ):
        raise ValueError("probability must be finite and within [0, 1]")
    ordered = frame.sort_values(
        [*EVENT_SEGMENT_KEYS, "canonical_time"],
        kind="mergesort",
    )
    detected = 0
    missed = 0
    false_alerts = 0
    duration_hours = 0.0
    for _, segment in ordered.groupby(
        list(EVENT_SEGMENT_KEYS),
        sort=False,
        dropna=False,
    ):
        times = pd.to_datetime(segment["canonical_time"], utc=True)
        if times.duplicated().any():
            raise ValueError("duplicate canonical_time inside event segment")
        contiguous_run = times.diff().ne(pd.Timedelta(seconds=1)).cumsum()
        for _, run in segment.groupby(contiguous_run, sort=False):
            run_duration = len(run) / 3600
            run_truth = run[truth_column].to_numpy(dtype=np.int8)
            run_predicted = (
                run[probability_column].to_numpy(dtype=np.float64) >= threshold
            )
            run_detected, run_missed, run_false_alerts = _event_counts(
                run_truth,
                run_predicted,
            )
            detected += run_detected
            missed += run_missed
            false_alerts += run_false_alerts
            duration_hours += run_duration

    result = evaluate_probabilities(
        ordered[truth_column].to_numpy(dtype=np.int8),
        ordered[probability_column].to_numpy(dtype=np.float64),
        threshold=threshold,
        duration_hours=duration_hours,
    )
    event_recall = detected / (detected + missed) if detected + missed else 0.0
    event_precision = (
        detected / (detected + false_alerts)
        if detected + false_alerts
        else 0.0
    )
    event_f1 = (
        2 * event_precision * event_recall / (event_precision + event_recall)
        if event_precision + event_recall
        else 0.0
    )
    result.update(
        {
            "event_recall": event_recall,
            "event_precision": event_precision,
            "event_f1": event_f1,
            "detected_events": detected,
            "missed_events": missed,
            "false_alerts": false_alerts,
            "false_alerts_per_hour": (
                false_alerts / duration_hours if duration_hours else 0.0
            ),
        }
    )
    return result


def select_event_threshold(
    truth: NDArray[np.int8],
    probability: NDArray[np.float64],
    *,
    duration_hours: float,
    candidates: Iterable[float] | None = None,
) -> float:
    """Choose validation event-F1 threshold; ties prefer fewer false alerts then higher cutoff."""

    if candidates is None:
        return _select_event_threshold_incremental(
            truth,
            probability,
            duration_hours=duration_hours,
        )

    values = np.asarray(list(candidates), dtype=np.float64)
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


def _select_event_threshold_incremental(
    truth: NDArray[np.int8],
    probability: NDArray[np.float64],
    *,
    duration_hours: float,
) -> float:
    """Evaluate every rounded score threshold without rescanning the full series."""

    if len(truth) != len(probability):
        raise ValueError("truth and probability lengths must match")
    thresholds = np.unique(np.round(probability, 6))
    if not len(thresholds):
        return 0.5

    truth_mask = truth.astype(bool)
    truth_segments = _segments(truth_mask)
    event_ids = np.full(len(truth), -1, dtype=np.int32)
    for event_id, (start, end) in enumerate(truth_segments):
        event_ids[start : end + 1] = event_id

    order = np.argsort(probability, kind="stable")[::-1]
    active = np.zeros(len(truth), dtype=bool)
    detected_events = np.zeros(len(truth_segments), dtype=bool)
    detected_count = 0
    false_alerts = 0
    cursor = 0
    best: tuple[float, float, float] | None = None

    for threshold in thresholds[::-1]:
        while cursor < len(order) and probability[order[cursor]] >= threshold:
            index = int(order[cursor])
            cursor += 1
            active[index] = True
            if truth_mask[index]:
                event_id = int(event_ids[index])
                if event_id >= 0 and not detected_events[event_id]:
                    detected_events[event_id] = True
                    detected_count += 1
                continue
            left_active = (
                index > 0
                and not truth_mask[index - 1]
                and active[index - 1]
            )
            right_active = (
                index + 1 < len(active)
                and not truth_mask[index + 1]
                and active[index + 1]
            )
            false_alerts += 1 - int(left_active) - int(right_active)

        missed = len(truth_segments) - detected_count
        recall = (
            detected_count / (detected_count + missed)
            if detected_count + missed
            else 0.0
        )
        precision = (
            detected_count / (detected_count + false_alerts)
            if detected_count + false_alerts
            else 0.0
        )
        event_f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        false_alerts_per_hour = (
            false_alerts / duration_hours if duration_hours else 0.0
        )
        score = (
            event_f1,
            -false_alerts_per_hour,
            float(threshold),
        )
        if best is None or score > best:
            best = score

    return 0.5 if best is None else best[2]
