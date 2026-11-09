"""Validation-only candidate and threshold selection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .train import BetaCandidate


@dataclass(frozen=True, slots=True)
class BetaSelection:
    algorithm: str
    source_set: str
    head: str
    threshold: float
    sensitivity: float
    far_per_hour: float
    median_lead_seconds: float | None
    candidate: BetaCandidate


def choose_candidate(candidates: list[BetaCandidate] | tuple[BetaCandidate, ...],
                     probabilities: dict[str, np.ndarray], labels: np.ndarray,
                     *, sensitivity_gate: float,
                     subjects: tuple[str, ...] | None = None,
                     decisions_ms: np.ndarray | None = None,
                     event_ranges: Mapping[str, tuple[tuple[int, int], ...]] | None = None,
                     ) -> BetaSelection:
    y = np.asarray(labels, dtype=np.int64)
    exposure_hours = valid_negative_exposure_hours(y)
    if exposure_hours <= 0 or len(np.unique(y)) != 2:
        raise ValueError("validation_selection_invalid")
    qualified = []
    for candidate in candidates:
        probability = np.asarray(probabilities[candidate.algorithm], dtype=float)
        if len(probability) != len(y) or not np.isfinite(probability).all():
            raise ValueError("validation_probability_invalid")
        for threshold in np.arange(0.05, 1.0, 0.05):
            predicted = probability >= threshold
            sensitivity, false_positive, lead = _episode_statistics(
                predicted, y, subjects, decisions_ms,
                lead_tolerance_ms=60_000 if candidate.head == "current" else 0,
                event_ranges=event_ranges, head=candidate.head,
            )
            far = false_positive / exposure_hours
            if sensitivity >= sensitivity_gate and far <= 1.0:
                lead_rank = lead if lead is not None else -float("inf")
                qualified.append((-sensitivity, far, -lead_rank, -float(threshold),
                                  0 if candidate.algorithm == "LR" else 1, candidate,
                                  float(threshold), sensitivity, lead))
    if not qualified:
        raise ValueError("NO_QUALIFIED_BETA_MODEL")
    neg_sensitivity, far, _, _, _, candidate, threshold, sensitivity, lead = min(qualified)
    assert -neg_sensitivity == sensitivity
    return BetaSelection(candidate.algorithm, candidate.source_set, candidate.head,
                         threshold, sensitivity, far, lead, candidate)


def valid_negative_exposure_hours(labels: np.ndarray) -> float:
    values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1 or not set(np.unique(values)) <= {0, 1}:
        raise ValueError("negative_exposure_labels_invalid")
    return float(np.sum(values == 0) * 10 / 3600)


def _episode_metrics(
    predicted: np.ndarray,
    labels: np.ndarray,
    subjects: tuple[str, ...] | None,
    decisions_ms: np.ndarray | None,
    *,
    lead_tolerance_ms: int = 0,
    event_ranges: Mapping[str, tuple[tuple[int, int], ...]] | None = None,
    head: str = "current",
) -> tuple[float, int]:
    sensitivity, false_alarms, _ = _episode_statistics(
        predicted, labels, subjects, decisions_ms,
        lead_tolerance_ms=lead_tolerance_ms,
        event_ranges=event_ranges, head=head,
    )
    return sensitivity, false_alarms


def _episode_statistics(
    predicted: np.ndarray,
    labels: np.ndarray,
    subjects: tuple[str, ...] | None,
    decisions_ms: np.ndarray | None,
    *,
    lead_tolerance_ms: int = 0,
    event_ranges: Mapping[str, tuple[tuple[int, int], ...]] | None = None,
    head: str = "current",
) -> tuple[float, int, float | None]:
    groups = subjects or tuple("single" for _ in labels)
    times = (np.asarray(decisions_ms, dtype=np.int64) if decisions_ms is not None
             else np.arange(len(labels), dtype=np.int64) * 10_000)
    event_count = matched = false_alarms = 0
    leads: list[float] = []
    subject_order = (tuple(event_ranges) if event_ranges is not None
                     else tuple(dict.fromkeys(groups)))
    for subject in subject_order:
        indices = [index for index, value in enumerate(groups) if value == subject]
        ordered = sorted(indices, key=lambda index: int(times[index]))
        alarms = _alarm_starts(predicted[ordered], times[ordered])
        truth = (list(event_ranges.get(subject, ())) if event_ranges is not None
                 else _positive_ranges(labels[ordered], times[ordered]))
        event_count += len(truth)
        used: set[int] = set()
        for alarm in alarms:
            match = next((number for number, (start, end) in enumerate(truth)
                          if number not in used and _alarm_matches(
                              alarm, start, end, head, lead_tolerance_ms,
                          )), None)
            if match is None:
                false_alarms += 1
            else:
                matched += 1
                used.add(match)
                start, _ = truth[match]
                leads.append((start - alarm) / 1000)
    return (matched / event_count if event_count else 0.0, false_alarms,
            float(np.median(leads)) if leads else None)


def _alarm_matches(alarm: int, onset: int, offset: int, head: str,
                   lead_tolerance_ms: int) -> bool:
    if head == "current":
        return onset - lead_tolerance_ms <= alarm <= offset
    if head == "onset60":
        return onset - 60_000 <= alarm < onset
    raise ValueError("beta_head_invalid")


def _alarm_starts(predicted: np.ndarray, times: np.ndarray) -> list[int]:
    starts: list[int] = []
    positive_count = negative_count = 0
    active = False
    previous = None
    for value, time in zip(predicted, times, strict=True):
        if previous is None or int(time) - previous != 10_000:
            positive_count = negative_count = 0
            active = False
        if value:
            positive_count += 1
            negative_count = 0
            if positive_count >= 2 and not active:
                starts.append(int(time))
                active = True
        else:
            negative_count += 1
            positive_count = 0
            if negative_count >= 2:
                active = False
        previous = int(time)
    return starts


def _positive_ranges(labels: np.ndarray, times: np.ndarray) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = previous = None
    for label, time in zip(labels, times, strict=True):
        if label and (start is None or previous is None or int(time) - previous != 10_000):
            if start is not None and previous is not None:
                ranges.append((start, previous))
            start = int(time)
        elif not label and start is not None and previous is not None:
            ranges.append((start, previous))
            start = None
        previous = int(time)
    if start is not None and previous is not None:
        ranges.append((start, previous))
    return ranges
