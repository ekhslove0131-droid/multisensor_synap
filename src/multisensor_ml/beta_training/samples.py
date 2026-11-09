"""M0 calibration and exact source-set M2 sample assembly."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import numpy.typing as npt
from synapbridge_beta_runtime.feature_types import FeatureFrame
from synapbridge_beta_runtime.model_types import (
    ModelFrame,
    SourceReference,
    SourceStandardModel,
)
from synapbridge_beta_runtime.pattern_runtime import assemble_model_frame
from synapbridge_beta_runtime.standard import calibrate_reference, fit_standard

from .cohort import LoadedCohort
from .labels import labels_at, temporal_stage_at

FAMILIES = ("watch", "h10", "muse")
SOURCE_SETS = (
    ("watch",), ("h10",), ("muse",), ("watch", "h10"),
    ("watch", "muse"), ("h10", "muse"), ("watch", "h10", "muse"),
)


@dataclass(frozen=True, slots=True)
class SampleSet:
    x: npt.NDArray[np.float32]
    y: npt.NDArray[np.int64]
    subjects: tuple[str, ...]
    response_kinds: tuple[str, ...]
    decisions_ms: npt.NDArray[np.int64]
    deviations: npt.NDArray[np.float64]
    temporal_stages: npt.NDArray[np.int64]
    event_ranges: Mapping[str, tuple[tuple[int, int], ...]]
    response_by_subject: Mapping[str, str]
    feature_names: tuple[str, ...]
    schema_sha: str


def fit_m0(
    cohort: LoadedCohort,
) -> tuple[dict[str, SourceStandardModel], dict[str, dict[str, SourceReference]]]:
    calibration = {subject: _calibration_frames(cohort, subject)
                   for subject in cohort.roles}
    standards = {}
    for family in FAMILIES:
        train = [frame for subject, by_family in calibration.items()
                 if cohort.roles[subject] == "TRAIN" for frame in by_family[family]]
        standards[family] = fit_standard(train, family)
    references = {
        subject: {
            family: calibrate_reference(standards[family], by_family[family])
            for family in FAMILIES
        }
        for subject, by_family in calibration.items()
    }
    return standards, references


def calibrate_locked(
    cohort: LoadedCohort, standards: dict[str, SourceStandardModel],
) -> dict[str, dict[str, SourceReference]]:
    return {
        subject: {
            family: calibrate_reference(standards[family], frames[family])
            for family in FAMILIES
        }
        for subject in cohort.roles
        for frames in (_calibration_frames(cohort, subject),)
    }


def assemble_samples(
    cohort: LoadedCohort,
    references: dict[str, dict[str, SourceReference]],
    source_set: tuple[str, ...],
    head: str,
    role: str,
) -> SampleSet:
    rows: list[tuple[ModelFrame, int, str, str, int]] = []
    for subject, subject_role in cohort.roles.items():
        if subject_role != role:
            continue
        history = sorted(
            (frame for family in FAMILIES for frame in cohort.frames[subject][family]),
            key=lambda frame: (frame.decision_ms, frame.family),
        )
        events = list(cohort.events[subject])
        response_kind = str(events[0]["response_kind"])
        start, end = _scored_bounds(cohort)
        decisions = sorted(
            {frame.decision_ms for frame in history if start < frame.decision_ms <= end}
        )
        for decision in decisions:
            model = assemble_model_frame(
                [frame for frame in history if frame.decision_ms <= decision],
                references[subject], source_set,
            )
            if model is None:
                continue
            current, onset = labels_at(events, subject, decision)
            label = current if head == "current" else onset
            if label is None or (head == "onset60" and decision + 60_000 > end):
                continue
            rows.append((model, int(label), subject, response_kind,
                         temporal_stage_at(events, subject, decision)))
    if not rows:
        raise ValueError("beta_samples_empty")
    first = rows[0][0]
    if any(row[0].feature_names != first.feature_names
           or row[0].schema_sha != first.schema_sha for row in rows):
        raise ValueError("beta_sample_schema_mismatch")
    return SampleSet(
        np.stack([row[0].values for row in rows]).astype(np.float32),
        np.asarray([row[1] for row in rows], dtype=np.int64),
        tuple(row[2] for row in rows),
        tuple(row[3] for row in rows),
        np.asarray([row[0].decision_ms for row in rows], dtype=np.int64),
        np.asarray([row[0].current_deviation for row in rows], dtype=np.float64),
        np.asarray([row[4] for row in rows], dtype=np.int64),
        MappingProxyType({
            subject: tuple((int(event["onset_ms"]), int(event["offset_ms"]))
                           for event in cohort.events[subject])
            for subject, subject_role in cohort.roles.items() if subject_role == role
        }),
        MappingProxyType({
            subject: str(cohort.events[subject][0]["response_kind"])
            for subject, subject_role in cohort.roles.items() if subject_role == role
        }),
        first.feature_names,
        first.schema_sha,
    )


def _calibration_frames(
    cohort: LoadedCohort, subject: str,
) -> dict[str, tuple[FeatureFrame, ...]]:
    config = cohort.manifest["config"]
    origin = int(cohort.manifest["timeline_origin_ms"])
    start = origin + int(config["prefill_seconds"]) * 1000
    end = start + int(config["calibration_seconds"]) * 1000
    return {
        family: tuple(frame for frame in cohort.frames[subject][family]
                      if start < frame.decision_ms <= end)
        for family in FAMILIES
    }


def _scored_bounds(cohort: LoadedCohort) -> tuple[int, int]:
    config = cohort.manifest["config"]
    origin = int(cohort.manifest["timeline_origin_ms"])
    start = origin + (int(config["prefill_seconds"])
                      + int(config["calibration_seconds"])) * 1000
    return start, origin + int(config["duration_seconds"]) * 1000
