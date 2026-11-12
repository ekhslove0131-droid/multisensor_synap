from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

ORACLE_LATENT_FACTORS: tuple[str, ...] = (
    "autonomic_arousal",
    "motor_activation",
    "cognitive_load",
    "sleep_pressure",
    "sensory_context",
    "recovery_capacity",
    "social_context",
)
FORBIDDEN_ORACLE_EXACT: frozenset[str] = frozenset(
    {
        "hard_negative_id",
        "hard_negative_kind",
        "hard_negative_type",
        "artifact_schedule_id",
        "participant_truth_baseline",
        "intensity_truth",
        "event_intensity_truth",
    }
)
SplitRole = Literal["train", "validation", "locked_test"]


class DatasetRecord(BaseModel):
    """One registered person dataset and its immutable lineage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1)
    source_domain: Literal["synthetic_truth_oracle", "real_observed"]
    generator_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str = Field(min_length=1)
    person_key: str = Field(min_length=1)
    time_column: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    logical_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_role: SplitRole


class SynchronizationRecord(BaseModel):
    """Public future-device alignment schema with a fail-closed oracle state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1)
    session_id: str | None
    person_key: str | None
    device_pair: str | None
    reference_device: Literal["Polar H10"]
    offset_ms: float | None
    drift_ppm: float | None
    jitter_ms: float | None = Field(default=None, ge=0)
    physiological_lag_ms: float | None
    overlap_sec: float | None = Field(default=None, ge=0)
    correlation: float | None = Field(default=None, ge=-1, le=1)
    corrected_time_axis: Literal["UTC"]
    watch_ecg_policy: Literal["calibration_only"]
    status: Literal[
        "ALIGNED",
        "REVIEW_REQUIRED",
        "INSUFFICIENT_OVERLAP",
        "NOT_AVAILABLE_TRUTH_ONLY",
    ]

    @model_validator(mode="after")
    def truth_only_has_no_measurements(self) -> SynchronizationRecord:
        measurements = (
            self.offset_ms,
            self.drift_ppm,
            self.jitter_ms,
            self.physiological_lag_ms,
            self.overlap_sec,
            self.correlation,
        )
        if self.status == "NOT_AVAILABLE_TRUTH_ONLY" and any(
            value is not None for value in measurements
        ):
            raise ValueError("truth-only synchronization must not invent measurements")
        return self


def assign_person_splits(people: Iterable[tuple[str, str]]) -> dict[tuple[str, str], SplitRole]:
    """Assign a deterministic 8/2/2 split independently inside each run."""

    grouped: dict[str, set[str]] = defaultdict(set)
    for run_id, person_id in people:
        grouped[run_id].add(person_id)

    result: dict[tuple[str, str], SplitRole] = {}
    for run_id, person_ids in sorted(grouped.items()):
        ordered = sorted(
            person_ids,
            key=lambda person_id: hashlib.sha256(
                f"{run_id}\0{person_id}".encode()
            ).digest(),
        )
        if len(ordered) != 12:
            raise ValueError(f"run {run_id!r} must contain exactly 12 people, got {len(ordered)}")
        for index, person_id in enumerate(ordered):
            role: SplitRole
            if index < 8:
                role = "train"
            elif index < 10:
                role = "validation"
            else:
                role = "locked_test"
            result[(run_id, person_id)] = role
    return result


def assert_oracle_columns(columns: Sequence[str]) -> None:
    """Fail closed when known Goal 1 truth labels appear in an oracle feature frame."""

    offenders = sorted(
        column
        for column in columns
        if column in FORBIDDEN_ORACLE_EXACT
        or column.startswith("active_target_")
        or column.startswith("participant_truth_")
        or column.startswith("artifact_schedule_")
    )
    if offenders:
        raise ValueError(f"truth leakage columns are forbidden: {', '.join(offenders)}")


def _event_times(events: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if {"onset_utc", "end_utc"}.issubset(events.columns):
        onset = (
            pd.to_datetime(events["onset_utc"], utc=True)
            .dt.as_unit("ns")
            .astype("int64")
            .to_numpy()
        )
        end = (
            pd.to_datetime(events["end_utc"], utc=True).dt.as_unit("ns").astype("int64").to_numpy()
        )
    elif {"onset_time_ns", "end_time_ns"}.issubset(events.columns):
        onset = events["onset_time_ns"].dropna().astype("int64").to_numpy()
        valid = events["onset_time_ns"].notna()
        end = events.loc[valid, "end_time_ns"].astype("int64").to_numpy()
    else:
        raise ValueError("events require onset/end UTC or nanosecond columns")
    return onset, end


def build_labels(timeline: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Build labels from the independent event table, never from latent truth flags."""

    if "timestamp_utc" in timeline:
        time_ns = (
            pd.to_datetime(timeline["timestamp_utc"], utc=True)
            .dt.as_unit("ns")
            .astype("int64")
            .to_numpy()
        )
    elif "time_ns" in timeline:
        time_ns = timeline["time_ns"].astype("int64").to_numpy()
    else:
        raise ValueError("timeline requires timestamp_utc or time_ns")

    target_events = events
    if "is_target" in target_events:
        target_events = target_events.loc[target_events["is_target"].astype(bool)]
    onset_ns, end_ns = _event_times(target_events)
    event_binary = np.zeros(len(timeline), dtype=np.int8)
    forecast_60s = np.zeros(len(timeline), dtype=np.int8)
    phase = np.full(len(timeline), "baseline", dtype=object)
    one_second = 1_000_000_000
    for onset, end in zip(onset_ns, end_ns, strict=True):
        event_mask = (time_ns >= onset) & (time_ns <= end)
        forecast_mask = (time_ns >= onset - 60 * one_second) & (time_ns <= onset - one_second)
        event_binary[event_mask] = 1
        forecast_60s[forecast_mask] = 1
        phase[forecast_mask] = "forecast"
        phase[event_mask] = "event"
    return pd.DataFrame(
        {
            "event_binary": event_binary,
            "forecast_60s": forecast_60s,
            "phase": phase,
        },
        index=timeline.index,
    )
