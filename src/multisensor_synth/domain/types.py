from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class ParticipantRecord:
    run_id: str
    person_id: str
    age_years: float
    biological_sex: str
    response_archetype: str
    resting_hr_bpm_truth: float
    rmssd_ms_truth: float
    eeg_aperiodic_exponent_truth: float
    eeg_relative_alpha_truth: float
    eeg_relative_gamma_truth: float
    eda_tonic_level_us_truth: float
    recovery_capacity_truth: float
    opposite_response_tendency_truth: float
    no_response_tendency_truth: float
    seed: int


@dataclass(frozen=True, slots=True)
class DailyContextRecord:
    run_id: str
    person_id: str
    day_index: int
    date_utc: date
    sleep_start_time_ns: int | None
    sleep_end_time_ns: int | None
    sleep_duration_sec: int
    fatigue_drift_truth: float
    activity_drift_truth: float
    baseline_drift_truth: float
    sensor_tolerance_truth: float
    seed: int


@dataclass(frozen=True, slots=True)
class EventRecord:
    run_id: str
    person_id: str
    event_id: str
    event_type: str
    is_target: bool
    hard_negative_kind: str | None
    start_time_ns: int
    pre_late_start_time_ns: int | None
    onset_time_ns: int | None
    peak_start_time_ns: int | None
    peak_end_time_ns: int | None
    recovery_early_end_time_ns: int | None
    recovery_late_end_time_ns: int | None
    end_time_ns: int
    intensity_truth: float
    archetype: str
    is_artifact_only_schedule: bool
    seed: int
