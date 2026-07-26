from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from zoneinfo import ZoneInfo

import numpy as np
from numpy.typing import NDArray

from multisensor_synth.config.models import ProjectConfig
from multisensor_synth.domain.seeds import SeedTree
from multisensor_synth.domain.types import (
    DailyContextRecord,
    EventRecord,
    ParticipantRecord,
)
from multisensor_synth.events.archetypes import ARCHETYPE_MODALITY_STRENGTH
from multisensor_synth.latent.daily_context import sample_sleep_hours
from multisensor_synth.latent.factors import generate_slow_factor

NANOSECONDS = 1_000_000_000


@dataclass(frozen=True, slots=True)
class LatentTimeline:
    run_id: list[str]
    person_id: list[str]
    truth_time_ns: NDArray[np.int64]
    timestamp_utc: NDArray[np.datetime64]
    date_utc: list[date]
    day_index: NDArray[np.int8]
    seconds_from_start: NDArray[np.int64]
    context_state: list[str]
    is_awake: NDArray[np.bool_]
    autonomic_arousal: NDArray[np.float32]
    motor_activation: NDArray[np.float32]
    cognitive_load: NDArray[np.float32]
    sleep_pressure: NDArray[np.float32]
    sensory_context: NDArray[np.float32]
    recovery_capacity: NDArray[np.float32]
    social_context: NDArray[np.float32]
    active_target_event_id: list[str | None]
    active_target_event_phase: list[str | None]
    active_hard_negative_event_ids: list[list[str]]
    active_hard_negative_types: list[list[str]]
    artifact_only_schedule_ids: list[list[str]]

    def __len__(self) -> int:
        return len(self.truth_time_ns)

    def factor_arrays(self) -> tuple[NDArray[np.float32], ...]:
        return (
            self.autonomic_arousal,
            self.motor_activation,
            self.cognitive_load,
            self.sleep_pressure,
            self.sensory_context,
            self.recovery_capacity,
            self.social_context,
        )


def _phase_at(event: EventRecord, timestamp_ns: int) -> str:
    boundaries = (
        ("pre_early", event.pre_late_start_time_ns),
        ("pre_late", event.onset_time_ns),
        ("onset", event.peak_start_time_ns),
        ("peak", event.peak_end_time_ns),
        ("recovery_early", event.recovery_early_end_time_ns),
        ("recovery_late", event.recovery_late_end_time_ns),
        ("post", event.end_time_ns),
    )
    for name, end in boundaries:
        if end is not None and timestamp_ns < end:
            return name
    return "post"


def _context_states(
    config: ProjectConfig,
    seconds: NDArray[np.int64],
    contexts: list[DailyContextRecord],
) -> tuple[list[str], NDArray[np.bool_]]:
    start = config.run.start_time_utc
    zone = ZoneInfo(config.run.timezone_context)
    utc_offset = start.astimezone(zone).utcoffset()
    if utc_offset is None:
        raise ValueError("timezone_context did not provide a UTC offset")
    offset_seconds = int(utc_offset.total_seconds())
    start_seconds_of_day = start.hour * 3600 + start.minute * 60 + start.second
    local_hours = (
        (start_seconds_of_day + offset_seconds + seconds) % 86_400
    ) / 3600
    day_indices = np.minimum(seconds // 86_400, len(contexts) - 1)
    states: list[str] = []
    awake = np.ones(len(seconds), dtype=np.bool_)

    schedules = [sample_sleep_hours(config, row.seed) for row in contexts]
    for index, local_hour in enumerate(local_hours):
        bedtime, duration = schedules[int(day_indices[index])]
        wake_hour = (bedtime + duration) % 24
        sleeping = (
            local_hour >= bedtime or local_hour < wake_hour
            if bedtime > wake_hour
            else bedtime <= local_hour < wake_hour
        )
        if sleeping:
            state = "sleep"
            awake[index] = False
        elif 6.5 <= local_hour < 8:
            state = "transition"
        elif 8 <= local_hour < 9:
            state = "meal_context"
        elif 9 <= local_hour < 11.5 or 13 <= local_hour < 15:
            state = "focused_task"
        elif 11.5 <= local_hour < 13 or 18 <= local_hour < 19.5:
            state = "meal_context"
        elif 16 <= local_hour < 17:
            state = "moderate_activity"
        elif 17 <= local_hour < 18:
            state = "light_activity"
        elif local_hour < 9 or local_hour >= 21:
            state = "wake_rest"
        else:
            state = "sedentary_activity"
        states.append(state)
    return states, awake


def _event_slice(
    event: EventRecord, run_start_ns: int, length: int
) -> tuple[int, int]:
    start = max(0, (event.start_time_ns - run_start_ns) // NANOSECONDS)
    end = min(
        length,
        (event.end_time_ns - run_start_ns + NANOSECONDS - 1) // NANOSECONDS,
    )
    return int(start), int(end)


def generate_person_timeline(
    config: ProjectConfig,
    participant: ParticipantRecord,
    all_contexts: list[DailyContextRecord],
    all_events: list[EventRecord],
    seeds: SeedTree,
    run_id: str,
) -> LatentTimeline:
    length = config.run.duration_sec
    seconds = np.arange(length, dtype=np.int64)
    run_start_ns = int(config.run.start_time_utc.timestamp() * NANOSECONDS)
    truth_time_ns = run_start_ns + seconds * NANOSECONDS
    timestamp_utc = truth_time_ns.astype("datetime64[ns]")
    contexts = sorted(
        (row for row in all_contexts if row.person_id == participant.person_id),
        key=lambda row: row.day_index,
    )
    person_events = [
        row for row in all_events if row.person_id == participant.person_id
    ]
    context_state, is_awake = _context_states(config, seconds, contexts)
    day_index = np.minimum(seconds // 86_400, len(contexts) - 1).astype(np.int8)
    date_values = [contexts[int(index)].date_utc for index in day_index]
    drift = np.asarray(
        [contexts[int(index)].baseline_drift_truth for index in day_index],
        dtype=np.float32,
    )

    bases = {
        "autonomic_arousal": np.clip(
            0.32 + (participant.resting_hr_bpm_truth - 80) / 180, 0.15, 0.65
        ),
        "motor_activation": 0.24,
        "cognitive_load": 0.34,
        "sleep_pressure": 0.40,
        "sensory_context": 0.30,
        "recovery_capacity": participant.recovery_capacity_truth,
        "social_context": 0.30,
    }
    factor_values: dict[str, NDArray[np.float32]] = {}
    for factor_name, base in bases.items():
        factor_values[factor_name] = generate_slow_factor(
            participant.person_id,
            factor_name,
            seconds,
            float(base),
            seeds,
            config.context.circadian.amplitude_variation * 0.12,
        )
        factor_values[factor_name] = np.clip(
            factor_values[factor_name] + drift * 0.25, 0, 1
        ).astype(np.float32)

    active_target_ids: list[str | None] = [None] * length
    active_target_phases: list[str | None] = [None] * length
    active_negative_ids: list[list[str]] = [[] for _ in range(length)]
    active_negative_types: list[list[str]] = [[] for _ in range(length)]
    artifact_ids: list[list[str]] = [[] for _ in range(length)]
    phase_strength = {
        "pre_early": 0.20,
        "pre_late": 0.40,
        "onset": 0.70,
        "peak": 1.00,
        "recovery_early": 0.60,
        "recovery_late": 0.30,
        "post": 0.12,
    }
    autonomic_scale, motor_scale, cognitive_scale = ARCHETYPE_MODALITY_STRENGTH[
        participant.response_archetype
    ]

    for event in person_events:
        start, end = _event_slice(event, run_start_ns, length)
        if event.is_artifact_only_schedule:
            for index in range(start, end):
                artifact_ids[index].append(event.event_id)
            continue
        if not event.is_target:
            for index in range(start, end):
                active_negative_ids[index].append(event.event_id)
                active_negative_types[index].append(event.event_type)
            if event.event_type == "ordinary_physical_activity":
                factor_values["motor_activation"][start:end] += 0.35
                factor_values["autonomic_arousal"][start:end] += 0.18
            elif event.event_type == "quiet_cognitive_load":
                factor_values["cognitive_load"][start:end] += 0.25
                factor_values["autonomic_arousal"][start:end] += 0.06
            elif event.event_type == "recovery_without_peak":
                factor_values["autonomic_arousal"][start:end] += 0.12
            elif event.event_type == "false_alarm_like_episode":
                factor_values["sensory_context"][start:end] += 0.18
            continue

        response_rng = np.random.default_rng(event.seed)
        response = 1.0
        draw = float(response_rng.random())
        if draw < participant.no_response_tendency_truth:
            response = 0.05
        elif draw < (
            participant.no_response_tendency_truth
            + participant.opposite_response_tendency_truth
        ):
            response = -0.20
        for index in range(start, end):
            phase = _phase_at(event, int(truth_time_ns[index]))
            active_target_ids[index] = event.event_id
            active_target_phases[index] = phase
            strength = phase_strength[phase] * event.intensity_truth * response
            factor_values["autonomic_arousal"][index] += strength * 0.34 * autonomic_scale
            factor_values["motor_activation"][index] += strength * 0.30 * motor_scale
            factor_values["cognitive_load"][index] += strength * 0.26 * cognitive_scale
            factor_values["sensory_context"][index] += strength * 0.18

    for name in factor_values:
        factor_values[name] = np.clip(factor_values[name], 0, 1).astype(np.float32)

    return LatentTimeline(
        run_id=[run_id] * length,
        person_id=[participant.person_id] * length,
        truth_time_ns=truth_time_ns,
        timestamp_utc=timestamp_utc,
        date_utc=date_values,
        day_index=day_index,
        seconds_from_start=seconds,
        context_state=context_state,
        is_awake=is_awake,
        autonomic_arousal=factor_values["autonomic_arousal"],
        motor_activation=factor_values["motor_activation"],
        cognitive_load=factor_values["cognitive_load"],
        sleep_pressure=factor_values["sleep_pressure"],
        sensory_context=factor_values["sensory_context"],
        recovery_capacity=factor_values["recovery_capacity"],
        social_context=factor_values["social_context"],
        active_target_event_id=active_target_ids,
        active_target_event_phase=active_target_phases,
        active_hard_negative_event_ids=active_negative_ids,
        active_hard_negative_types=active_negative_types,
        artifact_only_schedule_ids=artifact_ids,
    )
