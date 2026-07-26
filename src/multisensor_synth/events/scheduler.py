from __future__ import annotations

import math
from collections import defaultdict
from itertools import pairwise

import numpy as np

from multisensor_synth.config.models import ProjectConfig
from multisensor_synth.domain.seeds import SeedTree
from multisensor_synth.domain.types import EventRecord, ParticipantRecord
from multisensor_synth.events.hard_negatives import allocate_hard_negative_types
from multisensor_synth.events.phases import boundaries_from_start, sample_phase_durations

NANOSECONDS = 1_000_000_000
DAY_SECONDS = 86_400


def _start_time_ns(config: ProjectConfig) -> int:
    return int(config.run.start_time_utc.timestamp() * NANOSECONDS)


def _target_count(config: ProjectConfig, rng: np.random.Generator) -> int:
    bounds = config.events.target_events_per_day
    return int(rng.integers(bounds.min, bounds.max + 1))


def schedule_events(
    config: ProjectConfig,
    participants: list[ParticipantRecord],
    seeds: SeedTree,
    run_id: str,
) -> list[EventRecord]:
    run_start_ns = _start_time_ns(config)
    run_duration = config.run.duration_sec
    day_count = math.ceil(run_duration / DAY_SECONDS)
    targets: list[EventRecord] = []
    per_person_target_count: defaultdict[str, int] = defaultdict(int)

    for participant in participants:
        person_targets: list[EventRecord] = []
        target_sequence = 0
        for day_index in range(day_count):
            day_start = day_index * DAY_SECONDS
            day_end = min(run_duration, (day_index + 1) * DAY_SECONDS)
            day_rng = seeds.rng(f"person/{participant.person_id}/day/{day_index}")
            count = _target_count(config, day_rng)
            slot_width = (day_end - day_start) / count

            for slot_index in range(count):
                target_sequence += 1
                event_id = f"{participant.person_id}-T{target_sequence:03d}"
                namespace = f"person/{participant.person_id}/event/{event_id}"
                seed = seeds.child_seed(namespace)
                rng = np.random.default_rng(seed)
                durations = sample_phase_durations(config.events.phases, rng)
                total_duration = sum(durations.values())
                slot_start = int(day_start + slot_index * slot_width)
                slot_end = int(day_start + (slot_index + 1) * slot_width)
                latest_start = max(slot_start, slot_end - total_duration)
                if latest_start < slot_start:
                    raise ValueError("target phase durations do not fit configured run")
                start_second = (
                    slot_start
                    if latest_start == slot_start
                    else int(rng.integers(slot_start, latest_start + 1))
                )
                boundary = boundaries_from_start(
                    run_start_ns + start_second * NANOSECONDS,
                    durations,
                )
                if boundary["end_time_ns"] > run_start_ns + run_duration * NANOSECONDS:
                    raise ValueError("target event exceeds run boundary")
                event = EventRecord(
                    run_id=run_id,
                    person_id=participant.person_id,
                    event_id=event_id,
                    event_type=config.events.target_type,
                    is_target=True,
                    hard_negative_kind=None,
                    start_time_ns=boundary["start_time_ns"],
                    pre_late_start_time_ns=boundary["pre_late_start_time_ns"],
                    onset_time_ns=boundary["onset_time_ns"],
                    peak_start_time_ns=boundary["peak_start_time_ns"],
                    peak_end_time_ns=boundary["peak_end_time_ns"],
                    recovery_early_end_time_ns=boundary["recovery_early_end_time_ns"],
                    recovery_late_end_time_ns=boundary["recovery_late_end_time_ns"],
                    end_time_ns=boundary["end_time_ns"],
                    intensity_truth=float(
                        np.float32(
                            rng.beta(
                                config.events.intensity.alpha,
                                config.events.intensity.beta,
                            )
                        )
                    ),
                    archetype=participant.response_archetype,
                    is_artifact_only_schedule=False,
                    seed=seed,
                )
                person_targets.append(event)

        person_targets.sort(key=lambda item: item.onset_time_ns or 0)
        for previous, current in pairwise(person_targets):
            if previous.onset_time_ns is None or current.onset_time_ns is None:
                raise AssertionError("target onset must be populated")
            gap = current.onset_time_ns - previous.onset_time_ns
            if gap < config.events.minimum_gap_between_target_sec * NANOSECONDS:
                raise ValueError("configured target events cannot satisfy minimum gap")
            if (
                previous.peak_end_time_ns is not None
                and current.peak_start_time_ns is not None
                and previous.peak_end_time_ns > current.peak_start_time_ns
            ):
                raise ValueError("target peak intervals overlap")
        targets.extend(person_targets)
        per_person_target_count[participant.person_id] = len(person_targets)

    hard_negative_total = round(
        len(targets) * config.events.hard_negatives.ratio_to_target
    )
    type_rng = seeds.rng("hard_negative/types")
    hard_negative_types = allocate_hard_negative_types(
        hard_negative_total,
        config.events.hard_negatives.types.model_dump(),
        type_rng,
    )
    hard_negatives: list[EventRecord] = []
    type_index = 0
    participant_lookup = {row.person_id: row for row in participants}
    for person_id, count in per_person_target_count.items():
        participant = participant_lookup[person_id]
        for sequence in range(1, count + 1):
            if type_index >= len(hard_negative_types):
                break
            event_type = hard_negative_types[type_index]
            type_index += 1
            event_id = f"{person_id}-H{sequence:03d}"
            namespace = (
                f"person/{person_id}/artifact_only/{event_id}"
                if event_type == "sensor_artifact_episode"
                else f"person/{person_id}/hard_negative/{event_id}"
            )
            seed = seeds.child_seed(namespace)
            rng = np.random.default_rng(seed)
            duration = min(int(rng.integers(60, 301)), run_duration)
            latest_start = max(0, run_duration - duration)
            start_second = (
                0 if latest_start == 0 else int(rng.integers(0, latest_start + 1))
            )
            start_ns = run_start_ns + start_second * NANOSECONDS
            hard_negatives.append(
                EventRecord(
                    run_id=run_id,
                    person_id=person_id,
                    event_id=event_id,
                    event_type=event_type,
                    is_target=False,
                    hard_negative_kind=event_type,
                    start_time_ns=start_ns,
                    pre_late_start_time_ns=None,
                    onset_time_ns=None,
                    peak_start_time_ns=None,
                    peak_end_time_ns=None,
                    recovery_early_end_time_ns=None,
                    recovery_late_end_time_ns=None,
                    end_time_ns=start_ns + duration * NANOSECONDS,
                    intensity_truth=float(
                        np.float32(
                            rng.beta(
                                config.events.intensity.alpha,
                                config.events.intensity.beta,
                            )
                        )
                    ),
                    archetype=participant.response_archetype,
                    is_artifact_only_schedule=event_type
                    == "sensor_artifact_episode",
                    seed=seed,
                )
            )

    return sorted(targets + hard_negatives, key=lambda item: (item.person_id, item.start_time_ns))
