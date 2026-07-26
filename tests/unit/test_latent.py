from __future__ import annotations

from pathlib import Path

import numpy as np

from multisensor_synth.config.loader import load_config
from multisensor_synth.domain.contracts import CONTEXT_STATES
from multisensor_synth.domain.seeds import SeedTree
from multisensor_synth.events.scheduler import schedule_events
from multisensor_synth.latent.daily_context import generate_daily_contexts
from multisensor_synth.latent.state_process import generate_person_timeline
from multisensor_synth.population.generator import generate_participants

ROOT = Path(__file__).parents[2]


def _inputs() -> tuple[object, object, object, object]:
    config = load_config(ROOT / "configs" / "quick.yaml").config
    seeds = SeedTree(config.run.seed)
    participants = generate_participants(config, seeds, "run-test")
    contexts = generate_daily_contexts(config, participants, seeds, "run-test")
    events = schedule_events(config, participants, seeds, "run-test")
    return config, seeds, participants, (contexts, events)


def test_quick_daily_context_has_one_row_per_person() -> None:
    config, _, participants, context_and_events = _inputs()
    contexts, _ = context_and_events

    assert len(contexts) == config.run.participant_count
    assert {row.person_id for row in contexts} == {row.person_id for row in participants}
    assert all(row.day_index == 0 for row in contexts)
    assert all(0 <= row.sensor_tolerance_truth <= 1 for row in contexts)


def test_person_timeline_is_one_hz_bounded_and_non_flat() -> None:
    config, seeds, participants, context_and_events = _inputs()
    contexts, events = context_and_events
    person = participants[0]
    timeline = generate_person_timeline(config, person, contexts, events, seeds, "run-test")

    assert len(timeline) == 21_600
    assert np.all(np.diff(timeline.truth_time_ns) == 1_000_000_000)
    assert timeline.truth_time_ns[0] == int(config.run.start_time_utc.timestamp() * 1e9)
    assert set(timeline.context_state).issubset(CONTEXT_STATES)
    for factor in timeline.factor_arrays():
        assert factor.dtype == np.float32
        assert np.all(factor >= 0)
        assert np.all(factor <= 1)
        assert np.std(factor) > 0.001


def test_target_and_negative_axes_are_kept_separate() -> None:
    config, seeds, participants, context_and_events = _inputs()
    contexts, events = context_and_events
    timelines = [
        generate_person_timeline(config, person, contexts, events, seeds, "run-test")
        for person in participants
    ]
    timeline = timelines[0]

    assert any(value is not None for value in timeline.active_target_event_id)
    assert "peak" in timeline.active_target_event_phase
    assert any(values for values in timeline.active_hard_negative_event_ids)
    assert any(
        values
        for person_timeline in timelines
        for values in person_timeline.artifact_only_schedule_ids
    )
    for person_timeline in timelines:
        for negative_ids, artifact_ids in zip(
            person_timeline.active_hard_negative_event_ids,
            person_timeline.artifact_only_schedule_ids,
            strict=True,
        ):
            assert set(artifact_ids).isdisjoint(negative_ids)


def test_latent_generation_is_repeatable() -> None:
    config, _, participants, context_and_events = _inputs()
    contexts, events = context_and_events
    person = participants[0]
    first = generate_person_timeline(
        config, person, contexts, events, SeedTree(config.run.seed), "run-test"
    )
    second = generate_person_timeline(
        config, person, contexts, events, SeedTree(config.run.seed), "run-test"
    )

    assert np.array_equal(first.autonomic_arousal, second.autonomic_arousal)
    assert np.array_equal(first.motor_activation, second.motor_activation)
    assert first.active_target_event_phase == second.active_target_event_phase


def test_latent_content_is_independent_of_chunk_size() -> None:
    config, _, participants, context_and_events = _inputs()
    contexts, events = context_and_events
    person = participants[0]
    changed_run = config.run.model_copy(update={"chunk_duration_sec": 777})
    changed_config = config.model_copy(update={"run": changed_run})

    default_chunk = generate_person_timeline(
        config,
        person,
        contexts,
        events,
        SeedTree(config.run.seed),
        "run-test",
    )
    changed_chunk = generate_person_timeline(
        changed_config,
        person,
        contexts,
        events,
        SeedTree(config.run.seed),
        "run-test",
    )

    assert np.array_equal(
        default_chunk.autonomic_arousal,
        changed_chunk.autonomic_arousal,
    )
    assert default_chunk.active_target_event_id == changed_chunk.active_target_event_id
