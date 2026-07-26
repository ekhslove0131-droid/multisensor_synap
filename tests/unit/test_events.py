from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from multisensor_synth.config.loader import load_config
from multisensor_synth.domain.seeds import SeedTree
from multisensor_synth.events.phases import PHASE_ORDER, phase_curve
from multisensor_synth.events.scheduler import schedule_events
from multisensor_synth.population.generator import generate_participants

ROOT = Path(__file__).parents[2]


def _quick_events() -> tuple[object, list[object]]:
    config = load_config(ROOT / "configs" / "quick.yaml").config
    seeds = SeedTree(config.run.seed)
    participants = generate_participants(config, seeds, "run-test")
    return config, schedule_events(config, participants, seeds, "run-test")


def test_quick_has_two_targets_per_participant() -> None:
    _, events = _quick_events()
    counts = Counter(event.person_id for event in events if event.is_target)

    assert counts == {"P001": 2, "P002": 2, "P003": 2}


def test_event_ids_and_required_hard_negatives_are_present() -> None:
    _, events = _quick_events()

    assert len({event.event_id for event in events}) == len(events)
    assert {
        "ordinary_physical_activity",
        "quiet_cognitive_load",
        "sensor_artifact_episode",
        "recovery_without_peak",
        "false_alarm_like_episode",
    }.issubset({event.event_type for event in events})
    artifact_events = [event for event in events if event.event_type == "sensor_artifact_episode"]
    assert artifact_events
    assert all(event.is_artifact_only_schedule for event in artifact_events)


def test_target_phase_boundaries_are_ordered() -> None:
    _, events = _quick_events()

    for event in (item for item in events if item.is_target):
        boundaries = [
            event.start_time_ns,
            event.pre_late_start_time_ns,
            event.onset_time_ns,
            event.peak_start_time_ns,
            event.peak_end_time_ns,
            event.recovery_early_end_time_ns,
            event.recovery_late_end_time_ns,
            event.end_time_ns,
        ]
        assert all(value is not None for value in boundaries)
        assert boundaries == sorted(boundaries)
        assert event.intensity_truth >= 0
        assert event.intensity_truth <= 1


def test_target_peaks_do_not_overlap_and_respect_gap() -> None:
    config, events = _quick_events()
    for person_id in ("P001", "P002", "P003"):
        targets = sorted(
            (event for event in events if event.person_id == person_id and event.is_target),
            key=lambda event: event.onset_time_ns,
        )
        assert targets[1].onset_time_ns - targets[0].onset_time_ns >= (
            config.events.minimum_gap_between_target_sec * 1_000_000_000
        )
        assert targets[0].peak_end_time_ns <= targets[1].peak_start_time_ns


def test_phase_curves_are_bounded_and_have_named_order() -> None:
    x = np.linspace(0, 1, 101)

    assert PHASE_ORDER == (
        "pre_early",
        "pre_late",
        "onset",
        "peak",
        "recovery_early",
        "recovery_late",
        "post",
    )
    for name in ("smoothstep", "exponential_rise", "sigmoid", "plateau", "decay"):
        values = phase_curve(name, x)
        assert np.all(values >= 0)
        assert np.all(values <= 1)
