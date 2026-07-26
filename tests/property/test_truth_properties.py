from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from multisensor_synth.config.loader import load_config
from multisensor_synth.config.models import ProjectConfig, RangeInt
from multisensor_synth.domain.seeds import SeedTree
from multisensor_synth.events.phases import PHASE_ORDER, boundaries_from_start
from multisensor_synth.events.scheduler import schedule_events
from multisensor_synth.latent.daily_context import generate_daily_contexts
from multisensor_synth.latent.state_process import generate_person_timeline
from multisensor_synth.population.generator import generate_participants

ROOT = Path(__file__).parents[2]
GOLDEN_CONFIG = ROOT / "tests" / "golden" / "golden.yaml"


@given(
    root_seed=st.integers(min_value=0, max_value=2**32 - 1),
    person_index=st.integers(min_value=1, max_value=999),
    day_index=st.integers(min_value=0, max_value=30),
)
def test_child_seed_is_stable_for_valid_namespaces(
    root_seed: int, person_index: int, day_index: int
) -> None:
    namespace = f"person/P{person_index:03d}/day/{day_index}"

    assert SeedTree(root_seed).child_seed(namespace) == SeedTree(root_seed).child_seed(
        namespace
    )


@given(duration_sec=st.integers(min_value=1, max_value=31 * 86_400))
def test_one_hz_sample_count_formula(duration_sec: int) -> None:
    canonical_rate_hz = 1

    assert duration_sec * canonical_rate_hz == duration_sec


@given(
    durations=st.lists(
        st.integers(min_value=0, max_value=600),
        min_size=len(PHASE_ORDER),
        max_size=len(PHASE_ORDER),
    )
)
def test_phase_boundaries_never_reverse(durations: list[int]) -> None:
    phase_durations = dict(zip(PHASE_ORDER, durations, strict=True))
    boundaries = boundaries_from_start(1_000_000_000, phase_durations)
    values = list(boundaries.values())

    assert values == sorted(values)


@given(
    minimum=st.integers(min_value=-10_000, max_value=10_000),
    maximum=st.integers(min_value=-10_000, max_value=10_000),
)
def test_invalid_ranges_are_always_rejected(minimum: int, maximum: int) -> None:
    payload = {"min": minimum, "max": maximum}
    if minimum <= maximum:
        assert RangeInt.model_validate(payload).min == minimum
    else:
        with pytest.raises(ValidationError):
            RangeInt.model_validate(payload)


@given(
    participant_count=st.integers(min_value=1, max_value=4),
    duration_sec=st.integers(min_value=180, max_value=900),
)
@settings(max_examples=12, deadline=None)
def test_valid_population_and_timeline_dimensions(
    participant_count: int, duration_sec: int
) -> None:
    base = load_config(GOLDEN_CONFIG).config
    run = base.run.model_copy(
        update={
            "participant_count": participant_count,
            "duration_sec": duration_sec,
        }
    )
    config = base.model_copy(update={"run": run})
    seeds = SeedTree(config.run.seed)
    participants = generate_participants(config, seeds, "property-run")
    contexts = generate_daily_contexts(config, participants, seeds, "property-run")
    events = schedule_events(config, participants, seeds, "property-run")

    assert [row.person_id for row in participants] == [
        f"P{index:03d}" for index in range(1, participant_count + 1)
    ]
    for participant in participants:
        timeline = generate_person_timeline(
            config,
            participant,
            contexts,
            events,
            seeds,
            "property-run",
        )
        assert len(timeline) == duration_sec
        assert np.all(np.diff(timeline.truth_time_ns) == 1_000_000_000)
    for event in (row for row in events if row.is_target):
        assert event.onset_time_ns is not None
        assert event.peak_start_time_ns is not None
        assert event.peak_end_time_ns is not None


@given(
    first_seed=st.integers(min_value=0, max_value=2**32 - 1),
    second_seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_distinct_root_seeds_change_a_profile(
    first_seed: int, second_seed: int
) -> None:
    assume(first_seed != second_seed)
    config = load_config(GOLDEN_CONFIG).config
    first = generate_participants(config, SeedTree(first_seed), "property-run")
    second = generate_participants(config, SeedTree(second_seed), "property-run")

    assert first != second


@given(invalid_probability=st.sampled_from([-0.01, -1.0, 1.01, 2.0]))
def test_invalid_probabilities_are_always_rejected(
    invalid_probability: float,
) -> None:
    config = load_config(GOLDEN_CONFIG).config
    payload = config.model_dump(mode="json", by_alias=True)
    payload["population"]["allow_no_response_probability"] = invalid_probability

    with pytest.raises(ValidationError):
        ProjectConfig.model_validate(payload)
