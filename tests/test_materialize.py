from __future__ import annotations

from pathlib import Path

from multisensor_synth.config.loader import load_config

from multisensor_ml.materialize import write_derived_generator_config


def test_derived_generator_config_changes_only_seed_and_explicit_overrides(
    tmp_path: Path,
) -> None:
    base = Path(__file__).parents[2] / "multisensor_synth" / "configs" / "quick.yaml"
    destination = tmp_path / "seed.yaml"

    write_derived_generator_config(
        base,
        destination,
        seed=20260727,
        overrides={"run": {"participant_count": 12}},
    )
    loaded = load_config(destination)

    assert loaded.config.run.seed == 20260727
    assert loaded.config.run.participant_count == 12
    assert loaded.config.run.duration_sec == 21600
    assert loaded.config.events.target_events_per_day.min == 2
    assert loaded.config.profile == "quick"
