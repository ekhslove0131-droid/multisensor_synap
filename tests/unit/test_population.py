from __future__ import annotations

from pathlib import Path

from multisensor_synth.config.loader import load_config
from multisensor_synth.domain.seeds import SeedTree
from multisensor_synth.population.generator import generate_participants

ROOT = Path(__file__).parents[2]


def test_quick_population_has_valid_profiles() -> None:
    config = load_config(ROOT / "configs" / "quick.yaml").config
    participants = generate_participants(config, SeedTree(config.run.seed), "run-test")

    assert [row.person_id for row in participants] == ["P001", "P002", "P003"]
    assert all(6 <= row.age_years <= 17 for row in participants)
    assert all(row.rmssd_ms_truth > 0 for row in participants)
    assert all(0 <= row.eeg_relative_alpha_truth <= 1 for row in participants)
    assert all(0 <= row.eeg_relative_gamma_truth <= 1 for row in participants)
    assert all(0 <= row.recovery_capacity_truth <= 1 for row in participants)


def test_population_is_repeatable_and_seed_sensitive() -> None:
    config = load_config(ROOT / "configs" / "quick.yaml").config
    first = generate_participants(config, SeedTree(11), "run-test")
    repeated = generate_participants(config, SeedTree(11), "run-test")
    changed = generate_participants(config, SeedTree(12), "run-test")

    assert first == repeated
    assert first != changed
