from __future__ import annotations

import importlib.util
from pathlib import Path

from multisensor_synth.config.loader import load_config
from multisensor_synth.domain.contracts import FORBIDDEN_MEDICAL_EVENT_TERMS
from multisensor_synth.domain.seeds import SeedTree
from multisensor_synth.events.scheduler import schedule_events
from multisensor_synth.population.generator import generate_participants

ROOT = Path(__file__).parents[2]


def test_goal_one_has_no_feature_or_device_runtime_packages() -> None:
    assert importlib.util.find_spec("multisensor_synth.features") is None
    assert importlib.util.find_spec("multisensor_synth.devices") is None
    assert importlib.util.find_spec("multisensor_synth.signals") is None


def test_event_types_use_only_neutral_terms() -> None:
    config = load_config(ROOT / "configs" / "quick.yaml").config
    seeds = SeedTree(config.run.seed)
    participants = generate_participants(config, seeds, "run-test")
    events = schedule_events(config, participants, seeds, "run-test")

    assert all(
        forbidden not in event.event_type.lower()
        for event in events
        for forbidden in FORBIDDEN_MEDICAL_EVENT_TERMS
    )


def test_repository_contains_no_sdk_binary_or_credential_file() -> None:
    forbidden_suffixes = {".aar", ".jar", ".pem", ".key", ".p12", ".env"}
    forbidden = [
        path.relative_to(ROOT)
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and ".bootstrap-uv" not in path.parts
        and path.suffix.lower() in forbidden_suffixes
    ]

    assert forbidden == []
