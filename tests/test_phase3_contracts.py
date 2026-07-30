from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from multisensor_ml.phase3_contracts import (
    PHASE3_CANDIDATE_IDS,
    Phase3Config,
)
from multisensor_ml.settings import load_phase3_config


def valid_payload() -> dict[str, object]:
    return {
        "schema_version": "goal1.5/phase3-personal-pattern/v1",
        "series_id": "mvp3-oracle-v1",
        "split_counts": {"train": 24, "validation": 6, "locked_test": 6},
        "warmup_sec": 1800,
        "lookback_sec": 21600,
        "refresh_sec": 60,
        "weight_caps": [0.0, 0.25, 0.5, 1.0],
        "cumulative_horizons_sec": [1800, 21600, 86400, 259200],
        "baseline_load_influence_cap": 0.10,
        "run_locked_test": False,
        "raw_root": "../data/raw",
        "outcome_root": "../data/outcomes",
        "registry_root": "../data/registry",
        "artifact_root": "../artifacts/phase3",
    }


def test_phase3_contract_has_exact_candidates_and_safety_values() -> None:
    config = Phase3Config.model_validate(valid_payload())

    assert PHASE3_CANDIDATE_IDS == (
        "G0",
        "P1-025",
        "P1-050",
        "P1-100",
        "P2-025",
        "P2-050",
        "P2-100",
        "P2+CL",
        "P2+CL-B",
    )
    assert config.weight_caps == (0.0, 0.25, 0.5, 1.0)
    assert config.cumulative_horizons_sec == (1800, 21600, 86400, 259200)
    assert config.baseline_load_influence_cap == 0.10
    assert config.run_locked_test is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("split_counts", {"train": 23, "validation": 7, "locked_test": 6}),
        ("warmup_sec", 1799),
        ("lookback_sec", 1800),
        ("refresh_sec", 30),
        ("weight_caps", [0.0, 0.5, 1.0]),
        ("cumulative_horizons_sec", [1800, 21600, 86400]),
        ("baseline_load_influence_cap", 0.11),
        ("run_locked_test", True),
    ],
)
def test_phase3_contract_rejects_changed_fixed_values(
    field: str,
    value: object,
) -> None:
    payload = valid_payload()
    payload[field] = value

    with pytest.raises(ValidationError):
        Phase3Config.model_validate(payload)


def test_phase3_contract_rejects_unknown_fields() -> None:
    payload = valid_payload()
    payload["future_option"] = True

    with pytest.raises(ValidationError):
        Phase3Config.model_validate(payload)


def test_load_phase3_config_resolves_project_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "phase3.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        yaml.safe_dump(valid_payload(), sort_keys=True),
        encoding="utf-8",
    )

    loaded = load_phase3_config(config_path)

    assert loaded.raw_root == (config_path.parent / "../data/raw").resolve()
    assert loaded.outcome_root == (config_path.parent / "../data/outcomes").resolve()
    assert loaded.registry_root == (config_path.parent / "../data/registry").resolve()
    assert loaded.artifact_root == (config_path.parent / "../artifacts/phase3").resolve()
