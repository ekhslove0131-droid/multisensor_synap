from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from multisensor_ml.beta_training.artifacts import wire_refs
from multisensor_ml.beta_training.pipeline import (
    BetaTrainConfig,
    bundle_model_role,
    expected_fit_attempts,
)
from multisensor_ml.beta_training.stage_policy import causal_stages, select_stage_policy


def test_frozen_candidate_matrix_is_28_fit_attempts() -> None:
    config = BetaTrainConfig.load(Path("configs/beta_loop_v1.yaml"))
    assert config.thresholds == tuple(index / 20 for index in range(1, 20))
    assert expected_fit_attempts(config) == 28
    assert config.source_sets == (
        ("watch",), ("h10",), ("muse",), ("watch", "h10"),
        ("watch", "muse"), ("h10", "muse"), ("watch", "h10", "muse"),
    )
    assert len(config.sha256()) == 64


def test_partial_head_set_cannot_claim_selected_model_role() -> None:
    assert bundle_model_role("NO_QUALIFIED_BETA_MODEL", 13) is None
    assert bundle_model_role("BETA_SELECTED", 14) == "BETA_SELECTED"


def test_wire_contract_requires_current_eight_schema_and_golden_files(
    tmp_path: Path,
) -> None:
    names = (
        "ack", "admin_binding", "native", "observer", "phone_binding",
        "projection", "result", "run",
    )
    for folder in ("schemas/synthetic_beta_v1",
                   "services/inference/tests/synthetic_beta/golden"):
        root = tmp_path / folder
        root.mkdir(parents=True)
        for name in names:
            (root / f"{name}.json").write_text("{}")
    refs = wire_refs(tmp_path)
    assert len(refs) == 16
    assert {(ref["kind"], ref["name"]) for ref in refs} == {
        (kind, f"{name}.json")
        for kind in ("synthetic_beta_v1", "golden") for name in names
    }


def test_wire_contract_rejects_pre_admin_fourteen_file_shape(tmp_path: Path) -> None:
    names = ("ack", "native", "observer", "phone_binding", "projection", "result", "run")
    for folder in ("schemas/synthetic_beta_v1",
                   "services/inference/tests/synthetic_beta/golden"):
        root = tmp_path / folder
        root.mkdir(parents=True)
        for name in names:
            (root / f"{name}.json").write_text("{}")
    with pytest.raises(ValueError, match="beta_wire_contract_incomplete"):
        wire_refs(tmp_path)


def test_unknown_source_or_head_is_rejected() -> None:
    with pytest.raises(ValueError, match="beta_train_config_invalid"):
        BetaTrainConfig(source_sets=(("watch", "unknown"),))

    with pytest.raises(ValueError, match="beta_train_config_invalid"):
        BetaTrainConfig(current_sensitivity_gate=0.60)


def test_stage_policy_uses_temporal_phase_not_binary_proxy() -> None:
    deviations = np.array([0.2, 0.4, 1.2, 1.3, 1.31, 1.3, 0.9, 0.6,
                           0.4, 0.3, 0.2, 0.2], dtype=float)
    current = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    onset = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    phases = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0, 0], dtype=np.int64)
    value = select_stage_policy(
        deviations, current, onset, phases,
        ("subject",) * len(deviations), np.arange(len(deviations)) * 10_000,
    )
    assert value["d_on"] in (1.0, 1.5, 2.0)
    assert value["d_off"] == value["d_on"] / 2
    assert value["selection_role"] == "VALIDATION"
    assert value["selection_metric"] == (
        "stage_reversals_then_reference_temporal_phase_macro_f1_then_d_on"
    )
    assert "validation_phase_macro_f1" in value
    assert "validation_stage_reversals" in value


def test_beta_cli_configuration_does_not_eagerly_import_export_stack() -> None:
    script = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if (name == 'onnx' or name.startswith('onnx.')
            or name == 'skl2onnx' or name.startswith('skl2onnx.')):
        raise ImportError('blocked optional export stack')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from multisensor_ml.beta_training.cli import configure_beta_parser
"""
    result = subprocess.run(
        [sys.executable, "-c", script], text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_completed_episode_does_not_leak_peak_into_next_episode() -> None:
    first = np.array([0.2, 0.3, 2.0, 2.1, 2.1, 2.1, 1.5, 1.0,
                      0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
    second = np.array([0.2, 0.3, 1.1, 1.2, 1.3, 1.4, 1.4, 1.4, 0.7,
                       0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
    def inputs(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        onset = np.zeros(len(values), dtype=bool)
        onset[:2] = True
        current = np.zeros(len(values), dtype=bool)
        current[2:10] = True
        return current, onset

    first_current, first_onset = inputs(first)
    second_current, second_onset = inputs(second)
    combined = np.concatenate((first, second))
    combined_stage, _ = causal_stages(
        combined, np.concatenate((first_current, second_current)),
        np.concatenate((first_onset, second_onset)),
        ("subject",) * len(combined), np.arange(len(combined)) * 10_000, 1.0,
    )
    isolated_stage, _ = causal_stages(
        second, second_current, second_onset, ("subject",) * len(second),
        np.arange(len(second)) * 10_000, 1.0,
    )
    assert np.array_equal(combined_stage[len(first):], isolated_stage)
