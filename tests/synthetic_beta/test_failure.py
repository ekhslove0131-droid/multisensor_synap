from __future__ import annotations

import json
from pathlib import Path

from multisensor_ml.beta_training.failure import finalize_no_qualified


def test_failure_handoff_cannot_claim_selected_runtime(tmp_path: Path) -> None:
    round_manifest = tmp_path / "round.json"
    evaluation = tmp_path / "evaluation.json"
    round_manifest.write_text(json.dumps({
        "realm": "SYNTHETIC_BETA", "status": "NO_QUALIFIED_BETA_MODEL",
        "m2_fit_attempt_count": 28, "pattern_models": [{"head": "current"}],
        "selection_failures": [{"source_set": "h10", "head": "current",
                                "reason": "NO_QUALIFIED_BETA_MODEL"}],
        "dataset_manifest_sha256": "a" * 64,
        "training_config_sha256": "d" * 64,
        "generation_protocol_sha256": "b" * 64,
        "runtime_wheel_ref": {"uri": (tmp_path / "runtime.whl").as_uri(),
                              "sha256": "c" * 64},
        "wire_contract_refs": [],
    }))
    evaluation.write_text(json.dumps({
        "status": "NO_QUALIFIED_BETA_MODEL",
        "locked_use": {"status": "NOT_ACCESSED", "use_count": 0},
    }))
    runtime = tmp_path / "runtime.whl"
    runtime.write_bytes(b"runtime")
    result = finalize_no_qualified(
        round_manifest, evaluation, tmp_path / "final", corrected_fit_rounds=2,
        diagnostic_fit_attempts=2, diagnostic_contract_invalid_fit_attempts=56,
        corrected_runtime_wheel=runtime,
    )
    assert result["model_role"] is None
    assert result["delivery_eligible"] is False
    assert result["locked_test_use_count"] == 0
    assert result["total_recorded_m2_fit_attempt_count"] == 114
    assert result["runtime_wheel_ref"]["sha256"] != "c" * 64
