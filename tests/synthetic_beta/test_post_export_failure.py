from __future__ import annotations

import json
from pathlib import Path

from multisensor_ml.beta_training.post_export_failure import (
    finalize_post_export_failure,
)


def test_partial_round_can_only_be_finalized_as_no_qualified(tmp_path: Path) -> None:
    previous = tmp_path / "previous.json"
    previous.write_text(json.dumps({
        "realm": "SYNTHETIC_BETA", "status": "NO_QUALIFIED_BETA_MODEL",
        "locked_test_use_count": 0,
    }))
    partial = tmp_path / "round3"
    for source in ("watch", "h10", "muse", "watch+h10", "watch+muse",
                   "h10+muse", "watch+h10+muse"):
        for head in ("current", "onset60"):
            path = partial / "models" / source / f"{head}.onnx"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{source}:{head}".encode())
    runtime = tmp_path / "runtime.whl"
    runtime.write_bytes(b"runtime")
    result = finalize_post_export_failure(
        partial, previous, tmp_path / "final", corrected_fit_rounds=3,
        diagnostic_pre_freeze_fits=2, diagnostic_contract_invalid_fits=56,
        diagnostic_export_debug_fits=11, corrected_runtime_wheel=runtime,
    )
    assert result["status"] == "NO_QUALIFIED_BETA_MODEL"
    assert result["model_role"] is None
    assert result["selected_head_count"] == 0
    assert result["partial_model_file_count"] == 14
    assert result["total_recorded_m2_fit_attempt_count"] == 153
    assert result["locked_test_use_count"] == 0
