from __future__ import annotations

import json
from pathlib import Path

from scripts.run_tree_model_real_shadow import build_shadow_artifact


def test_neon_snapshot_fails_closed_without_model_ready_contract() -> None:
    snapshot = json.loads(Path("reports/neon_mapping_live_snapshot_ko.json").read_text())

    artifact = build_shadow_artifact(snapshot)

    assert artifact["status"] == "BLOCKED_NOT_MODEL_READY"
    assert artifact["prediction_executed"] is False
    assert artifact["accuracy_status"] == "NOT VERIFIED"
    assert artifact["locked_test_read"] is False
    assert "CORRECTED_UTC_MISSING" in artifact["blockers"]
    assert "OBSERVER_STAGE_LABELS_MISSING" in artifact["blockers"]
