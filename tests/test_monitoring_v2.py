from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from multisensor_ml.monitoring_v2 import (
    PATTERN_CODES,
    STANDARD_HORIZON_SEC,
    _future_normal_mean,
    _variant_base_features,
)
from multisensor_ml.monitoring_v2_onnx import validate_monitoring_manifest


def test_future_target_is_causal_and_excludes_current_row() -> None:
    values = np.arange(6, dtype="float64")
    result = _future_normal_mean(values, np.zeros(6, dtype=bool), 2)
    np.testing.assert_allclose(result[:4], [1.5, 2.5, 3.5, 4.5])
    assert np.isnan(result[4:]).all()


def test_future_target_ignores_event_rows() -> None:
    values = np.arange(6, dtype="float64")
    event = np.array([False, True, False, False, False, False])
    result = _future_normal_mean(values, event, 2)
    np.testing.assert_allclose(result[:4], [2.0, 2.5, 3.5, 4.5])


def test_sensor_view_feature_contract() -> None:
    watch = _variant_base_features("galaxy_watch")
    dual = _variant_base_features("galaxy_watch_h10")
    assert set(watch).issubset(dual)
    assert any(name.startswith("h10_") for name in dual)
    assert not any("truth" in name or "event_id" in name for name in dual)


def test_manifest_rejects_real_or_locked_scope() -> None:
    base = {
        "schema_version": "goal1.5/monitoring-onnx-bundle/v1",
        "bundle_kind": "standard_30m_plus_virtual_patterns",
        "variant": "galaxy_watch",
        "feature_names": ["watch_load"],
        "pattern_feature_names": ["watch_load", "standard_forecast", "standard_residual"],
        "standard_model_path": "standard_30m.onnx",
        "pattern_model_paths": {PATTERN_CODES[0]: "pattern.onnx"},
        "pattern_codes": list(PATTERN_CODES),
        "model_scope": "oracle/sanity",
        "real_data_status": "NOT VERIFIED",
        "locked_test_read": False,
    }
    validate_monitoring_manifest(base)
    invalid = dict(base, locked_test_read=True)
    with pytest.raises(ValueError, match="locked_test_read"):
        validate_monitoring_manifest(invalid)


def test_generated_dataset_contract_if_materialized() -> None:
    manifest_path = Path("data/model_ready/standard-pattern-v2-quick/manifest.json")
    if not manifest_path.is_file():
        pytest.skip("local synthetic materialization is not part of a clean checkout")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["standard_horizon_sec"] == STANDARD_HORIZON_SEC
    assert manifest["split_contract"] == {"train": 24, "validation": 6, "locked_test": 6}
    assert manifest["locked_test_read"] is False
    assert set(manifest["pattern_codes"]) == set(PATTERN_CODES)
