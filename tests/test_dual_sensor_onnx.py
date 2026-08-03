from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from multisensor_ml.dual_sensor_onnx import (
    SENSOR_VARIANTS,
    feature_names_for_variant,
    validate_bundle_manifest,
)

FEATURES = (
    "autonomic_arousal__robust_z",
    "motor_activation__robust_z",
    "cognitive_load__mean_5s",
    "time_sin",
    "context__sleep",
)


def test_sensor_feature_contract_is_ordered_and_pair_superset() -> None:
    watch = feature_names_for_variant(FEATURES, "galaxy_watch")
    pair = feature_names_for_variant(FEATURES, "galaxy_watch_h10")

    assert SENSOR_VARIANTS == ("galaxy_watch", "galaxy_watch_h10")
    assert watch == (
        "motor_activation__robust_z",
        "cognitive_load__mean_5s",
        "time_sin",
        "context__sleep",
    )
    assert pair == FEATURES
    assert set(watch).issubset(pair)


def test_bundle_manifest_rejects_real_or_locked_test_scope() -> None:
    valid = {
        "schema_version": "goal1.5/onnx-bundle/v1",
        "variant": "galaxy_watch",
        "feature_names": ["f1"],
        "input_name": "features",
        "model_path": "model.onnx",
        "model_scope": "oracle/sanity",
        "real_data_status": "NOT VERIFIED",
        "locked_test_read": False,
    }

    validate_bundle_manifest(valid)
    with pytest.raises(ValueError, match="locked_test_read"):
        validate_bundle_manifest({**valid, "locked_test_read": True})
    with pytest.raises(ValueError, match="real_data_status"):
        validate_bundle_manifest({**valid, "real_data_status": "VERIFIED"})


def test_onnx_bundle_manifest_round_trip(tmp_path: Path) -> None:
    manifest = {
        "schema_version": "goal1.5/onnx-bundle/v1",
        "variant": "galaxy_watch_h10",
        "feature_names": ["f1", "f2"],
        "input_name": "features",
        "model_path": "model.onnx",
        "model_scope": "oracle/sanity",
        "real_data_status": "NOT VERIFIED",
        "locked_test_read": False,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    validate_bundle_manifest(loaded)
    assert np.asarray([1.0], dtype="float32").dtype == np.float32
