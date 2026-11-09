from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from multisensor_ml.beta_training.bundle import (
    _session_options,
    export_candidate_onnx,
    predict_onnx,
)
from multisensor_ml.beta_training.train import fit_candidate_models


def test_cpu_onnx_probability_and_decision_parity(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    train_x = rng.normal(size=(100, 3)).astype(np.float32)
    train_y = (train_x[:, 0] > 0).astype(np.int64)
    candidate = fit_candidate_models(train_x, train_y, "watch", "current")[0]
    path = tmp_path / "model.onnx"
    receipt = export_candidate_onnx(candidate, path, train_x[:8], threshold=0.55)
    runtime = predict_onnx(path, train_x[:8])
    expected = candidate.estimator.predict_proba(train_x[:8])[:, 1]
    assert np.max(np.abs(runtime - expected)) <= 1e-5
    assert np.array_equal(runtime >= 0.55, expected >= 0.55)
    assert receipt["provider"] == "CPUExecutionProvider"
    assert receipt["max_absolute_error"] <= 1e-5
    assert receipt["decision_mismatch_count"] == 0
    assert receipt["class_order"] == [0, 1]
    assert not list(tmp_path.glob("*.pkl"))


def test_hgb_candidate_also_exports_to_cpu_onnx(tmp_path: Path) -> None:
    rng = np.random.default_rng(17)
    train_x = rng.normal(size=(120, 4)).astype(np.float32)
    train_y = (train_x[:, 0] + train_x[:, 2] > 0).astype(np.int64)
    candidate = fit_candidate_models(train_x, train_y, "watch", "onset60")[1]
    path = tmp_path / "hgb.onnx"
    receipt = export_candidate_onnx(candidate, path, train_x[:12], threshold=0.5)
    assert receipt["algorithm"] == "HGB"
    assert receipt["opset"] == 17
    model = onnx.load(path)
    assert {(item.domain, item.version) for item in model.opset_import} == {
        ("", 17), ("ai.onnx.ml", 3),
    }
    assert len(model.opset_import) == 2


def test_hgb_export_preserves_float32_scaler_tree_boundary(tmp_path: Path) -> None:
    rng = np.random.default_rng(20261007)
    train_x = rng.normal(size=(400, 6)).astype(np.float32)
    train_y = (train_x[:, 0] + 0.3 * train_x[:, 1] > 0).astype(np.int64)
    candidate = fit_candidate_models(train_x, train_y, "watch", "current")[1]
    scaler = candidate.estimator.named_steps["scale"]
    classifier = candidate.estimator.named_steps["classifier"]
    golden = train_x[:128].copy()
    scaled = scaler.transform(golden)
    scaler_onnx = convert_sklearn(
        scaler, initial_types=[("features", FloatTensorType([None, 6]))],
        target_opset=17,
    )
    session = ort.InferenceSession(
        scaler_onnx.SerializeToString(), sess_options=_session_options(),
        providers=["CPUExecutionProvider"],
    )
    onnx_scaled = np.asarray(session.run(None, {"features": golden})[-1])
    root = classifier._predictors[0][0].nodes[0]
    feature = int(root["feature_idx"])
    delta = onnx_scaled[:, feature] - scaled[:, feature]
    row = int(np.flatnonzero(delta != 0)[0])
    root["num_threshold"] = (
        scaled[row, feature] if delta[row] > 0 else onnx_scaled[row, feature]
    )
    path = tmp_path / "hgb-boundary.onnx"
    export_candidate_onnx(candidate, path, golden, threshold=0.5)
    assert predict_onnx(path, golden) == pytest.approx(
        candidate.estimator.predict_proba(golden)[:, 1], abs=1e-5,
    )
