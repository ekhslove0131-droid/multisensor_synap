"""CPU ONNX export and parity checks without pickle runtime artifacts."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnx
import onnx.helper
import onnxruntime as ort
from onnx import TensorProto, numpy_helper
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from .train import BetaCandidate


def export_candidate_onnx(candidate: BetaCandidate, path: Path, golden_x: np.ndarray,
                          *, threshold: float) -> dict[str, object]:
    if path.exists() or path.suffix != ".onnx":
        raise FileExistsError("onnx output exists or is invalid")
    model = _convert_candidate(candidate)
    payload = model.SerializeToString()
    expected = candidate.estimator.predict_proba(np.asarray(golden_x, dtype=np.float32))[:, 1]
    actual = _predict_session(ort.InferenceSession(
        payload, sess_options=_session_options(), providers=["CPUExecutionProvider"]
    ), golden_x)
    class_order = [int(value) for value in candidate.estimator.steps[-1][1].classes_]
    if class_order != [0, 1]:
        raise ValueError("ONNX_CLASS_ORDER_INVALID")
    max_error = float(np.max(np.abs(actual - expected)))
    decision_mismatches = int(np.sum((actual >= threshold) != (expected >= threshold)))
    if max_error > 1e-5:
        raise ValueError("ONNX_PARITY_FAILED")
    if decision_mismatches:
        raise ValueError("EXPORT_DECISION_UNSTABLE")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    return {"schema_version": "synthetic_beta_onnx_receipt_v1",
            "algorithm": candidate.algorithm, "source_set": candidate.source_set,
            "head": candidate.head, "threshold": threshold, "opset": 17,
            "provider": "CPUExecutionProvider", "input_name": "features",
            "output_name": "probabilities", "input_width": candidate.input_width,
            "sha256": hashlib.sha256(payload).hexdigest(), "golden_rows": len(golden_x),
            "absolute_tolerance": 1e-5, "max_absolute_error": max_error,
            "decision_mismatch_count": decision_mismatches, "class_order": class_order,
            "converter_compatibility": ("float32_sub_div_scaler_and_bool_coercion_v1"
                                         if candidate.algorithm == "HGB" else None)}


def _convert_candidate(candidate: BetaCandidate) -> onnx.ModelProto:
    classifier = candidate.estimator.steps[-1][1]
    options = {id(classifier): {"zipmap": False}}
    if candidate.algorithm != "HGB":
        model = cast(onnx.ModelProto, convert_sklearn(
            candidate.estimator,
            initial_types=[("features", FloatTensorType([None, candidate.input_width]))],
            target_opset={"": 17, "ai.onnx.ml": 3}, options=options,
        ))
        return _freeze_opsets(model)
    scaler = candidate.estimator.steps[0][1]
    mean = np.asarray(scaler.mean_, dtype=np.float32)
    scale = np.asarray(scaler.scale_, dtype=np.float32)
    if (mean.shape != (candidate.input_width,) or scale.shape != mean.shape
            or not np.isfinite(mean).all() or not np.isfinite(scale).all()
            or np.any(scale <= 0)):
        raise ValueError("HGB_SCALER_CONTRACT_INVALID")
    with _tree_attribute_compatibility(True):
        model = cast(onnx.ModelProto, convert_sklearn(
            classifier,
            initial_types=[("scaled_features",
                            FloatTensorType([None, candidate.input_width]))],
            target_opset={"": 17, "ai.onnx.ml": 3}, options=options,
        ))
    graph = model.graph
    del graph.input[:]
    graph.input.append(onnx.helper.make_tensor_value_info(
        "features", TensorProto.FLOAT, [None, candidate.input_width],
    ))
    graph.initializer.extend((
        numpy_helper.from_array(mean, name="scaler_mean"),
        numpy_helper.from_array(scale, name="scaler_scale"),
    ))
    original = list(graph.node)
    del graph.node[:]
    graph.node.extend((
        onnx.helper.make_node("Sub", ("features", "scaler_mean"), ("centered",)),
        onnx.helper.make_node("Div", ("centered", "scaler_scale"),
                              ("scaled_features",)),
        *original,
    ))
    return _freeze_opsets(model)


def _freeze_opsets(model: onnx.ModelProto) -> onnx.ModelProto:
    del model.opset_import[:]
    for domain, version in (("", 17), ("ai.onnx.ml", 3)):
        item = model.opset_import.add()
        item.domain = domain
        item.version = version
    onnx.checker.check_model(model)
    return model


def predict_onnx(path: Path, values: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(path), sess_options=_session_options(),
                                   providers=["CPUExecutionProvider"])
    return _predict_session(session, values)


def _predict_session(session: ort.InferenceSession, values: np.ndarray) -> np.ndarray:
    outputs = session.run(None, {"features": np.asarray(values, dtype=np.float32)})
    probability = outputs[-1]
    if probability.ndim != 2 or probability.shape[1] != 2:
        raise ValueError("onnx_probability_shape_invalid")
    return np.asarray(probability[:, 1], dtype=float)


def _session_options() -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    return options


@contextmanager
def _tree_attribute_compatibility(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    original = onnx.helper.make_attribute

    def safe_make_attribute(key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key == "nodes_missing_value_tracks_true":
            value = [int(item) for item in value]
        return original(key, value, *args, **kwargs)

    onnx.helper.make_attribute = safe_make_attribute
    try:
        yield
    finally:
        onnx.helper.make_attribute = original
