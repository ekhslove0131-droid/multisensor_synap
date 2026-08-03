"""Fail-closed loader and inference adapter for monitoring-v2 ONNX bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from multisensor_ml.monitoring_v2 import V2_BUNDLE_SCHEMA, V2_VARIANTS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_monitoring_manifest(manifest: dict[str, object]) -> None:
    required = {
        "schema_version",
        "bundle_kind",
        "variant",
        "feature_names",
        "pattern_feature_names",
        "standard_model_path",
        "pattern_model_paths",
        "pattern_codes",
        "model_scope",
        "real_data_status",
        "locked_test_read",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        raise ValueError(f"monitoring-v2 manifest is missing {missing}")
    if manifest["schema_version"] != V2_BUNDLE_SCHEMA:
        raise ValueError("unsupported monitoring-v2 bundle schema")
    if manifest["variant"] not in V2_VARIANTS:
        raise ValueError(f"unsupported monitoring-v2 variant: {manifest['variant']!r}")
    for field in ("feature_names", "pattern_feature_names", "pattern_codes"):
        value = manifest[field]
        if not isinstance(value, list) or not value or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError(f"{field} must be a non-empty string list")
    if not isinstance(manifest["standard_model_path"], str):
        raise ValueError("standard_model_path must be relative")
    if Path(str(manifest["standard_model_path"])).is_absolute():
        raise ValueError("standard_model_path must be relative")
    paths = manifest["pattern_model_paths"]
    if not isinstance(paths, dict):
        raise ValueError("pattern_model_paths must be a mapping")
    if any(Path(str(path)).is_absolute() for path in paths.values()):
        raise ValueError("pattern model paths must be relative")
    if manifest["model_scope"] != "oracle/sanity":
        raise ValueError("monitoring-v2 model_scope must remain oracle/sanity")
    if manifest["real_data_status"] != "NOT VERIFIED":
        raise ValueError("real_data_status must remain NOT VERIFIED")
    if manifest["locked_test_read"] is not False:
        raise ValueError("locked_test_read must remain false")


def _probability(outputs: list[Any]) -> float:
    for output in outputs:
        array = np.asarray(output)
        if array.ndim == 2 and array.shape[1] >= 2:
            return float(array[0, 1])
    for output in outputs:
        array = np.asarray(output)
        if (
            array.ndim == 1
            and array.size == 1
            and np.issubdtype(array.dtype, np.floating)
        ):
            return float(array[0])
    raise ValueError("pattern ONNX output has no probability column")


def _regression(outputs: list[Any]) -> float:
    for output in outputs:
        array = np.asarray(output)
        if array.size == 1 and np.issubdtype(array.dtype, np.number):
            return float(array.reshape(-1)[0])
    raise ValueError("standard ONNX output has no scalar regression value")


class MonitoringV2Bundle:
    """Load one variant's standard model and its virtual-pattern heads."""

    def __init__(
        self, root: Path, manifest: dict[str, object], standard: Any, patterns: dict[str, Any]
    ):
        self.root = root
        self.manifest = manifest
        self.standard = standard
        self.patterns = patterns

    @classmethod
    def load(cls, root: Path) -> MonitoringV2Bundle:
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("install the onnx-serving extra") from error
        root = root.resolve()
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("monitoring-v2 manifest must be an object")
        validate_monitoring_manifest(manifest)
        hashes = manifest.get("model_sha256", {})
        if not isinstance(hashes, dict):
            raise ValueError("model_sha256 must be a mapping")
        standard_path = root / str(manifest["standard_model_path"])
        if not standard_path.is_file():
            raise FileNotFoundError(f"standard ONNX model is missing: {standard_path}")
        if hashes.get(standard_path.name) != _sha256(standard_path):
            raise ValueError("standard ONNX hash does not match manifest")
        standard = ort.InferenceSession(
            str(standard_path), providers=["CPUExecutionProvider"]
        )
        if not standard.get_inputs() or standard.get_inputs()[0].name != "features":
            raise ValueError("standard ONNX input must be named features")
        patterns: dict[str, Any] = {}
        for code, relative in cast(dict[str, str], manifest["pattern_model_paths"]).items():
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(f"pattern ONNX model is missing: {path}")
            if hashes.get(path.name) != _sha256(path):
                raise ValueError(f"pattern ONNX hash does not match manifest: {code}")
            session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            if not session.get_inputs() or session.get_inputs()[0].name != "features":
                raise ValueError(f"pattern ONNX input must be named features: {code}")
            patterns[code] = session
        return cls(root, manifest, standard, patterns)

    @property
    def variant(self) -> str:
        return str(self.manifest["variant"])

    @property
    def feature_names(self) -> tuple[str, ...]:
        values = cast(list[object], self.manifest["feature_names"])
        return tuple(str(value) for value in values)

    @property
    def pattern_feature_names(self) -> tuple[str, ...]:
        values = cast(list[object], self.manifest["pattern_feature_names"])
        return tuple(str(value) for value in values)

    def predict_one(self, features: dict[str, float]) -> dict[str, object]:
        missing = [name for name in self.feature_names if name not in features]
        if missing:
            raise ValueError(f"missing standard features: {missing[:5]}")
        base = np.asarray([[float(features[name]) for name in self.feature_names]], dtype="float32")
        standard_forecast = _regression(self.standard.run(None, {"features": base}))
        pattern_features = dict(features)
        pattern_features["standard_forecast"] = standard_forecast
        pattern_features["standard_residual"] = (
            float(features["watch_load"]) - standard_forecast
        )
        missing_pattern = [
            name for name in self.pattern_feature_names if name not in pattern_features
        ]
        if missing_pattern:
            raise ValueError(f"missing pattern features: {missing_pattern[:5]}")
        matrix = np.asarray(
            [[float(pattern_features[name]) for name in self.pattern_feature_names]],
            dtype="float32",
        )
        thresholds = self.manifest.get("thresholds", {})
        probabilities: dict[str, float] = {}
        predictions: dict[str, bool] = {}
        for code, session in self.patterns.items():
            probability = min(
                1.0,
                max(0.0, _probability(session.run(None, {"features": matrix}))),
            )
            threshold = (
                float(thresholds.get(code, 0.5))
                if isinstance(thresholds, dict)
                else 0.5
            )
            probabilities[code] = probability
            predictions[code] = probability >= threshold
        return {
            "status": "PREDICTED",
            "variant": self.variant,
            "standard_forecast_30m": standard_forecast,
            "pattern_probabilities": probabilities,
            "pattern_predicted": predictions,
            "model_scope": self.manifest["model_scope"],
            "real_data_status": self.manifest["real_data_status"],
            "locked_test_read": False,
        }
