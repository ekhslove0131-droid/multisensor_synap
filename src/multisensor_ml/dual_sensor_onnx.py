"""Sensor-view contracts and safe ONNX inference for the two-device MVP."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

SensorVariant = Literal["galaxy_watch", "galaxy_watch_h10"]
SENSOR_VARIANTS: tuple[SensorVariant, SensorVariant] = (
    "galaxy_watch",
    "galaxy_watch_h10",
)

# The current oracle features are latent proxies, not physical device channels.
# We keep this mapping explicit so a future observed adapter can replace it
# without silently changing a model's input contract.
WATCH_FEATURE_BASES: frozenset[str] = frozenset(
    {
        "motor_activation",
        "cognitive_load",
        "sleep_pressure",
        "sensory_context",
        "recovery_capacity",
        "social_context",
    }
)
H10_FEATURE_BASES: frozenset[str] = frozenset({"autonomic_arousal"})
SHARED_CONTEXT_PREFIXES: tuple[str, ...] = (
    "time_",
    "weekday_",
    "context__",
)


def _variant(value: str) -> SensorVariant:
    if value not in SENSOR_VARIANTS:
        raise ValueError(f"unsupported sensor variant: {value!r}")
    return value


def feature_names_for_variant(
    all_feature_names: Sequence[str], variant: str
) -> tuple[str, ...]:
    """Return a stable, source-order feature list for a sensor view."""

    selected_variant = _variant(variant)
    selected: list[str] = []
    for name in all_feature_names:
        base = name.split("__", 1)[0]
        shared = name == "is_awake" or name.startswith(SHARED_CONTEXT_PREFIXES)
        watch = base in WATCH_FEATURE_BASES
        h10 = base in H10_FEATURE_BASES
        if shared or watch or (selected_variant == "galaxy_watch_h10" and h10):
            selected.append(str(name))
    if not selected:
        raise ValueError(f"no features available for sensor variant: {selected_variant}")
    return tuple(selected)


def validate_bundle_manifest(manifest: Mapping[str, object]) -> None:
    """Fail closed on scope, schema, or locked-test violations."""

    required = {
        "schema_version",
        "variant",
        "feature_names",
        "input_name",
        "model_path",
        "model_scope",
        "real_data_status",
        "locked_test_read",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        raise ValueError(f"ONNX bundle manifest is missing fields: {missing}")
    if manifest["schema_version"] != "goal1.5/onnx-bundle/v1":
        raise ValueError("unsupported ONNX bundle schema_version")
    _variant(str(manifest["variant"]))
    feature_names = manifest["feature_names"]
    if not isinstance(feature_names, list) or not feature_names or not all(
        isinstance(value, str) and value for value in feature_names
    ):
        raise ValueError("feature_names must be a non-empty string list")
    if not isinstance(manifest["input_name"], str) or not manifest["input_name"]:
        raise ValueError("input_name must be a non-empty string")
    model_path = manifest["model_path"]
    if not isinstance(model_path, str) or not model_path or Path(model_path).is_absolute():
        raise ValueError("model_path must be a relative path")
    if manifest["model_scope"] != "oracle/sanity":
        raise ValueError("model_scope must remain oracle/sanity")
    if manifest["real_data_status"] != "NOT VERIFIED":
        raise ValueError("real_data_status must remain NOT VERIFIED")
    if manifest["locked_test_read"] is not False:
        raise ValueError("locked_test_read must be false")


def _positive_probability(outputs: Sequence[object]) -> float:
    for output in outputs:
        if isinstance(output, list) and output and isinstance(output[0], Mapping):
            first = output[0]
            for key in ("1", 1, "positive", "probability"):
                if key in first:
                    return float(cast(Any, first[key]))
        array = np.asarray(output)
        if array.ndim == 2 and array.shape[1] >= 2:
            return float(array[0, 1])
        if array.ndim == 1 and array.size == 1:
            return float(array[0])
    raise ValueError("ONNX output did not contain a positive probability")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(slots=True)
class OnnxBundle:
    """A validated ONNX model and its immutable input/threshold contract."""

    root: Path
    manifest: dict[str, object]
    _session: Any

    @classmethod
    def load(cls, root: Path) -> OnnxBundle:
        bundle_root = root.resolve()
        manifest_path = bundle_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"ONNX bundle manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("ONNX bundle manifest must be an object")
        validate_bundle_manifest(manifest)
        model_path = bundle_root / str(manifest["model_path"])
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model is missing: {model_path}")
        expected_hash = manifest.get("model_sha256")
        if expected_hash is not None and (
            not isinstance(expected_hash, str) or _sha256(model_path) != expected_hash
        ):
            raise ValueError("ONNX model SHA-256 does not match the bundle manifest")
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]
        except ImportError as error:  # pragma: no cover - serving extra guard
            raise RuntimeError("install the onnx-serving extra to load ONNX") from error
        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        input_names = {item.name for item in session.get_inputs()}
        if str(manifest["input_name"]) not in input_names:
            raise ValueError("manifest input_name is not present in the ONNX graph")
        return cls(bundle_root, manifest, session)

    @property
    def variant(self) -> SensorVariant:
        return _variant(str(self.manifest["variant"]))

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(cast(list[str], self.manifest["feature_names"]))

    @property
    def threshold(self) -> float:
        value = self.manifest.get("threshold", 0.5)
        if not isinstance(value, (int, float)):
            raise ValueError("threshold must be numeric")
        return float(value)

    def predict_one(self, features: Mapping[str, object]) -> dict[str, object]:
        missing = [name for name in self.feature_names if name not in features]
        if missing:
            raise ValueError(f"missing ONNX features: {missing[:5]}")
        values = np.asarray(
            [[float(cast(Any, features[name])) for name in self.feature_names]], dtype="float32"
        )
        outputs = self._session.run(
            None,
            {str(self.manifest["input_name"]): values},
        )
        probability = min(1.0, max(0.0, _positive_probability(outputs)))
        return {
            "status": "PREDICTED",
            "variant": self.variant,
            "model_id": self.manifest.get("model_id", "unknown"),
            "probability": probability,
            "event_predicted": probability >= self.threshold,
            "threshold": self.threshold,
            "data_status": self.manifest.get("model_scope"),
            "real_data_status": self.manifest.get("real_data_status"),
            "locked_test_read": False,
        }
