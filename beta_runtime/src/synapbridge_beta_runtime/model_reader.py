"""Fail-closed reader for local immutable BETA_ONLY runtime bundles."""

from __future__ import annotations

import hashlib
import json
from types import MappingProxyType

import numpy as np
import onnxruntime as ort

from .codec import canonical_bytes
from .model_contract import pattern_models as _pattern_models
from .model_contract import personal_references as _personal_references
from .model_contract import read_ref as _read_ref
from .model_contract import source_schemas as _source_schemas
from .model_contract import validate_manifest as _validate_manifest
from .model_types import (
    BetaModelBundle,
    BetaStagePolicy,
    LoadedBetaRuntime,
    ModelFrame,
    PatternScores,
)
from .types import BetaScope, LocalArtifact, local_path


def read_beta_runtime(manifest_ref: LocalArtifact, scope: BetaScope) -> LoadedBetaRuntime:
    payload = local_path(manifest_ref.uri).read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest_ref.sha256:
        raise ValueError("beta_bundle_sha_mismatch")
    value = json.loads(payload)
    _validate_manifest(value, scope)
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if hashlib.sha256(canonical_bytes(body)).hexdigest() != value["content_sha256"]:
        raise ValueError("beta_bundle_content_mismatch")
    source_schemas = _source_schemas(value["standard_refs"])
    _read_ref(value["runtime_wheel_ref"], parse=False)
    references, unready = _personal_references(
        value["personal_reference_index"], scope.subject_uuid
    )
    if not references:
        raise ValueError("beta_reference_set_invalid")
    models = _pattern_models(value["pattern_models"])
    policy_value = value["policy"]
    policy = BetaStagePolicy(
        policy_value["version"], float(policy_value["d_on"]),
        float(policy_value["d_off"]), value["policy_sha256"],
    )
    policies = MappingProxyType({source_set: policy for source_set in models})
    start, end = (int(item) for item in value["calibration_bounds_ms"])
    bounds = MappingProxyType({family: (start, end) for family in references})
    bundle = BetaModelBundle(
        manifest_ref.sha256,
        tuple(ref["sha256"] for ref in value["standard_refs"]),
        tuple(model.model.sha256 for heads in models.values() for model in heads.values()),
        policy.sha256,
    )
    return LoadedBetaRuntime(
        bundle, MappingProxyType(references), MappingProxyType(unready),
        MappingProxyType(source_schemas),
        MappingProxyType({key: MappingProxyType(item) for key, item in models.items()}),
        policies, bounds, value["kernel_sha256"], manifest_ref.sha256,
    )


def predict_pattern(frame: ModelFrame, runtime: LoadedBetaRuntime) -> PatternScores:
    heads = runtime.pattern_models.get(frame.source_set)
    if heads is None or set(heads) != {"current", "onset60"}:
        raise ValueError("beta_pattern_model_unavailable")
    scores = {}
    for head, model in heads.items():
        if (model.feature_names != frame.feature_names
                or model.feature_schema_sha != frame.schema_sha):
            raise ValueError("beta_model_frame_mismatch")
        scores[head] = _predict(model.model, frame.values)
    return PatternScores(scores["current"], scores["onset60"])


def _predict(ref: LocalArtifact, values: np.ndarray) -> float:
    payload = local_path(ref.uri).read_bytes()
    if hashlib.sha256(payload).hexdigest() != ref.sha256:
        raise ValueError("beta_artifact_sha_mismatch")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(payload, sess_options=options,
                                   providers=["CPUExecutionProvider"])
    output = session.run(None, {"features": np.asarray(values, dtype=np.float32)[None, :]})[-1]
    return float(output[0, 1])
