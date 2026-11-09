"""Frozen bundle and artifact validation before locked-test access."""

from __future__ import annotations

import hashlib
from pathlib import Path

from synapbridge_beta_runtime.codec import canonical_bytes
from synapbridge_beta_runtime.types import local_path


def validate_bundle_contract(value: object) -> None:
    if (not isinstance(value, dict) or value.get("realm") != "SYNTHETIC_BETA"
            or value.get("allowed_use") != "BETA_ONLY"
            or value.get("locked_test_used") is not False):
        raise ValueError("beta_bundle_invalid")


def validate_selected_frozen_inputs(
    bundle: dict[str, object], dataset_path: Path,
) -> tuple[Path, str]:
    if (bundle.get("status") != "BETA_SELECTED"
            or bundle.get("model_role") != "BETA_SELECTED"):
        raise ValueError("beta_bundle_invalid")
    content_sha = bundle.get("content_sha256")
    body = {key: item for key, item in bundle.items() if key != "content_sha256"}
    if (not isinstance(content_sha, str)
            or hashlib.sha256(canonical_bytes(body)).hexdigest() != content_sha):
        raise ValueError("beta_bundle_content_mismatch")
    resolved = dataset_path.resolve()
    dataset_sha = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if dataset_sha != bundle.get("dataset_manifest_sha256"):
        raise ValueError("locked_dataset_sha_mismatch")
    models = bundle.get("pattern_models")
    if not isinstance(models, list) or len(models) != 14:
        raise ValueError("beta_bundle_invalid")
    expected = {
        (source, head)
        for source in ("watch", "h10", "muse", "watch+h10", "watch+muse",
                       "h10+muse", "watch+h10+muse")
        for head in ("current", "onset60")
    }
    identities = {
        (str(model.get("source_set")), str(model.get("head")))
        for model in models if isinstance(model, dict)
    }
    if identities != expected:
        raise ValueError("beta_bundle_invalid")
    for model in models:
        if not isinstance(model, dict):
            raise ValueError("beta_bundle_invalid")
        content = local_path(str(model.get("model_uri"))).read_bytes()
        if hashlib.sha256(content).hexdigest() != model.get("model_sha256"):
            raise ValueError("locked_model_sha_mismatch")
    return resolved, dataset_sha
