"""Create-only helpers for local synthetic beta training artifacts."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path

from synapbridge_beta_runtime.codec import canonical_bytes
from synapbridge_beta_runtime.model_types import SourceReference, SourceStandardModel

WIRE_NAMES = (
    "ack.json", "admin_binding.json", "native.json", "observer.json",
    "phone_binding.json", "projection.json", "result.json", "run.json",
)


def write_standards(
    output: Path, standards: Mapping[str, SourceStandardModel],
) -> list[dict[str, str]]:
    refs = []
    for family, model in standards.items():
        body = {"family": model.family, "floors": dict(model.floors),
                "feature_order": model.feature_order, "schema_sha": model.schema_sha}
        path = output / "standards" / f"{family}.json"
        payload = canonical_bytes(body)
        write_create_only(path, payload)
        refs.append({"family": family, "uri": path.resolve().as_uri(),
                     "sha256": sha256(payload)})
    return refs


def write_references(
    output: Path, references: Mapping[str, Mapping[str, SourceReference]],
) -> dict[str, str]:
    body = {subject: {family: {"centers": dict(ref.centers),
                              "scales": dict(ref.scales),
                              "feature_order": ref.feature_order,
                              "calibration_sha": ref.calibration_sha,
                              "scored_valid_seconds": ref.scored_valid_seconds,
                              "support_kind": ref.support_kind}
                      for family, ref in by_family.items()}
            for subject, by_family in references.items()}
    path = output / "personal_references.json"
    payload = canonical_bytes(body)
    write_create_only(path, payload)
    return {"uri": path.resolve().as_uri(), "sha256": sha256(payload)}


def wire_refs(root: Path) -> list[dict[str, str]]:
    refs = []
    for folder in ("schemas/synthetic_beta_v1",
                   "services/inference/tests/synthetic_beta/golden"):
        directory = root / folder
        if {path.name for path in directory.glob("*.json")} != set(WIRE_NAMES):
            raise ValueError("beta_wire_contract_incomplete")
        for name in WIRE_NAMES:
            path = directory / name
            refs.append({"name": path.name, "kind": folder.split("/")[-1],
                         "sha256": sha256(path.read_bytes())})
    return refs


def copy_runtime_wheel(output: Path, source: Path) -> dict[str, str]:
    if source.suffix != ".whl":
        raise ValueError("beta_runtime_wheel_invalid")
    payload = source.read_bytes()
    target = output / "runtime" / source.name
    write_create_only(target, payload)
    return {"uri": target.resolve().as_uri(), "sha256": sha256(payload)}


def write_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
