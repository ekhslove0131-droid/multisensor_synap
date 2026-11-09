"""Bounded prepared-cohort reader with role-specific label access."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from synapbridge_beta_runtime.codec import canonical_bytes
from synapbridge_beta_runtime.feature_catalog import feature_order, schema_sha
from synapbridge_beta_runtime.feature_types import FeatureFrame
from synapbridge_beta_runtime.types import BetaScope, local_path


@dataclass(frozen=True, slots=True)
class LoadedCohort:
    manifest: dict[str, Any]
    frames: dict[str, dict[str, tuple[FeatureFrame, ...]]]
    events: dict[str, tuple[dict[str, Any], ...]]
    roles: dict[str, str]


def load_cohort(manifest_path: Path, roles: tuple[str, ...]) -> LoadedCohort:
    allowed = {"TRAIN", "VALIDATION", "LOCKED_TEST"}
    if not roles or not set(roles) <= allowed:
        raise ValueError("cohort_roles_invalid")
    manifest = json.loads(manifest_path.read_bytes())
    _manifest_contract(manifest)
    _feature_schema_contract(manifest["feature_schema_refs"])
    scope = BetaScope(str(manifest["beta_run_uuid"]), int(manifest["run_epoch"]), "")
    frames: dict[str, dict[str, tuple[FeatureFrame, ...]]] = defaultdict(dict)
    subject_roles: dict[str, str] = {}
    for ref in manifest["feature_refs"]:
        if ref["split"] not in roles:
            continue
        subject = str(ref["subject_uuid"])
        subject_roles[subject] = str(ref["split"])
        rows = _json_lines(ref)
        family = str(ref["family"])
        frames[subject][family] = tuple(
            _feature_frame(
                scope, subject, family, row,
            )
            for row in rows
        )
    events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ref in manifest["label_refs"]:
        if ref["role"] not in roles:
            continue
        for row in _json_lines(ref):
            subject = str(row["subject_uuid"])
            if subject_roles.get(subject) != ref["role"]:
                raise ValueError("prepared_label_role_mismatch")
            events[subject].append(row)
    if not subject_roles or any(subject not in events for subject in subject_roles):
        raise ValueError("prepared_cohort_empty")
    return LoadedCohort(
        manifest,
        dict(frames),
        {subject: tuple(sorted(rows, key=lambda item: int(item["onset_ms"])))
         for subject, rows in events.items()},
        subject_roles,
    )


def _feature_frame(
    scope: BetaScope, subject: str, family: str, row: dict[str, Any],
) -> FeatureFrame:
    raw_values = row["values"]
    if not isinstance(raw_values, dict) or not set(raw_values) <= set(feature_order(family)):
        raise ValueError("prepared_feature_values_invalid")
    ordered = {name: float(raw_values[name]) for name in feature_order(family)
               if name in raw_values}
    return FeatureFrame(
        BetaScope(scope.run_uuid, scope.epoch, subject),
        int(row["decision_ms"]), family,
        MappingProxyType(ordered),
        (int(row["valid_support_ms"][0]), int(row["valid_support_ms"][1])),
        float(row["quality"]), str(row["schema_sha"]),
    )


def _json_lines(ref: dict[str, Any]) -> list[dict[str, Any]]:
    content = local_path(str(ref["uri"])).read_bytes()
    if hashlib.sha256(content).hexdigest() != ref["sha256"]:
        raise ValueError("prepared_ref_sha_mismatch")
    rows = [json.loads(line) for line in content.splitlines()]
    if len(rows) != ref["row_count"]:
        raise ValueError("prepared_ref_count_mismatch")
    return rows


def _manifest_contract(value: object) -> None:
    required = {
        "schema_version", "realm", "beta_run_uuid", "run_epoch", "config",
        "timeline_origin_ms", "source_manifest_sha256", "protocol_ref", "split_artifact",
        "label_refs", "feature_schema_refs", "feature_refs", "content_sha256",
    }
    if (not isinstance(value, dict) or set(value) != required
            or value["schema_version"] != "synthetic_beta_prepared_dataset_v1"
            or value["realm"] != "SYNTHETIC_BETA"
            or hashlib.sha256(canonical_bytes({
                key: item for key, item in value.items() if key != "content_sha256"
            })).hexdigest() != value["content_sha256"]):
        raise ValueError("prepared_manifest_invalid")


def _feature_schema_contract(refs: object) -> None:
    if not isinstance(refs, list) or len(refs) != 3:
        raise ValueError("prepared_feature_schema_invalid")
    families = set()
    for ref in refs:
        if not isinstance(ref, dict) or set(ref) != {"family", "uri", "sha256", "schema_sha"}:
            raise ValueError("prepared_feature_schema_invalid")
        family = str(ref["family"])
        content = local_path(str(ref["uri"])).read_bytes()
        if (hashlib.sha256(content).hexdigest() != ref["sha256"]
                or ref["schema_sha"] != schema_sha(family)):
            raise ValueError("prepared_feature_schema_invalid")
        value = json.loads(content)
        if value.get("sha256") != ref["schema_sha"]:
            raise ValueError("prepared_feature_schema_invalid")
        families.add(family)
    if families != {"watch", "h10", "muse"}:
        raise ValueError("prepared_feature_schema_invalid")
