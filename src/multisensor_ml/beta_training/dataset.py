"""Stream observed native parts into raw-free causal feature artifacts."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from synapbridge_beta_runtime.codec import canonical_bytes, decode_native
from synapbridge_beta_runtime.feature_catalog import feature_contract
from synapbridge_beta_runtime.features import derive_features
from synapbridge_beta_runtime.types import NativePart, local_path

from .split import PreparedSplit, validate_splits


@dataclass(frozen=True, slots=True)
class PreparedBetaDataset:
    manifest_path: Path
    manifest_sha256: str
    splits: tuple[PreparedSplit, ...]


def prepare_dataset(source_manifest: Path, output: Path) -> PreparedBetaDataset:
    if output.exists():
        raise FileExistsError("prepared output already exists")
    output.mkdir(parents=True)
    source = json.loads(source_manifest.read_bytes())
    _source_contract(source)
    protocol_bytes = local_path(source["protocol_ref"]["uri"]).read_bytes()
    if _sha(protocol_bytes) != source["protocol_ref"]["sha256"]:
        raise ValueError("generation_protocol_sha_mismatch")
    observed_ref = source["observed_manifest"]
    observed_path = local_path(observed_ref["uri"])
    observed_bytes = observed_path.read_bytes()
    if hashlib.sha256(observed_bytes).hexdigest() != observed_ref["sha256"]:
        raise ValueError("observed_manifest_sha_mismatch")
    observed = json.loads(observed_bytes)
    split_rows = json.loads(local_path(source["split_artifact"]["uri"]).read_bytes())
    splits = validate_splits(split_rows)
    roles = {subject: split.role for split in splits for subject in split.subject_uuids}
    feature_schema_refs = _write_feature_schemas(output)
    label_refs = _split_labels(source["truth_artifact"], roles, output)
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for ref in observed["parts"]:
        grouped[(ref["subject_uuid"], ref["family"])].append(ref)
    feature_refs = []
    tasks = [
        (subject, family, sorted(refs, key=lambda item: cast(int, item["start_ms"])),
         output / "features" / subject / f"{family}.jsonl")
        for (subject, family), refs in sorted(grouped.items())
    ]
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("fork")
    ) as executor:
        derived = executor.map(_derive_task, tasks)
    for subject, family, feature_path, rows in derived:
        feature_refs.append({"subject_uuid": subject, "split": roles[subject],
                             "family": family, "uri": feature_path.resolve().as_uri(),
                             "sha256": _sha(feature_path.read_bytes()), "row_count": rows})
    body = {"schema_version": "synthetic_beta_prepared_dataset_v1",
            "realm": "SYNTHETIC_BETA", "beta_run_uuid": source["beta_run_uuid"],
            "run_epoch": source["run_epoch"], "config": source["config"],
            "timeline_origin_ms": min(int(ref["start_ms"]) for ref in observed["parts"]),
            "source_manifest_sha256": _sha(source_manifest.read_bytes()),
            "protocol_ref": source["protocol_ref"],
            "split_artifact": source["split_artifact"], "label_refs": label_refs,
            "feature_schema_refs": feature_schema_refs, "feature_refs": feature_refs}
    manifest = {**body, "content_sha256": hashlib.sha256(canonical_bytes(body)).hexdigest()}
    manifest_path = output / "manifest.json"
    content = canonical_bytes(manifest)
    _write(manifest_path, content)
    return PreparedBetaDataset(manifest_path, _sha(content), splits)


def _derive_task(
    task: tuple[str, str, list[dict[str, Any]], Path],
) -> tuple[str, str, Path, int]:
    subject, family, refs, path = task
    return subject, family, path, _derive_group(refs, path)


def _write_feature_schemas(output: Path) -> list[dict[str, str]]:
    refs = []
    for family in ("watch", "h10", "muse"):
        contract = feature_contract(family)
        payload = canonical_bytes(contract)
        path = output / "feature_schemas" / f"{family}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        _write(path, payload)
        refs.append({"family": family, "uri": path.resolve().as_uri(),
                     "sha256": _sha(payload), "schema_sha": str(contract["sha256"])})
    return refs


def _derive_group(refs: list[dict[str, Any]], path: Path) -> int:
    history: deque[NativePart] = deque(maxlen=18)
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        for ref in refs:
            manifest_bytes = local_path(ref["manifest_uri"]).read_bytes()
            content = local_path(ref["content_uri"]).read_bytes()
            if (_sha(manifest_bytes) != ref["manifest_sha256"]
                    or _sha(content) != ref["content_sha256"]):
                raise ValueError("native_ref_sha_mismatch")
            part = decode_native(json.loads(manifest_bytes), content)
            history.append(part)
            frames = derive_features(tuple(history), part.end_ms)
            if frames:
                frame = frames[0]
                row = {"decision_ms": frame.decision_ms, "family": frame.family,
                       "values": dict(frame.values), "valid_support_ms": frame.valid_support_ms,
                       "quality": frame.quality, "schema_sha": frame.schema_sha}
                handle.write(canonical_bytes(row) + b"\n")
                count += 1
    return count


def _source_contract(value: object) -> None:
    fields = {"schema_version", "realm", "beta_run_uuid", "run_epoch", "config",
              "config_sha256", "protocol_ref", "observed_manifest", "truth_artifact",
              "split_artifact"}
    if (not isinstance(value, dict) or set(value) != fields
            or value["schema_version"] != "synthetic_beta_training_source_v1"
            or value["realm"] != "SYNTHETIC_BETA"):
        raise ValueError("training_source_invalid")


def _split_labels(
    truth_ref: dict[str, str], roles: dict[str, str], output: Path,
) -> list[dict[str, object]]:
    truth_path = local_path(truth_ref["uri"])
    content = truth_path.read_bytes()
    if _sha(content) != truth_ref["sha256"]:
        raise ValueError("truth_artifact_sha_mismatch")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in content.splitlines():
        row = json.loads(line)
        role = roles.get(row.get("subject_uuid"))
        if role is None or row.get("split") != role:
            raise ValueError("truth_split_mismatch")
        grouped[role].append(row)
    refs = []
    for role in ("TRAIN", "VALIDATION", "LOCKED_TEST"):
        rows = sorted(
            grouped[role],
            key=lambda item: (str(item["subject_uuid"]), int(item["onset_ms"])),
        )
        payload = b"".join(canonical_bytes(row) + b"\n" for row in rows)
        path = output / "labels" / f"{role.lower()}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        _write(path, payload)
        refs.append({"role": role, "uri": path.resolve().as_uri(),
                     "sha256": _sha(payload), "row_count": len(rows)})
    return refs


def _write(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
