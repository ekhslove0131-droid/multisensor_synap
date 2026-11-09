"""Create-only cohort export kept separate from waveform generation."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from types import MappingProxyType

from synapbridge_beta_runtime.codec import decode_native
from synapbridge_beta_runtime.types import NativePart, local_path

from .codec import canonical_bytes, write_json_create_only, write_part
from .config import (
    FAMILIES,
    BetaConfig,
    BetaSignalState,
    LocalObservedManifest,
    deterministic_uuid4,
)
from .generator import generate_part
from .protocol import PROTOCOL, PROTOCOL_SHA256


def generate_observed(config: BetaConfig, output: Path) -> LocalObservedManifest:
    if output.exists():
        raise FileExistsError("beta output already exists")
    output.mkdir(parents=True)
    config_value = {name: getattr(config, name) for name in config.__dataclass_fields__}
    config_sha = hashlib.sha256(canonical_bytes(config_value)).hexdigest()
    protocol_path = output / "protocol.json"
    protocol_sha = write_json_create_only(protocol_path, PROTOCOL)
    if protocol_sha != PROTOCOL_SHA256:
        raise ValueError("beta_protocol_sha_mismatch")
    run_uuid = deterministic_uuid4(config.seed, config_sha, protocol_sha, "run")
    origin_ms = 1_000_000
    subjects, parts, truth_rows, split_rows = [], [], [], []
    for index in range(config.subject_count):
        split = _split(config, index)
        subject = deterministic_uuid4(config.seed, "subject", index)
        sources = {
            family: deterministic_uuid4(config.seed, subject, family)
            for family in FAMILIES
        }
        events = _events(config, origin_ms)
        state = BetaSignalState(
            run_uuid, 1, subject, MappingProxyType(sources), config.seed + index,
            origin_ms, index, split, 66.0 + (index % 9),
            ("RESPONSIVE", "WEAK", "INVERSE", "NO_RESPONSE")[index % 4], events,
        )
        subjects.append({"subject_uuid": subject, "source_uuids": sources})
        split_rows.append({"subject_uuid": subject, "split": split})
        truth_rows.extend(_truth_rows(state))
        for window_no, offset in enumerate(range(0, config.duration_seconds, 10)):
            start = origin_ms + offset * 1000
            for family in FAMILIES:
                parts.append(write_part(
                    generate_part(state, family, start),
                    output / "observed" / "native",
                    window_no,
                ))
    truth_path = output / "truth" / "events.jsonl"
    truth_bytes = b"".join(canonical_bytes(row) + b"\n" for row in truth_rows)
    _write_bytes(truth_path, truth_bytes)
    split_path = output / "truth" / "splits.json"
    split_sha = write_json_create_only(split_path, split_rows)
    observed_body = {
        "schema_version": "synthetic_beta_observed_manifest_v1",
        "realm": "SYNTHETIC_BETA", "beta_run_uuid": run_uuid, "run_epoch": 1,
        "config_sha256": config_sha, "protocol_sha256": protocol_sha,
        "subjects": subjects, "parts": parts,
    }
    observed = {**observed_body, "content_sha256": _digest(observed_body)}
    observed_path = output / "observed_manifest.json"
    observed_sha = write_json_create_only(observed_path, observed)
    training = {
        "schema_version": "synthetic_beta_training_source_v1",
        "realm": "SYNTHETIC_BETA", "beta_run_uuid": run_uuid, "run_epoch": 1,
        "config": config_value, "config_sha256": config_sha,
        "protocol_ref": _ref(protocol_path, protocol_sha),
        "observed_manifest": _ref(observed_path, observed_sha),
        "truth_artifact": _ref(truth_path, hashlib.sha256(truth_bytes).hexdigest()),
        "split_artifact": _ref(split_path, split_sha),
    }
    manifest_path = output / "manifest.json"
    manifest_sha = write_json_create_only(manifest_path, training)
    return LocalObservedManifest(manifest_path, observed_path, truth_path, manifest_sha)


def iter_native_parts(manifest: LocalObservedManifest) -> Iterator[NativePart]:
    training = json.loads(manifest.manifest_path.read_bytes())
    observed = json.loads(local_path(training["observed_manifest"]["uri"]).read_bytes())
    for item in observed["parts"]:
        manifest_value = json.loads(local_path(item["manifest_uri"]).read_bytes())
        yield decode_native(manifest_value, local_path(item["content_uri"]).read_bytes())


def _split(config: BetaConfig, index: int) -> str:
    if index < config.train_subjects:
        return "TRAIN"
    if index < config.train_subjects + config.validation_subjects:
        return "VALIDATION"
    return "LOCKED_TEST"


def _events(config: BetaConfig, origin: int) -> tuple[tuple[int, int], ...]:
    scored_start = config.prefill_seconds + config.calibration_seconds
    starts = ((scored_start + 300, scored_start + 900) if config.scored_seconds >= 1200
              else tuple(scored_start + 2 + index * 4
                         for index in range(config.events_per_subject)))
    return tuple(
        (origin + value * 1000,
         origin + min(config.duration_seconds - 1, value + 180) * 1000)
        for value in starts[:config.events_per_subject]
    )


def _truth_rows(state: BetaSignalState) -> list[dict[str, object]]:
    return [
        {"event_uuid": deterministic_uuid4(state.seed, state.subject_uuid, onset),
         "subject_uuid": state.subject_uuid, "split": state.split,
         "onset_ms": onset, "offset_ms": offset,
         "pre_early_start_ms": onset - 60_000,
         "pre_late_start_ms": onset - 30_000,
         "recovery_early_start_ms": offset,
         "recovery_late_start_ms": offset + 60_000,
         "phase_end_ms": offset + 120_000,
         "response_kind": state.response_kind}
        for onset, offset in state.events_ms
    ]


def _ref(path: Path, sha: str) -> dict[str, str]:
    return {"uri": path.resolve().as_uri(), "sha256": sha}


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
