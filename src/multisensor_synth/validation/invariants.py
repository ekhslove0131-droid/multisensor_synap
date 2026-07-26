from __future__ import annotations

import json
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import cast

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from multisensor_synth.domain.contracts import (
    FORBIDDEN_MEDICAL_EVENT_TERMS,
    HARD_NEGATIVE_TYPES,
)
from multisensor_synth.export.parquet import logical_table_hash
from multisensor_synth.validation.report import (
    ValidationFinding,
    ValidationReport,
)
from multisensor_synth.validation.schema import (
    FORBIDDEN_GOAL_ONE_DIRECTORIES,
    REQUIRED_TRUTH_FILES,
)


def _error(
    findings: list[ValidationFinding],
    code: str,
    message: str,
    **evidence: object,
) -> None:
    findings.append(
        ValidationFinding(
            code=code,
            severity="ERROR",
            message=message,
            evidence=evidence,
        )
    )


def _required_int(value: object, field: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def validate_truth_run(run_dir: str | Path) -> ValidationReport:
    path = Path(run_dir)
    findings: list[ValidationFinding] = []
    checks = 0
    metrics: dict[str, object] = {}
    manifest_path = path / "manifest.json"
    manifest: dict[str, object] = {}
    if not manifest_path.is_file():
        _error(findings, "FILE_MANIFEST", "manifest.json is missing")
    else:
        manifest = cast(
            dict[str, object],
            json.loads(manifest_path.read_text(encoding="utf-8")),
        )
    checks += 1

    for relative in REQUIRED_TRUTH_FILES:
        checks += 1
        if not (path / relative).is_file():
            _error(findings, "FILE_REQUIRED", "required truth file is missing", path=relative)
    for directory in FORBIDDEN_GOAL_ONE_DIRECTORIES:
        checks += 1
        if (path / directory).exists():
            _error(
                findings,
                "LAYER_BOUNDARY",
                "deferred layer exists in Goal 1 truth-only run",
                path=directory,
            )
    if findings:
        return ValidationReport(
            status="FAIL",
            scope="truth",
            run_id=str(manifest.get("run_id")) if manifest else None,
            checks_run=checks,
            findings=tuple(findings),
            metrics=metrics,
        )

    participants = pq.read_table(path / "truth" / "participants.parquet")
    daily = pq.read_table(path / "truth" / "daily_context.parquet")
    latent = pq.read_table(path / "truth" / "latent_timeline.parquet")
    events = pq.read_table(path / "truth" / "events.parquet")
    loaded_tables = {
        "truth/participants.parquet": participants,
        "truth/daily_context.parquet": daily,
        "truth/latent_timeline.parquet": latent,
        "truth/events.parquet": events,
    }
    metrics.update(
        {
            "participant_rows": participants.num_rows,
            "daily_context_rows": daily.num_rows,
            "latent_rows": latent.num_rows,
            "event_rows": events.num_rows,
        }
    )

    checks += 1
    manifest_tables_value = manifest.get("tables")
    manifest_tables = (
        manifest_tables_value if isinstance(manifest_tables_value, dict) else {}
    )
    for relative, table in loaded_tables.items():
        details_value = manifest_tables.get(relative)
        details = details_value if isinstance(details_value, dict) else {}
        expected_hash = details.get("logical_sha256")
        actual_hash = logical_table_hash(table)
        if not isinstance(expected_hash, str) or actual_hash != expected_hash:
            _error(
                findings,
                "LOGICAL_HASH",
                "truth table logical content does not match manifest",
                path=relative,
                expected=expected_hash,
                actual=actual_hash,
            )

    person_ids = cast(list[str], participants["person_id"].to_pylist())
    checks += 1
    if len(person_ids) != len(set(person_ids)):
        _error(findings, "PARTICIPANT_UNIQUE", "person_id values are not unique")

    run_contract_value = manifest.get("run_contract")
    run_contract = run_contract_value if isinstance(run_contract_value, dict) else {}
    duration_value = run_contract.get("duration_sec")
    duration_sec = (
        duration_value
        if isinstance(duration_value, int)
        else latent.num_rows // max(participants.num_rows, 1)
    )
    participant_count_value = run_contract.get("participant_count")
    participant_count = (
        participant_count_value
        if isinstance(participant_count_value, int)
        else participants.num_rows
    )
    expected_rows = participant_count * duration_sec
    checks += 1
    if latent.num_rows != expected_rows:
        _error(
            findings,
            "LATENT_ROW_COUNT",
            "latent row count does not match participant and duration contract",
            expected=expected_rows,
            actual=latent.num_rows,
        )

    latent_people = cast(list[str], latent["person_id"].to_pylist())
    latent_times = latent["truth_time_ns"].to_numpy(zero_copy_only=False)
    checks += 1
    for person_id in person_ids:
        mask = np.asarray([value == person_id for value in latent_people])
        person_times = latent_times[mask]
        if len(person_times) == 0 or not np.all(np.diff(person_times) == 1_000_000_000):
            _error(
                findings,
                "TIMESTAMP_MONOTONIC",
                "truth_time_ns must increase by exactly one second",
                person_id=person_id,
            )
        elif len(person_times) != duration_sec:
            _error(
                findings,
                "TIMELINE_RANGE",
                "all participants must share the configured timeline length",
                person_id=person_id,
                expected=duration_sec,
                actual=len(person_times),
            )

    checks += 1
    for factor in (
        "autonomic_arousal",
        "motor_activation",
        "cognitive_load",
        "sleep_pressure",
        "sensory_context",
        "recovery_capacity",
        "social_context",
    ):
        bounds = cast(dict[str, float], pc.min_max(latent[factor]).as_py())
        if bounds["min"] < 0 or bounds["max"] > 1:
            _error(
                findings,
                "LATENT_RANGE",
                "latent factor is outside [0,1]",
                factor=factor,
                bounds=bounds,
            )

    event_rows = cast(list[dict[str, object]], events.to_pylist())
    target_counts = Counter(
        str(row.get("person_id"))
        for row in event_rows
        if row.get("is_target") is True
    )
    metrics["target_counts"] = dict(target_counts)
    checks += 1
    if manifest.get("profile") == "quick" and any(
        target_counts[person_id] != 2 for person_id in person_ids
    ):
        _error(
            findings,
            "QUICK_TARGET_COUNT",
            "quick profile requires exactly two targets per participant",
            counts=dict(target_counts),
        )

    checks += 1
    event_types = {str(row.get("event_type")) for row in event_rows}
    if manifest.get("profile") == "quick" and not set(HARD_NEGATIVE_TYPES).issubset(
        event_types
    ):
        _error(
            findings,
            "HARD_NEGATIVE_TYPES",
            "quick run is missing required hard-negative types",
            present=sorted(event_types),
        )
    if any(
        term in event_type.lower()
        for event_type in event_types
        for term in FORBIDDEN_MEDICAL_EVENT_TERMS
    ):
        _error(findings, "NEUTRAL_EVENT_NAMES", "medical event term found")

    checks += 1
    for row in event_rows:
        if row.get("is_target") is True:
            boundary_values = [
                row.get("start_time_ns"),
                row.get("pre_late_start_time_ns"),
                row.get("onset_time_ns"),
                row.get("peak_start_time_ns"),
                row.get("peak_end_time_ns"),
                row.get("recovery_early_end_time_ns"),
                row.get("recovery_late_end_time_ns"),
                row.get("end_time_ns"),
            ]
            if not all(isinstance(value, int) for value in boundary_values):
                _error(
                    findings,
                    "PHASE_ORDER",
                    "target phase boundaries are missing or reversed",
                    event_id=row.get("event_id"),
                )
                continue
            boundaries = cast(list[int], boundary_values)
            if boundaries != sorted(boundaries):
                _error(
                    findings,
                    "PHASE_ORDER",
                    "target phase boundaries are missing or reversed",
                    event_id=row.get("event_id"),
                )

    checks += 1
    minimum_gap_value = run_contract.get("minimum_gap_between_target_sec")
    minimum_gap_sec = (
        minimum_gap_value if isinstance(minimum_gap_value, int) else 0
    )
    targets_by_person: dict[str, list[dict[str, object]]] = {}
    for row in event_rows:
        if row.get("is_target") is True:
            targets_by_person.setdefault(str(row.get("person_id")), []).append(row)
    for person_id, targets in targets_by_person.items():
        targets.sort(
            key=lambda row: _required_int(
                row["onset_time_ns"],
                "onset_time_ns",
            )
        )
        for previous, current in pairwise(targets):
            previous_onset = _required_int(
                previous["onset_time_ns"],
                "onset_time_ns",
            )
            current_onset = _required_int(
                current["onset_time_ns"],
                "onset_time_ns",
            )
            if current_onset - previous_onset < minimum_gap_sec * 1_000_000_000:
                _error(
                    findings,
                    "TARGET_GAP",
                    "target onset gap is below the configured minimum",
                    person_id=person_id,
                    previous_event_id=previous.get("event_id"),
                    current_event_id=current.get("event_id"),
                )
            previous_peak_end = _required_int(
                previous["peak_end_time_ns"],
                "peak_end_time_ns",
            )
            current_peak_start = _required_int(
                current["peak_start_time_ns"],
                "peak_start_time_ns",
            )
            if previous_peak_end > current_peak_start:
                _error(
                    findings,
                    "TARGET_PEAK_OVERLAP",
                    "target peak intervals overlap",
                    person_id=person_id,
                    previous_event_id=previous.get("event_id"),
                    current_event_id=current.get("event_id"),
                )

    checks += 1
    for table_path in REQUIRED_TRUTH_FILES:
        metadata = pq.read_schema(path / table_path).metadata or {}
        if any(
            metadata.get(key) != b"true"
            for key in (b"synthetic", b"non_diagnostic", b"not_for_clinical_use")
        ):
            _error(
                findings,
                "SAFETY_METADATA",
                "truth table is missing safety metadata",
                path=table_path,
            )

    return ValidationReport(
        status="FAIL" if findings else "PASS",
        scope="truth",
        run_id=str(manifest.get("run_id")) if manifest else None,
        checks_run=checks,
        findings=tuple(findings),
        metrics=metrics,
    )
