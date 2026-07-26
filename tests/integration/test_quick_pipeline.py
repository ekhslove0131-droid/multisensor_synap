from __future__ import annotations

import json
import socket
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from multisensor_synth.orchestration.pipeline import generate_truth_run
from multisensor_synth.validation.invariants import validate_truth_run

ROOT = Path(__file__).parents[2]


def test_quick_truth_only_end_to_end(tmp_path: Path) -> None:
    generated = generate_truth_run(ROOT / "configs" / "quick.yaml", tmp_path)
    run_dir = generated.run_dir

    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "config_snapshot" / "source.yaml").is_file()
    assert (run_dir / "config_snapshot" / "resolved.yaml").is_file()
    assert (run_dir / "truth" / "participants.parquet").is_file()
    assert (run_dir / "truth" / "daily_context.parquet").is_file()
    assert (run_dir / "truth" / "latent_timeline.parquet").is_file()
    assert (run_dir / "truth" / "events.parquet").is_file()
    assert (run_dir / "reports" / "truth_validation_report.json").is_file()
    assert not (run_dir / "observed").exists()
    assert not (run_dir / "model_ready").exists()
    assert not (run_dir / "clips").exists()

    participants = pq.read_table(run_dir / "truth" / "participants.parquet")
    latent = pq.read_table(run_dir / "truth" / "latent_timeline.parquet")
    events = pq.read_table(run_dir / "truth" / "events.parquet")
    assert participants.num_rows == 3
    assert latent.num_rows == 64_800
    target_rows = events.filter(events["is_target"]).num_rows
    assert target_rows == 6
    assert events.schema.names == [
        "run_id",
        "person_id",
        "event_id",
        "event_type",
        "is_target",
        "hard_negative_kind",
        "start_time_ns",
        "pre_late_start_time_ns",
        "onset_time_ns",
        "peak_start_time_ns",
        "peak_end_time_ns",
        "recovery_early_end_time_ns",
        "recovery_late_end_time_ns",
        "end_time_ns",
        "intensity_truth",
        "archetype",
        "is_artifact_only_schedule",
        "seed",
    ]

    report = json.loads(
        (run_dir / "reports" / "truth_validation_report.json").read_text()
    )
    assert report["status"] == "PASS"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["run_contract"] == {
        "canonical_rate_hz": 1,
        "duration_sec": 21_600,
        "minimum_gap_between_target_sec": 1800,
        "participant_count": 3,
    }
    assert validate_truth_run(run_dir).status == "PASS"


def test_quick_logical_content_is_reproducible_across_output_roots(
    tmp_path: Path,
) -> None:
    first = generate_truth_run(ROOT / "configs" / "quick.yaml", tmp_path / "first")
    second = generate_truth_run(ROOT / "configs" / "quick.yaml", tmp_path / "second")

    first_manifest = json.loads((first.run_dir / "manifest.json").read_text())
    second_manifest = json.loads((second.run_dir / "manifest.json").read_text())
    assert first_manifest["config_sha256"] == second_manifest["config_sha256"]
    assert first_manifest["child_seed_namespaces"] == second_manifest[
        "child_seed_namespaces"
    ]
    assert first_manifest["tables"] == second_manifest["tables"]


def test_validation_rejects_overlapping_target_peaks(tmp_path: Path) -> None:
    generated = generate_truth_run(ROOT / "configs" / "quick.yaml", tmp_path)
    events_path = generated.run_dir / "truth" / "events.parquet"
    events = pq.read_table(events_path)
    rows = events.to_pylist()
    targets = [
        row
        for row in rows
        if row["person_id"] == "P001" and row["is_target"]
    ]
    targets.sort(key=lambda row: row["onset_time_ns"])
    first, second = targets
    shift_ns = first["onset_time_ns"] + 10_000_000_000 - second["onset_time_ns"]
    boundary_keys = (
        "start_time_ns",
        "pre_late_start_time_ns",
        "onset_time_ns",
        "peak_start_time_ns",
        "peak_end_time_ns",
        "recovery_early_end_time_ns",
        "recovery_late_end_time_ns",
        "end_time_ns",
    )
    for key in boundary_keys:
        second[key] += shift_ns
    pq.write_table(pa.Table.from_pylist(rows, schema=events.schema), events_path)

    report = validate_truth_run(generated.run_dir)
    finding_codes = {finding.code for finding in report.findings}

    assert report.status == "FAIL"
    assert "TARGET_GAP" in finding_codes
    assert "TARGET_PEAK_OVERLAP" in finding_codes


def test_validation_rejects_logical_content_hash_mismatch(tmp_path: Path) -> None:
    generated = generate_truth_run(ROOT / "configs" / "quick.yaml", tmp_path)
    participants_path = generated.run_dir / "truth" / "participants.parquet"
    participants = pq.read_table(participants_path)
    rows = participants.to_pylist()
    rows[0]["age_years"] += 0.25
    pq.write_table(
        pa.Table.from_pylist(rows, schema=participants.schema),
        participants_path,
    )

    report = validate_truth_run(generated.run_dir)

    assert report.status == "FAIL"
    assert "LOGICAL_HASH" in {finding.code for finding in report.findings}


def test_quick_generation_does_not_open_network_connections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_network(*_: object, **__: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", reject_network)
    monkeypatch.setattr(socket.socket, "connect", reject_network)

    generated = generate_truth_run(ROOT / "configs" / "quick.yaml", tmp_path)

    assert generated.report.status == "PASS"
