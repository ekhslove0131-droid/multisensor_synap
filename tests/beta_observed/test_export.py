from __future__ import annotations

import json
from pathlib import Path

import pytest
from synapbridge_beta_runtime.types import local_path

from multisensor_synth.beta_observed.exporter import generate_observed, iter_native_parts

from .fixtures import beta_config


def test_export_separates_observed_and_hidden_truth(tmp_path: Path) -> None:
    output = tmp_path / "beta"
    manifest = generate_observed(beta_config(), output)

    training = json.loads(manifest.manifest_path.read_bytes())
    observed = json.loads(local_path(training["observed_manifest"]["uri"]).read_bytes())
    assert set(observed) == {
        "schema_version", "realm", "beta_run_uuid", "run_epoch", "config_sha256",
        "protocol_sha256", "subjects", "parts", "content_sha256",
    }
    assert "truth" not in json.dumps(observed).lower()
    assert training["truth_artifact"]["uri"].endswith("truth/events.jsonl")
    assert training["protocol_ref"]["uri"].endswith("protocol.json")
    assert len(training["protocol_ref"]["sha256"]) == 64
    protocol = json.loads(local_path(training["protocol_ref"]["uri"]).read_bytes())
    assert protocol["noise"]["label_dependent"] is False
    assert protocol["morphology"]["neurokit_version"] == "0.2.13"
    assert protocol["lags_ms"]["ppg"] == 200
    assert protocol["population"]["response_assignment"] == "subject_index_mod_4"
    assert protocol["event_schedule"]["duration_seconds"] == 180
    assert len(protocol["generator_source_sha256"]) == 64
    assert observed["beta_run_uuid"] == training["beta_run_uuid"]
    truth = [json.loads(line) for line in manifest.truth_path.read_text().splitlines()]
    assert all(
        row["pre_early_start_ms"] < row["pre_late_start_ms"] < row["onset_ms"]
        < row["offset_ms"] == row["recovery_early_start_ms"]
        < row["recovery_late_start_ms"] < row["phase_end_ms"]
        for row in truth
    )
    assert len(tuple(iter_native_parts(manifest))) == 2 * 3 * 3


def test_export_is_create_only(tmp_path: Path) -> None:
    output = tmp_path / "beta"
    generate_observed(beta_config(), output)
    with pytest.raises(FileExistsError):
        generate_observed(beta_config(), output)
