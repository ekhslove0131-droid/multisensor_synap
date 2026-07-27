from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from multisensor_ml.cli import main
from multisensor_ml.factory import build_outcome_artifacts
from multisensor_ml.outcomes import (
    BEHAVIOR_CODES,
    build_behavior_labels,
    build_stage_labels,
    load_behavior_ontology,
)
from multisensor_ml.receipts import create_stage_receipt, validate_stage_receipt
from multisensor_ml.settings import load_factory_config


def test_stage_labels_keep_no_event_separate_from_low() -> None:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    timeline = pd.DataFrame(
        {
            "run_id": ["run-1"] * 9,
            "person_id": ["P001"] * 9,
            "timestamp_utc": pd.date_range(start, periods=9, freq="s"),
        }
    )
    events = pd.DataFrame(
        [
            {
                "run_id": "run-1",
                "person_id": "P001",
                "event_id": "E001",
                "is_target": True,
                "start_time_ns": start.value + 1_000_000_000,
                "pre_late_start_time_ns": start.value + 2_000_000_000,
                "onset_time_ns": start.value + 3_000_000_000,
                "peak_start_time_ns": start.value + 4_000_000_000,
                "peak_end_time_ns": start.value + 5_000_000_000,
                "recovery_early_end_time_ns": start.value + 6_000_000_000,
                "recovery_late_end_time_ns": start.value + 7_000_000_000,
                "end_time_ns": start.value + 8_000_000_000,
            }
        ]
    )

    result = build_stage_labels(timeline, events)

    assert result["stage_code"].tolist() == [
        "NO_EVENT",
        "LOW",
        "MEDIUM",
        "MEDIUM",
        "HIGH",
        "DECREASING",
        "RECOVERY",
        "RECOVERY",
        "NO_EVENT",
    ]
    assert result["event_id"].tolist() == [
        None,
        "E001",
        "E001",
        "E001",
        "E001",
        "E001",
        "E001",
        "E001",
        None,
    ]


def test_behavior_labels_are_deterministic_and_use_only_neutral_codes() -> None:
    events = pd.DataFrame(
        [
            {
                "run_id": "run-1",
                "person_id": "P001",
                "event_id": "T001",
                "is_target": True,
                "hard_negative_kind": None,
                "intensity_truth": 0.8,
                "archetype": "motor_first",
                "context": "focused_task",
            },
            {
                "run_id": "run-1",
                "person_id": "P001",
                "event_id": "H001",
                "is_target": False,
                "hard_negative_kind": "ordinary_physical_activity",
                "intensity_truth": 0.2,
                "archetype": "partial_response",
                "context": "light_activity",
            },
        ]
    )

    first_behaviors, first_tags = build_behavior_labels(events, random_state=17)
    second_behaviors, second_tags = build_behavior_labels(events, random_state=17)

    pd.testing.assert_frame_equal(first_behaviors, second_behaviors)
    pd.testing.assert_frame_equal(first_tags, second_tags)
    assert not first_behaviors.empty
    assert set(first_behaviors["behavior_code"]).issubset(BEHAVIOR_CODES)
    assert set(first_tags["review_tag"]).issubset(
        {"reported_meltdown_like", "reported_sensory_seeking_like"}
    )
    assert "intensity_truth" not in first_behaviors.columns
    assert "archetype" not in first_behaviors.columns


def test_receipt_validation_rejects_tampered_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "outcome-manifest.json"
    artifact.write_text(json.dumps({"schema": "goal1.5/behavior-outcomes/v1"}))
    receipt_path = tmp_path / "receipt.json"
    create_stage_receipt(
        pipeline_run_id="factory-run-1",
        stage_id="labels",
        artifact_uri=artifact,
        output_path=receipt_path,
        status="SUCCESS",
        versions={"label_version": "labels-v1"},
        message_ko="라벨 생성 완료",
    )

    receipt = validate_stage_receipt(receipt_path)
    assert receipt.stage_id == "labels"
    assert receipt.status == "SUCCESS"
    assert (
        main(["factory", "validate", "--receipt", str(receipt_path)])
        == 0
    )

    artifact.write_text(json.dumps({"schema": "tampered"}))
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        validate_stage_receipt(receipt_path)


def test_factory_config_resolves_all_paths_from_yaml_location(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "factory.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "goal1.5/synthetic-factory/v1",
                "series_id": "factory-quick",
                "generator_config": "../generator/quick.yaml",
                "behavior_ontology": "../behavior.yaml",
                "seeds": [11, 12],
                "generator_overrides": {"run": {"participant_count": 12}},
                "data_root": "../data",
                "outcome_root": "../outcomes",
                "random_state": 17,
            }
        )
    )

    config = load_factory_config(config_path)

    assert config.generator_config == (tmp_path / "generator" / "quick.yaml").resolve()
    assert config.behavior_ontology == (tmp_path / "behavior.yaml").resolve()
    assert config.data_root == (tmp_path / "configs" / "../data").resolve()
    assert config.outcome_root == (tmp_path / "configs" / "../outcomes").resolve()


def test_behavior_ontology_rejects_summary_tags_as_training_codes(
    tmp_path: Path,
) -> None:
    ontology = tmp_path / "behavior.yaml"
    ontology.write_text(
        yaml.safe_dump(
            {
                "schema_version": "goal1.5/behavior-ontology/v1",
                "behavior_codes": ["reported_meltdown_like"],
                "review_tags": ["reported_meltdown_like"],
            }
        )
    )

    with pytest.raises(ValueError, match="review terms are forbidden behavior codes"):
        load_behavior_ontology(ontology)


def test_outcome_factory_writes_immutable_artifacts_and_reuses_matching_input(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run-1"
    truth = run / "truth"
    truth.mkdir(parents=True)
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    timeline = pd.DataFrame(
        {
            "run_id": ["run-1"] * 10,
            "person_id": ["P001"] * 10,
            "timestamp_utc": pd.date_range(start, periods=10, freq="s"),
            "context_state": ["focused_task"] * 10,
        }
    )
    events = pd.DataFrame(
        [
            {
                "run_id": "run-1",
                "person_id": "P001",
                "event_id": "T001",
                "event_type": "multimodal_arousal_episode",
                "is_target": True,
                "hard_negative_kind": None,
                "start_time_ns": start.value + 1_000_000_000,
                "pre_late_start_time_ns": start.value + 2_000_000_000,
                "onset_time_ns": start.value + 3_000_000_000,
                "peak_start_time_ns": start.value + 4_000_000_000,
                "peak_end_time_ns": start.value + 5_000_000_000,
                "recovery_early_end_time_ns": start.value + 6_000_000_000,
                "recovery_late_end_time_ns": start.value + 7_000_000_000,
                "end_time_ns": start.value + 8_000_000_000,
                "intensity_truth": 0.8,
                "archetype": "motor_first",
            }
        ]
    )
    pq.write_table(
        pa.Table.from_pandas(timeline, preserve_index=False),
        truth / "latent_timeline.parquet",
    )
    pq.write_table(
        pa.Table.from_pandas(events, preserve_index=False),
        truth / "events.parquet",
    )
    registry = tmp_path / "registry"
    registry.mkdir()
    (registry / "manifest.json").write_text(
        json.dumps(
            {
                "series_id": "factory-fixture",
                "status": "oracle/sanity",
                "runs": [{"run_id": "run-1", "source_path": str(run)}],
            }
        )
    )

    first = build_outcome_artifacts(registry, tmp_path / "outcomes", random_state=17)
    second = build_outcome_artifacts(registry, tmp_path / "outcomes", random_state=17)

    assert first.status == "SUCCESS"
    assert second.status == "REUSED"
    manifest = json.loads(first.manifest_json.read_text())
    assert manifest["schema_version"] == "goal1.5/behavior-outcomes/v1"
    assert manifest["synchronization_status"] == "NOT_AVAILABLE_TRUTH_ONLY"
    assert set(manifest["files"]) == {
        "outcome_behaviors.parquet",
        "outcome_events.parquet",
        "outcome_review_tags.parquet",
        "outcome_stages.parquet",
    }
    stages = pq.read_table(first.root / "outcome_stages.parquet").to_pandas()
    assert stages.loc[stages["stage_code"] == "LOW", "event_id"].tolist() == ["T001"]

    (first.root / "outcome_behaviors.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="immutable outcome artifact hash mismatch"):
        build_outcome_artifacts(registry, tmp_path / "outcomes", random_state=17)
