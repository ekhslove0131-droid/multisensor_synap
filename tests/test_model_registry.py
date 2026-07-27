from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from multisensor_ml.model_registry import ModelRegistry, import_oracle_bundle
from multisensor_ml.receipts import create_stage_receipt
from multisensor_ml.settings import load_training_registry_config


def _register_release_parts(registry: ModelRegistry, suffix: str) -> dict[str, str]:
    versions: dict[str, str] = {}
    for kind in (
        "dataset",
        "label_set",
        "baseline",
        "standard_type",
        "stage_model",
        "behavior_model",
        "router",
    ):
        version_id = f"{kind}-{suffix}"
        registry.register_version(
            kind=kind,
            version_id=version_id,
            artifact_uri=f"/artifacts/{version_id}.json",
            artifact_sha256=(suffix * 64)[:64],
            metadata={"suffix": suffix},
        )
        versions[kind] = version_id
    return versions


def test_registry_rejects_mutating_an_existing_version(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry.sqlite")
    registry.initialize()
    registry.register_version(
        kind="dataset",
        version_id="dataset-v1",
        artifact_uri="/artifacts/dataset-v1.json",
        artifact_sha256="a" * 64,
        metadata={"source_domain": "synthetic_truth_oracle"},
    )

    with pytest.raises(ValueError, match="immutable version conflict"):
        registry.register_version(
            kind="dataset",
            version_id="dataset-v1",
            artifact_uri="/artifacts/dataset-v2.json",
            artifact_sha256="b" * 64,
            metadata={"source_domain": "synthetic_truth_oracle"},
        )


def test_promoting_a_release_retires_the_previous_champion(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry.sqlite")
    registry.initialize()
    first = _register_release_parts(registry, "a")
    second = _register_release_parts(registry, "b")
    registry.create_release(
        release_id="release-a",
        versions=first,
        source_domain="synthetic_truth_oracle",
    )
    registry.create_release(
        release_id="release-b",
        versions=second,
        source_domain="synthetic_truth_oracle",
    )

    registry.promote_release("release-a", audit_reason="first validated release")
    registry.promote_release("release-b", audit_reason="second validated release")

    assert registry.release_status("release-a") == "retired"
    assert registry.release_status("release-b") == "champion"


def test_locked_test_registration_rejects_duplicate_model_dataset_pair(
    tmp_path: Path,
) -> None:
    registry = ModelRegistry(tmp_path / "registry.sqlite")
    registry.initialize()
    registry.record_locked_test(
        model_sha256="a" * 64,
        dataset_sha256="b" * 64,
        audit_reason="initial locked acceptance",
    )

    with pytest.raises(ValueError, match="duplicate locked_test"):
        registry.record_locked_test(
            model_sha256="a" * 64,
            dataset_sha256="b" * 64,
            audit_reason="attempted duplicate acceptance",
        )
    assert registry.locked_test_exists(
        model_sha256="a" * 64,
        dataset_sha256="b" * 64,
    )


def test_import_oracle_bundle_creates_an_explicit_legacy_candidate(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    baseline = bundle / "global_baseline.json"
    baseline.write_text("{}")
    manifest = {
        "bundle_schema": "goal1.5/oracle-model-bundle/v1",
        "status": "oracle/sanity",
        "real_data_status": "NOT VERIFIED",
        "lineage": {
            "prepared_series_id": "quick12-oracle-v1",
            "prepared_manifest_sha256": "a" * 64,
            "prepared_root": "/prepared/quick12-oracle-v1",
        },
        "files": {
            "global_baseline.json": hashlib.sha256(b"{}").hexdigest()
        },
        "models": [],
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    registry = ModelRegistry(tmp_path / "registry.sqlite")
    registry.initialize()

    release_id = import_oracle_bundle(registry, bundle)

    assert release_id.startswith("oracle-legacy-quick12-oracle-v1-")
    assert registry.release_status(release_id) == "candidate"


def test_training_registry_config_resolves_runtime_paths(tmp_path: Path) -> None:
    config = tmp_path / "configs" / "registry.yaml"
    config.parent.mkdir()
    config.write_text(
        """
schema_version: goal1.5/training-registry-config/v1
registry_path: ../runtime/goal15.sqlite
artifact_root: ../artifacts
outcome_root: ../outcomes
data_root: ../data
series_id: quick12-oracle-v1
run_id: training-quick12-v1
input_receipt: ../outcomes/quick12-oracle-v1/factory-receipt.json
random_state: 17
korean_router_version: ko-v1
""".lstrip()
    )

    loaded = load_training_registry_config(config)

    assert loaded.registry_path == (config.parent / "../runtime/goal15.sqlite").resolve()
    assert loaded.artifact_root == (config.parent / "../artifacts").resolve()
    assert loaded.outcome_root == (config.parent / "../outcomes").resolve()
    assert loaded.data_root == (config.parent / "../data").resolve()
    assert loaded.input_receipt == (
        config.parent / "../outcomes/quick12-oracle-v1/factory-receipt.json"
    ).resolve()


def test_registry_records_a_validated_stage_receipt_once(tmp_path: Path) -> None:
    artifact = tmp_path / "manifest.json"
    artifact.write_text("{}")
    receipt_path = tmp_path / "receipt.json"
    create_stage_receipt(
        pipeline_run_id="run-1",
        stage_id="labels",
        artifact_uri=artifact,
        output_path=receipt_path,
        status="SUCCESS",
        versions={"label_version": "labels-v1"},
        message_ko="라벨 생성 완료",
    )
    registry = ModelRegistry(tmp_path / "registry.sqlite")
    registry.initialize()

    assert registry.record_stage_receipt(receipt_path) == "CREATED"
    assert registry.record_stage_receipt(receipt_path) == "REUSED"
