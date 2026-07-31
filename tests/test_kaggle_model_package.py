import json
from pathlib import Path

import pytest

from multisensor_ml.kaggle_model_package import (
    build_kaggle_model_package,
    collect_hierarchical_payload,
    verify_kaggle_model_package,
    verify_payload,
)
from multisensor_ml.registry import sha256_file


def test_collected_payload_has_every_hierarchical_head(
    package_config, built_wheel: Path, tmp_path: Path
) -> None:
    manifest = collect_hierarchical_payload(
        package_config, tmp_path / "payload", built_wheel
    )

    assert manifest["status"] == "oracle/sanity"
    assert manifest["real_data_status"] == "NOT VERIFIED"
    assert manifest["selected_event_model"] == "hist_gradient_boosting"
    assert manifest["selected_stage_model"] == "hist_gradient_boosting"
    assert len(manifest["selected_behavior_models"]) == 10
    assert manifest["standard_types"] == ["STD-A"]
    assert manifest["selected_k"] == 1


def test_payload_has_no_unsafe_pickle_extensions(
    package_config, built_wheel: Path, tmp_path: Path
) -> None:
    root = tmp_path / "payload"
    collect_hierarchical_payload(package_config, root, built_wheel)
    forbidden = {".pkl", ".pickle", ".joblib"}
    assert not [path for path in root.rglob("*") if path.suffix in forbidden]


def test_verify_payload_rejects_tampered_model(
    package_config, built_wheel: Path, tmp_path: Path
) -> None:
    root = tmp_path / "payload"
    collect_hierarchical_payload(package_config, root, built_wheel)
    model = next(root.rglob("*.skops"))
    model.write_bytes(model.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="payload hash mismatch"):
        verify_payload(root)


def test_verify_payload_checks_declared_unknown_types(
    package_config, built_wheel: Path, tmp_path: Path
) -> None:
    root = tmp_path / "payload"
    collect_hierarchical_payload(package_config, root, built_wheel)
    manifest_path = root / "payload_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["models"][0]["unknown_types"] = ["unsafe.Type"]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="unknown type"):
        verify_payload(root)


def test_built_package_contains_reproducible_validation_sample(
    package_config, built_wheel: Path
) -> None:
    package = build_kaggle_model_package(package_config, built_wheel)

    assert package.archive.name == "model_payload.tar.gz"
    assert package.manifest.name == "model_manifest.json"
    verified = verify_kaggle_model_package(package.root)
    assert verified["sample_split_role"] == "validation"
    assert verified["locked_test_read"] is False
    assert verified["sample_rows"] == 60
    assert verified["reproduction_status"] == "REPRODUCED"


def test_repeated_builds_have_identical_archive_hash(
    package_config, built_wheel: Path, tmp_path: Path
) -> None:
    first = build_kaggle_model_package(
        package_config.model_copy(update={"output_root": tmp_path / "a"}), built_wheel
    )
    second = build_kaggle_model_package(
        package_config.model_copy(update={"output_root": tmp_path / "b"}), built_wheel
    )

    assert sha256_file(first.archive) == sha256_file(second.archive)


def test_outer_verifier_rejects_archive_tamper(
    package_config, built_wheel: Path
) -> None:
    package = build_kaggle_model_package(package_config, built_wheel)
    package.archive.write_bytes(package.archive.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="outer package hash mismatch"):
        verify_kaggle_model_package(package.root)
