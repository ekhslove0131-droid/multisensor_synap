import json
from pathlib import Path

import pytest

from multisensor_ml.kaggle_model_package import (
    collect_hierarchical_payload,
    verify_payload,
)


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
