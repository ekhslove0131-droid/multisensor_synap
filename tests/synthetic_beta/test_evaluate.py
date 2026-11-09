from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from synapbridge_beta_runtime.codec import canonical_bytes

from multisensor_ml.beta_training.evaluate import (
    claim_locked_test,
    locked_claim_path,
    window_metrics,
)
from multisensor_ml.beta_training.evaluation_contract import validate_selected_frozen_inputs
from multisensor_ml.beta_training.evaluation_run import evaluate_beta


def test_locked_eval_cannot_be_repeated(tmp_path: Path) -> None:
    ledger = tmp_path / "locked-use.json"
    first = claim_locked_test(ledger, "a" * 64, "b" * 64)
    assert first["status"] == "LOCKED_TEST_CLAIMED"
    with pytest.raises(ValueError, match="LOCKED_TEST_ALREADY_USED"):
        claim_locked_test(ledger, "a" * 64, "b" * 64)


def test_locked_claim_location_is_bound_to_dataset_not_output_directory(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "prepared" / "manifest.json"
    assert locked_claim_path(dataset) == tmp_path / "prepared" / ".locked-test-use.json"


def test_selected_eval_validates_dataset_and_model_sha_before_claim(tmp_path: Path) -> None:
    dataset = tmp_path / "prepared" / "manifest.json"
    dataset.parent.mkdir()
    dataset.write_bytes(b"dataset")
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    models = [
        {"source_set": source, "head": head, "model_uri": model.as_uri(),
         "model_sha256": hashlib.sha256(b"model").hexdigest()}
        for source in ("watch", "h10", "muse", "watch+h10", "watch+muse",
                       "h10+muse", "watch+h10+muse")
        for head in ("current", "onset60")
    ]
    body: dict[str, object] = {
        "realm": "SYNTHETIC_BETA", "allowed_use": "BETA_ONLY",
        "status": "BETA_SELECTED", "model_role": "BETA_SELECTED",
        "locked_test_used": False,
        "dataset_manifest_sha256": hashlib.sha256(b"dataset").hexdigest(),
        "pattern_models": models,
    }
    bundle = {**body, "content_sha256": hashlib.sha256(canonical_bytes(body)).hexdigest()}
    resolved, digest = validate_selected_frozen_inputs(bundle, dataset)
    assert resolved == dataset.resolve()
    assert digest == body["dataset_manifest_sha256"]
    bundle["dataset_manifest_sha256"] = "0" * 64
    bundle["content_sha256"] = hashlib.sha256(canonical_bytes({
        key: value for key, value in bundle.items() if key != "content_sha256"
    })).hexdigest()
    with pytest.raises(ValueError, match="locked_dataset_sha_mismatch"):
        validate_selected_frozen_inputs(bundle, dataset)


def test_window_metrics_report_imbalance_and_calibration() -> None:
    metrics = window_metrics(
        np.array([0, 0, 1, 1], dtype=np.int64),
        np.array([0.1, 0.2, 0.8, 0.9]),
        0.5,
    )
    assert metrics["class_counts"] == {"negative": 2, "positive": 2}
    assert metrics["auprc"] == pytest.approx(1.0)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["brier"] < 0.05


def test_no_qualified_bundle_does_not_open_locked_dataset(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.json"
    bundle.write_text(
        '{"realm":"SYNTHETIC_BETA","allowed_use":"BETA_ONLY",'
        '"status":"NO_QUALIFIED_BETA_MODEL","locked_test_used":false}'
    )
    output = tmp_path / "evaluation"
    receipt = evaluate_beta(bundle, tmp_path / "missing-locked-dataset.json", output)
    assert receipt["status"] == "NO_QUALIFIED_BETA_MODEL"
    assert receipt["locked_use"] == {"status": "NOT_ACCESSED", "use_count": 0}
    assert not (output / "locked-use.json").exists()
    assert not (tmp_path / ".locked-test-use.json").exists()
