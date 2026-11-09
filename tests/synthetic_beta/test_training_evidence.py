from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from synapbridge_beta_runtime.codec import canonical_bytes

from multisensor_ml.beta_training.artifacts import sha256
from multisensor_ml.beta_training.training_evidence import (
    complete_training_evidence,
    write_model_evidence,
    write_policy_evidence,
)

SOURCES = (
    "watch", "h10", "muse", "watch+h10", "watch+muse", "h10+muse",
    "watch+h10+muse",
)


def test_model_evidence_survives_later_finalization_failure(tmp_path: Path) -> None:
    lineage = _lineage()
    model = _model(tmp_path, "watch", "current")
    ref = write_model_evidence(tmp_path, model, lineage)
    assert Path(ref["uri"].removeprefix("file://")).is_file()
    with pytest.raises(ValueError, match="training_evidence_set_incomplete"):
        complete_training_evidence(tmp_path, lineage, [ref], None)
    assert Path(ref["uri"].removeprefix("file://")).is_file()
    with pytest.raises(FileExistsError, match="training_evidence_exists"):
        write_model_evidence(tmp_path, model, lineage)


def test_complete_evidence_is_exactly_fourteen_models_and_policy(tmp_path: Path) -> None:
    lineage = _lineage()
    expected, refs = [], []
    for source in SOURCES:
        for head in ("current", "onset60"):
            model = _model(tmp_path, source, head)
            refs.append(write_model_evidence(tmp_path, model, lineage))
            expected.append(model)
    policy = {
        "version": "synthetic_beta_stage_policy_v1",
        "candidate_grid": [1.0, 1.5, 2.0],
        "d_on": 1.0,
        "d_off": 0.5,
    }
    policy_ref = write_policy_evidence(tmp_path, policy, lineage)
    models, recovered_policy = complete_training_evidence(
        tmp_path, lineage, refs, policy_ref,
    )
    assert models == expected
    assert recovered_policy == policy

    model_path = Path(expected[0]["model_uri"].removeprefix("file://"))
    model_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="model_evidence_artifact_mismatch"):
        complete_training_evidence(tmp_path, lineage, refs, policy_ref)


def test_evidence_rejects_nested_collision_and_mutated_policy(tmp_path: Path) -> None:
    lineage = _lineage()
    refs = [
        write_model_evidence(tmp_path, _model(tmp_path, source, head), lineage)
        for source in SOURCES for head in ("current", "onset60")
    ]
    policy = {"version": "synthetic_beta_stage_policy_v1",
              "candidate_grid": [1.0, 1.5, 2.0], "d_on": 1.0, "d_off": 0.5}
    policy_ref = write_policy_evidence(tmp_path, policy, lineage)
    nested = tmp_path / "evidence/models/watch/h10/current.json"
    nested.parent.mkdir(parents=True)
    nested.write_bytes((tmp_path / "evidence/models/watch+h10/current.json").read_bytes())
    with pytest.raises(ValueError, match="training_evidence_set_incomplete"):
        complete_training_evidence(tmp_path, lineage, refs, policy_ref)
    nested.unlink()

    policy_path = Path(policy_ref["uri"].removeprefix("file://"))
    value = json.loads(policy_path.read_bytes())
    value["selection_role"] = "LOCKED_TEST"
    value["policy"] = {"version": "bad", "candidate_grid": [9.0],
                       "d_on": 9.0, "d_off": 8.0}
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    value["content_sha256"] = sha256(canonical_bytes(body))
    policy_path.write_bytes(canonical_bytes(value))
    policy_ref["sha256"] = sha256(policy_path.read_bytes())
    with pytest.raises(
        ValueError, match=r"policy_evidence_invalid|training_evidence_lineage_mismatch",
    ):
        complete_training_evidence(tmp_path, lineage, refs, policy_ref)


def test_complete_evidence_accepts_canonical_relative_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    output = Path("relative-output")
    lineage = _lineage()
    refs = [
        write_model_evidence(output, _model(output, source, head), lineage)
        for source in SOURCES for head in ("current", "onset60")
    ]
    policy = {"version": "synthetic_beta_stage_policy_v1",
              "candidate_grid": [1.0, 1.5, 2.0], "d_on": 1.0, "d_off": 0.5}
    policy_ref = write_policy_evidence(output, policy, lineage)
    models, recovered = complete_training_evidence(output, lineage, refs, policy_ref)
    assert len(models) == 14
    assert recovered == policy


def _lineage() -> dict[str, object]:
    digest = "a" * 64
    return {
        "dataset_manifest_sha256": digest,
        "training_config_sha256": digest,
        "generation_protocol_sha256": digest,
        "kernel_sha256": digest,
        "runtime_lock_sha256": digest,
        "standard_refs_sha256": digest,
        "personal_reference_sha256": digest,
        "calibration_bounds_ms": [1_000, 1_801_000],
    }


def _model(tmp_path: Path, source: str, head: str) -> dict[str, object]:
    path = tmp_path / "models" / source / f"{head}.onnx"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"{source}:{head}".encode())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "source_set": source,
        "head": head,
        "algorithm": "HGB",
        "threshold": 0.5,
        "validation_sensitivity": 0.75,
        "validation_far_per_hour": 0.5,
        "validation_median_lead_seconds": 20.0 if head == "onset60" else -10.0,
        "feature_schema_sha256": "b" * 64,
        "feature_names": ["feature"],
        "model_uri": path.resolve().as_uri(),
        "model_sha256": digest,
        "onnx_receipt": {
            "sha256": digest,
            "max_absolute_error": 1e-7,
            "decision_mismatch_count": 0,
            "class_order": [0, 1],
        },
    }
