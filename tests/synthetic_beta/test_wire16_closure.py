from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

from synapbridge_beta_runtime.codec import canonical_bytes

from multisensor_ml.beta_training.wire16_closure import finalize_wire16_compatibility


def test_wire16_closure_preserves_no_qualified_boundary(tmp_path: Path) -> None:
    root = Path.cwd()
    apps = Path(
        "/Users/baital/dev/apps/.worktrees/kidsignal-gcs-realtime/"
        "kidsignal-android-suite"
    )
    pin = apps.parent / ".superpowers/sdd/2026-10-06-synthetic-beta-loop/" \
        "wire16-current-candidate.md"
    failure = root / "artifacts/beta-loop-v1/manifest.json"
    previous = root / (
        "artifacts/beta-loop-v1-pre-round3-no-qualified-609bbf/manifest.json"
    )
    failure_before = hashlib.sha256(failure.read_bytes()).hexdigest()
    previous_before = hashlib.sha256(previous.read_bytes()).hexdigest()
    wheel = _runtime_wheel(tmp_path)

    receipt = finalize_wire16_compatibility(
        round3_root=root / "artifacts/beta-loop-v1-round3",
        wire_root=apps,
        wire_pin_path=pin,
        failure_manifest=failure,
        wire16_runtime_wheel=wheel,
        output=tmp_path / "closure",
    )

    assert receipt["status"] == "NO_QUALIFIED_BETA_MODEL"
    assert receipt["closure_status"] == (
        "WIRE16_COMPATIBLE_PARTIAL_ARTIFACT_NOT_QUALIFIED"
    )
    assert receipt["model_role"] is None
    assert receipt["delivery_eligible"] is False
    assert receipt["promotion_eligible"] is False
    assert receipt["locked_test_use_count"] == 0
    assert receipt["recovery_fit_count"] == 0
    assert receipt["wire_ref_count"] == 16
    assert receipt["partial_model_file_count"] == 14
    assert receipt["qualification_blockers"] == [
        "ROUND3_IMMUTABLE_MANIFEST_NOT_CREATED",
        "ORIGINAL_PER_MODEL_EXPORT_RECEIPTS_NOT_PERSISTED",
        "ORIGINAL_VALIDATION_SELECTION_EVIDENCE_NOT_PERSISTED",
    ]
    assert receipt["source_failure_sha256"] == failure_before
    assert receipt["previous_failure_sha256"] == previous_before
    wheel_path = Path(unquote(urlparse(receipt["wire16_runtime_wheel_ref"]["uri"]).path))
    assert wheel_path.parent == tmp_path / "closure/runtime"
    assert hashlib.sha256(wheel_path.read_bytes()).hexdigest() == (
        receipt["wire16_runtime_wheel_ref"]["sha256"]
    )
    assert not list(root.rglob(".locked-test-use.json"))
    assert hashlib.sha256(failure.read_bytes()).hexdigest() == failure_before
    assert hashlib.sha256(previous.read_bytes()).hexdigest() == previous_before

    path = tmp_path / "closure/compatibility_receipt.json"
    assert json.loads(path.read_bytes()) == receipt
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    assert hashlib.sha256(canonical_bytes(body)).hexdigest() == receipt["content_sha256"]


def test_wire16_closure_rejects_mutated_pin(tmp_path: Path) -> None:
    root = Path.cwd()
    apps = Path(
        "/Users/baital/dev/apps/.worktrees/kidsignal-gcs-realtime/"
        "kidsignal-android-suite"
    )
    source = apps.parent / ".superpowers/sdd/2026-10-06-synthetic-beta-loop/" \
        "wire16-current-candidate.md"
    pin = tmp_path / "wire16.md"
    pin.write_bytes(source.read_bytes().replace(b"46d587fe", b"06d587fe", 1))

    try:
        finalize_wire16_compatibility(
            round3_root=root / "artifacts/beta-loop-v1-round3",
            wire_root=apps,
            wire_pin_path=pin,
            failure_manifest=root / "artifacts/beta-loop-v1/manifest.json",
            wire16_runtime_wheel=_runtime_wheel(tmp_path),
            output=tmp_path / "closure",
        )
    except ValueError as error:
        assert str(error) == "wire16_pin_contract_invalid"
    else:
        raise AssertionError("mutated wire pin was accepted")


def _runtime_wheel(tmp_path: Path) -> Path:
    source = Path("beta_runtime/src/synapbridge_beta_runtime")
    wheel = tmp_path / "synapbridge_beta_runtime-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for path in sorted(source.glob("*.py")):
            archive.write(path, f"synapbridge_beta_runtime/{path.name}")
    return wheel
