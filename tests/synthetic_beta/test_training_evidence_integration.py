from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from multisensor_ml.beta_training.artifacts import sha256
from multisensor_ml.beta_training.pipeline import BetaTrainConfig
from multisensor_ml.beta_training.training_run import train_beta


def test_training_run_preserves_evidence_before_wire_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset.json"
    dataset.write_bytes(b"dataset")
    wheel = tmp_path / "runtime.whl"
    wheel.write_bytes(b"wheel")
    output = tmp_path / "output"
    config = BetaTrainConfig()
    protocol_sha, reference_sha = "c" * 64, "d" * 64
    standard_refs = [
        {"family": family, "uri": f"file:///{family}.json", "sha256": "e" * 64}
        for family in ("watch", "h10", "muse")
    ]
    references = {f"subject-{index}": {} for index in range(30)}
    cohort = SimpleNamespace(manifest={
        "config": {"prefill_seconds": 300, "calibration_seconds": 1800},
        "timeline_origin_ms": 1_000,
        "protocol_ref": {"sha256": protocol_sha},
        "beta_run_uuid": "10000000-0000-4000-8000-000000000001",
        "run_epoch": 1,
    })
    fit_calls = 0

    class Estimator:
        def predict_proba(self, values: np.ndarray) -> np.ndarray:
            return np.tile(np.asarray([[0.25, 0.75]]), (len(values), 1))

    def fake_candidates(
        values: np.ndarray, labels: np.ndarray, source: str, head: str,
    ) -> list[SimpleNamespace]:
        nonlocal fit_calls
        fit_calls += 2
        return [SimpleNamespace(
            algorithm=algorithm, estimator=Estimator(), source_set=source,
            head=head, input_width=values.shape[1],
        ) for algorithm in ("LR", "HGB")]

    def fake_samples(*_args: object) -> SimpleNamespace:
        return SimpleNamespace(
            x=np.ones((4, 2), dtype=np.float32),
            y=np.asarray([0, 1, 0, 1], dtype=np.int64),
            subjects=("a", "a", "b", "b"),
            decisions_ms=np.asarray([10_000, 20_000, 10_000, 20_000]),
            event_ranges={"a": ((10_000, 20_000),), "b": ((10_000, 20_000),)},
            schema_sha="b" * 64, feature_names=("one", "two"),
            deviations=np.ones(4), temporal_stages=np.asarray([1, 2, 1, 2]),
        )

    def fake_selection(
        candidates: list[SimpleNamespace], *_args: object, **_kwargs: object,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            candidate=candidates[1], algorithm="HGB", threshold=0.5,
            sensitivity=0.75, far_per_hour=0.5, median_lead_seconds=20.0,
        )

    def fake_export(
        candidate: SimpleNamespace, path: Path, *_args: object, **_kwargs: object,
    ) -> dict[str, object]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{candidate.source_set}:{candidate.head}".encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return {"sha256": digest, "max_absolute_error": 1e-7,
                "decision_mismatch_count": 0, "class_order": [0, 1]}

    _patch_pipeline(
        monkeypatch, cohort, references, standard_refs, reference_sha, wheel,
        fake_samples, fake_candidates, fake_selection, fake_export,
    )
    with pytest.raises(ValueError, match="wire-finalization-failed"):
        train_beta(dataset, output, wire_root=tmp_path, runtime_wheel=wheel, config=config)
    assert fit_calls == 28
    evidence_files = list((output / "evidence/models").rglob("*.json"))
    assert len(evidence_files) == 14
    assert (output / "evidence/policy.json").is_file()


def _patch_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    cohort: SimpleNamespace,
    references: dict[str, dict[object, object]],
    standard_refs: list[dict[str, str]],
    reference_sha: str,
    wheel: Path,
    samples: object,
    candidates: object,
    selection: object,
    export: object,
) -> None:
    prefix = "multisensor_ml.beta_training.training_run"
    monkeypatch.setattr(f"{prefix}.load_cohort", lambda *_: cohort)
    monkeypatch.setattr(f"{prefix}.fit_m0", lambda *_: ({}, references))
    monkeypatch.setattr(f"{prefix}.assemble_samples", samples)
    monkeypatch.setattr(f"{prefix}.fit_candidate_models", candidates)
    monkeypatch.setattr(f"{prefix}.choose_candidate", selection)
    monkeypatch.setattr(f"{prefix}.export_candidate_onnx", export)
    monkeypatch.setattr(
        f"{prefix}.select_policy_from_validation",
        lambda *_: {"version": "synthetic_beta_stage_policy_v1",
                    "candidate_grid": [1.0, 1.5, 2.0], "d_on": 1.0, "d_off": 0.5},
    )
    monkeypatch.setattr(
        f"{prefix}.beta_artifacts.write_standards", lambda *_: standard_refs,
    )
    monkeypatch.setattr(
        f"{prefix}.beta_artifacts.write_references",
        lambda *_: {"uri": "file:///references.json", "sha256": reference_sha},
    )
    monkeypatch.setattr(
        f"{prefix}.beta_artifacts.copy_runtime_wheel",
        lambda *_: {"uri": wheel.as_uri(), "sha256": sha256(wheel.read_bytes())},
    )
    monkeypatch.setattr(
        f"{prefix}.beta_artifacts.wire_refs",
        lambda *_: (_ for _ in ()).throw(ValueError("wire-finalization-failed")),
    )
