from __future__ import annotations

import hashlib
import inspect
import json
import zipfile
from dataclasses import replace
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import pytest

from multisensor_ml.model_sync_receipt import parse_model_sync_receipt
from multisensor_ml.observational_contract import FEATURE_NAMES
from multisensor_ml.platform_contract import label_review_schema
from multisensor_ml.standard_baseline_bigquery import (
    EXPECTED_MODEL_READER,
    RUNTIME_TO_TRAINER_INDICES,
    STANDARD_BASELINE_VIEW_FIELDS,
    STANDARD_TRAINER_FEATURE_NAMES,
    TARGET_SOURCE_RUNTIME_INDEX,
)
from multisensor_ml.standard_candidate_bundle import (
    CandidateCoefficients,
    StandardCandidateExportResult,
    _validate_prepared_standard_dataset,
    run_receipt_bound_standard_candidate_export,
    verify_standard_candidate_bundle,
)

STANDARD_COHORT_UUID = "00000000-0000-4000-8000-000000000099"
WATCH_SCHEMA_UUID = "9b842d8c-8889-5259-acca-77baa0c7729d"
WATCH_SCHEMA_HASH = "2857f8a16cd4450a18f8701c1c9c9f397dee4b291fefd2475279883a0f8a29de"
EXPECTED_ARCHIVE_MEMBERS = {
    "SHA256SUMS.json",
    "feature_schema.json",
    "golden_fixture.json",
    "golden_output.json",
    "label_review_schema.json",
    "manifest.json",
    "model.onnx",
    "runtime.json",
    "runtime_requirements.txt",
    "sealed_evaluation_request.json",
    "split_manifest.json",
    "training_evidence.json",
    "training_projection.json",
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _attach_public_digests(rows: list[dict[str, object]]) -> None:
    members = sorted(
        [
            [
                row["training_subject_uuid"],
                row["training_capture_set_uuid"],
                row["standard_hour_uuid"],
                row["split_role"],
                row["hour_start_ms"],
                row["source_row_digest"],
            ]
            for row in rows
        ]
    )
    first = rows[0]
    cohort_material = [
        first["stable_standard_version"],
        first["feature_schema_uuid"],
        first["feature_schema_hash"],
        first["split_policy"],
        first["purge_seconds"],
        first["eligibility_policy"],
        members,
    ]
    split_material = [[member[0], member[2], member[3]] for member in members]
    for row in rows:
        row["public_cohort_digest"] = _sha256_json(cohort_material)
        row["public_split_digest"] = _sha256_json(split_material)


def _standard_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    subjects = {
        "TRAIN": str(uuid5(NAMESPACE_URL, "bundle-train-subject")),
        "VALIDATION": str(uuid5(NAMESPACE_URL, "bundle-validation-subject")),
    }
    captures = {
        split: str(uuid5(NAMESPACE_URL, f"bundle-{split.lower()}-capture"))
        for split in subjects
    }
    index = 0
    for split, targets in (("TRAIN", (0.2, 0.8)), ("VALIDATION", (0.3, 0.7))):
        for target in targets:
            index += 1
            features = {
                name: float(position + 1) / 10.0
                for position, name in enumerate(FEATURE_NAMES)
            }
            features["watch_eda_z"] = target
            features["watch_load_median_300"] = target
            rows.append(
                {
                    "standard_cohort_uuid": STANDARD_COHORT_UUID,
                    "cohort_digest": "c" * 64,
                    "split_digest": "d" * 64,
                    "public_cohort_digest": "",
                    "public_split_digest": "",
                    "stable_standard_version": "stable-stress-standard-v1",
                    "feature_schema_uuid": WATCH_SCHEMA_UUID,
                    "feature_schema_hash": WATCH_SCHEMA_HASH,
                    "split_policy": "PERSON_GROUP",
                    "purge_seconds": 1800,
                    "eligibility_policy": "ELIGIBLE_NO_PATTERN_REAL",
                    "training_subject_uuid": subjects[split],
                    "training_capture_set_uuid": captures[split],
                    "standard_hour_uuid": str(
                        uuid5(NAMESPACE_URL, f"bundle-standard-hour-{index}")
                    ),
                    "source_set": ["watch"],
                    "hour_start_ms": index * 3_600_000,
                    "hour_end_ms": (index + 1) * 3_600_000,
                    "feature_values": features,
                    "target_name": "no_pattern_median",
                    "target_value": target,
                    "target_unit": "positive_robust_z",
                    "valid_coverage_seconds": 3000,
                    "split_role": split,
                    "source_row_digest": f"{index:064x}",
                }
            )
    rows.sort(
        key=lambda row: (
            str(row["training_subject_uuid"]),
            int(row["hour_start_ms"]),
            str(row["standard_hour_uuid"]),
        )
    )
    _attach_public_digests(rows)
    return rows


def _model_sync_receipt(
    rows: list[dict[str, object]], *, plane: str = "standard"
) -> dict[str, object]:
    body: dict[str, object] = {
        "contract_version": 1,
        "status": "READY_FOR_MODEL_SYNC_NOT_TRAINED",
        "plane": plane,
        "cohort_uuid": STANDARD_COHORT_UUID,
        "public_cohort_digest": rows[0]["public_cohort_digest"],
        "public_split_digest": rows[0]["public_split_digest"],
        "feature_schema_uuid": WATCH_SCHEMA_UUID,
        "feature_schema_hash": WATCH_SCHEMA_HASH,
        "split_policy": "PERSON_GROUP",
        "purge_seconds": 1800,
        "authorized_field_count": 24 if plane == "standard" else 26,
        "train_row_count": 2,
        "validation_row_count": 2,
        "fit_call_count": 0,
    }
    return {**body, "handoff_digest": _sha256_json(body)}


def _export_request(
    rows: list[dict[str, object]], *, evidence_status: str = "CONTRACT_EVIDENCE_ONLY"
) -> dict[str, object]:
    receipt = parse_model_sync_receipt(_model_sync_receipt(rows))
    body: dict[str, object] = {
        "schema_version": "kidsignal-standard-candidate-export-request/v1",
        "evidence_status": evidence_status,
        "model_sync_handoff_digest": receipt.handoff_digest,
        "standard_cohort_uuid": receipt.cohort_uuid,
        "public_cohort_digest": receipt.public_cohort_digest,
        "public_split_digest": receipt.public_split_digest,
        "feature_schema_uuid": receipt.feature_schema_uuid,
        "feature_schema_hash": receipt.feature_schema_hash,
        "target_version": "stable-stress-standard-v1",
        "model_release_uuid": "22222222-2222-4222-8222-222222222222",
        "model_release": "stable-standard-candidate-v1",
        "fixture_uuid": "11111111-1111-4111-8111-111111111111",
        "evaluation_uuid": "33333333-3333-4333-8333-333333333333",
        "sealed_evaluation_input_uuid": "44444444-4444-4444-8444-444444444444",
        "sealed_evaluation_input_digest": "e" * 64,
        "reference_model_release_uuid": "66666666-6666-4666-8666-666666666666",
        "reference_artifact_sha256": "f" * 64,
        "created_at": "2026-08-27T00:00:00Z",
        "generated_at": "2026-08-27T00:00:01Z",
        "requested_metrics": ["MAE", "RMSE"],
    }
    return {**body, "request_sha256": _sha256_json(body)}


class _SchemaField:
    def __init__(self, name: str) -> None:
        self.name = name


class _Row:
    def __init__(self, values: dict[str, object]) -> None:
        self._values = values

    def items(self):
        return self._values.items()


class _Iterator:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.schema = tuple(_SchemaField(name) for name in STANDARD_BASELINE_VIEW_FIELDS)
        self._rows = tuple(
            _Row({name: row[name] for name in STANDARD_BASELINE_VIEW_FIELDS})
            for row in rows
        )

    def __iter__(self):
        return iter(self._rows)


class _Job:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._iterator = _Iterator(rows)

    def result(self) -> _Iterator:
        return self._iterator


class _Client:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.query_calls = 0

    def query(self, query: str, **kwargs: object) -> _Job:
        del query, kwargs
        self.query_calls += 1
        return _Job(self.rows)


class _FakeCoefficientProvider:
    performs_real_fit = False

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[int, int], tuple[int, ...]]] = []

    def provide(
        self, candidate_id: str, train_features: np.ndarray, train_targets: np.ndarray
    ) -> CandidateCoefficients:
        self.calls.append((candidate_id, train_features.shape, train_targets.shape))
        if candidate_id == "mean_constant_v1":
            return CandidateCoefficients(candidate_id, (0.0,) * 15, 0.5)
        weights = [0.0] * 15
        weights[0] = 1.0
        return CandidateCoefficients(candidate_id, tuple(weights), 0.0)


def _job_config(name: str, parameter_type: str, value: str) -> object:
    return {
        "query_parameters": [
            {"name": name, "type": parameter_type, "value": value}
        ]
    }


def test_receipt_bound_standard_export_uses_train_only_and_builds_exact_bundle(
    tmp_path: Path,
) -> None:
    rows = _standard_rows()
    client = _Client(rows)
    provider = _FakeCoefficientProvider()
    output = tmp_path / "candidate.zip"

    result = run_receipt_bound_standard_candidate_export(
        model_sync_receipt=_model_sync_receipt(rows),
        client=client,
        observed_at_utc="2026-08-27T00:00:00Z",
        observed_principal=EXPECTED_MODEL_READER,
        export_request=_export_request(rows),
        output_path=output,
        candidate_provider=provider,
        job_config_factory=_job_config,
    )

    assert isinstance(result, StandardCandidateExportResult)
    assert result.evidence_status == "CONTRACT_EVIDENCE_ONLY"
    assert result.selected_candidate == "ridge_l2_alpha_1_v1"
    assert result.fit_call_count == 0
    assert result.candidate_provider_call_count == 2
    assert client.query_calls == 1
    assert provider.calls == [
        ("mean_constant_v1", (2, 15), (2,)),
        ("ridge_l2_alpha_1_v1", (2, 15), (2,)),
    ]
    assert result.metrics["ridge_l2_alpha_1_v1"]["mae"] == pytest.approx(0.0)
    assert result.metrics["ridge_l2_alpha_1_v1"]["rmse"] == pytest.approx(0.0)

    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == EXPECTED_ARCHIVE_MEMBERS
        manifest = json.loads(archive.read("manifest.json"))
        label_document = json.loads(archive.read("label_review_schema.json"))
        projection = json.loads(archive.read("training_projection.json"))
        golden = json.loads(archive.read("golden_output.json"))
        evidence = json.loads(archive.read("training_evidence.json"))
        assert manifest["contract_status"] == "FROZEN_CONTRACT"
        assert manifest["model_status"] == "TRAINED_NOT_EVALUATED"
        assert manifest["target_version"] == "stable-stress-standard-v1"
        assert manifest["split_policy"] == "person_group"
        assert manifest["real_data_status"] == "NOT VERIFIED"
        assert manifest["stage"] is None
        assert manifest["delivery_eligible"] is False
        assert manifest["promotion_eligible"] is False
        assert manifest["sealed_evaluation_status"] == "NOT_REQUESTED"
        assert manifest["label_review_schema_hash"] == _sha256_json(label_document)
        assert manifest["label_review_schema_hash"] != label_document["sha256"]
        assert projection["runtime_to_trainer_indices"] == list(
            RUNTIME_TO_TRAINER_INDICES
        )
        assert projection["target_source_runtime_index"] == TARGET_SOURCE_RUNTIME_INDEX
        assert evidence == {
            "training_status": "TRAINED_NOT_EVALUATED",
            "real_cohort_rows": 4,
        }
        outputs = [case["expected_output"] for case in golden["cases"]]
        assert outputs[0] == outputs[1]
        serialized = b"\n".join(archive.read(name) for name in archive.namelist())
        for row in rows:
            assert str(row["training_subject_uuid"]).encode() not in serialized
            assert str(row["training_capture_set_uuid"]).encode() not in serialized
            assert str(row["standard_hour_uuid"]).encode() not in serialized

    verified = verify_standard_candidate_bundle(output)
    assert verified["status"] == "VERIFIED_IMMUTABLE_STANDARD_CANDIDATE"
    assert verified["paired_prediction_byte_exact"] is True

    second_client = _Client(rows)
    second_provider = _FakeCoefficientProvider()
    with pytest.raises(FileExistsError):
        run_receipt_bound_standard_candidate_export(
            model_sync_receipt=_model_sync_receipt(rows),
            client=second_client,
            observed_at_utc="2026-08-27T00:00:00Z",
            observed_principal=EXPECTED_MODEL_READER,
            export_request=_export_request(rows),
            output_path=output,
            candidate_provider=second_provider,
            job_config_factory=_job_config,
        )
    assert second_client.query_calls == 0
    assert second_provider.calls == []


def test_public_coordinator_rejects_behavior_receipt_before_query(tmp_path: Path) -> None:
    rows = _standard_rows()
    client = _Client(rows)

    with pytest.raises(ValueError, match="standard receipt"):
        run_receipt_bound_standard_candidate_export(
            model_sync_receipt=_model_sync_receipt(rows, plane="behavior"),
            client=client,
            observed_at_utc="2026-08-27T00:00:00Z",
            observed_principal=EXPECTED_MODEL_READER,
            export_request=_export_request(rows),
            output_path=tmp_path / "forbidden.zip",
            candidate_provider=_FakeCoefficientProvider(),
            job_config_factory=_job_config,
        )

    assert client.query_calls == 0


def test_immutable_zip_bytes_and_entry_metadata_are_reproducible(
    tmp_path: Path,
) -> None:
    rows = _standard_rows()
    outputs = (tmp_path / "first.zip", tmp_path / "second.zip")

    for output in outputs:
        run_receipt_bound_standard_candidate_export(
            model_sync_receipt=_model_sync_receipt(rows),
            client=_Client(rows),
            observed_at_utc="2026-08-27T00:00:00Z",
            observed_principal=EXPECTED_MODEL_READER,
            export_request=_export_request(rows),
            output_path=output,
            candidate_provider=_FakeCoefficientProvider(),
            job_config_factory=_job_config,
        )

    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    assert hashlib.sha256(outputs[0].read_bytes()).digest() == hashlib.sha256(
        outputs[1].read_bytes()
    ).digest()
    with zipfile.ZipFile(outputs[0]) as archive:
        for entry in archive.infolist():
            assert entry.date_time == (1980, 1, 1, 0, 0, 0)
            assert entry.create_system == 3
            assert entry.external_attr == 0o100644 << 16
            assert entry.compress_type == zipfile.ZIP_DEFLATED
            assert entry.extra == b""
            assert entry.comment == b""


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        (
            "target_version",
            "observational_future_mean_1800_valid90_gap5_v2",
            "target_version",
        ),
        ("public_split_digest", "0" * 64, "public_split_digest"),
    ],
)
def test_export_request_lineage_mismatch_fails_before_provider(
    tmp_path: Path, field: str, replacement: object, message: str
) -> None:
    rows = _standard_rows()
    request = _export_request(rows)
    body = {key: value for key, value in request.items() if key != "request_sha256"}
    body[field] = replacement
    request = {**body, "request_sha256": _sha256_json(body)}
    provider = _FakeCoefficientProvider()

    with pytest.raises(ValueError, match=message):
        run_receipt_bound_standard_candidate_export(
            model_sync_receipt=_model_sync_receipt(rows),
            client=_Client(rows),
            observed_at_utc="2026-08-27T00:00:00Z",
            observed_principal=EXPECTED_MODEL_READER,
            export_request=request,
            output_path=tmp_path / "blocked.zip",
            candidate_provider=provider,
            job_config_factory=_job_config,
        )

    assert provider.calls == []
    assert not (tmp_path / "blocked.zip").exists()


def test_internal_prepared_contract_detects_target_projection_tamper(
    tmp_path: Path,
) -> None:
    rows = _standard_rows()
    from multisensor_ml.standard_baseline_bigquery import (
        run_read_only_standard_cohort_sdk_reader,
    )

    read_result = run_read_only_standard_cohort_sdk_reader(
        client=_Client(rows),
        standard_cohort_uuid=STANDARD_COHORT_UUID,
        expected_public_cohort_digest=str(rows[0]["public_cohort_digest"]),
        expected_public_split_digest=str(rows[0]["public_split_digest"]),
        observed_at_utc="2026-08-27T00:00:00Z",
        observed_principal=EXPECTED_MODEL_READER,
        job_config_factory=_job_config,
    )
    assert read_result.prepared_dataset is not None
    tampered = replace(
        read_result.prepared_dataset,
        trainer_feature_names=(*STANDARD_TRAINER_FEATURE_NAMES[:-1], "watch_load_median_300"),
    )
    with pytest.raises(ValueError, match="trainer feature order"):
        _validate_prepared_standard_dataset(tampered)
    assert not (tmp_path / "tampered.zip").exists()


def test_zero_rows_never_reach_candidate_provider(tmp_path: Path) -> None:
    rows = _standard_rows()
    provider = _FakeCoefficientProvider()

    with pytest.raises(RuntimeError, match="BLOCKED_NO_REAL_COHORT"):
        run_receipt_bound_standard_candidate_export(
            model_sync_receipt=_model_sync_receipt(rows),
            client=_Client([]),
            observed_at_utc="2026-08-27T00:00:00Z",
            observed_principal=EXPECTED_MODEL_READER,
            export_request=_export_request(rows),
            output_path=tmp_path / "zero.zip",
            candidate_provider=provider,
            job_config_factory=_job_config,
        )

    assert provider.calls == []
    assert not (tmp_path / "zero.zip").exists()


def test_label_document_full_hash_regression() -> None:
    document = label_review_schema()
    assert document["sha256"] == (
        "f4db47381130215cfbb35d2f2a6187ef22bdedd33e4da209c4c0a93ad010981f"
    )
    assert _sha256_json(document) == (
        "be344f4a7b9e4c5cbd08494783ac18063bdc01ef26c89a532580710657203b57"
    )


def test_public_coordinator_has_no_manual_dataset_or_lineage_parameters() -> None:
    parameters = inspect.signature(
        run_receipt_bound_standard_candidate_export
    ).parameters
    assert set(parameters) == {
        "model_sync_receipt",
        "client",
        "observed_at_utc",
        "observed_principal",
        "export_request",
        "output_path",
        "candidate_provider",
        "job_config_factory",
    }
    assert not {
        "prepared_dataset",
        "train_features",
        "validation_features",
        "standard_cohort_uuid",
        "public_cohort_digest",
        "public_split_digest",
    }.intersection(parameters)


def test_intake_notebook_remains_read_only_and_does_not_import_training_entrypoint() -> None:
    root = Path(__file__).parents[1]
    for path in (
        root / "scripts/build_kaggle_bigquery_intake_notebook.py",
        root / "kaggle/10_kidsignal_bigquery_intake.ipynb",
    ):
        source = path.read_text(encoding="utf-8")
        assert "standard_candidate_bundle" not in source
        assert "run_receipt_bound_standard_candidate_export" not in source


@pytest.mark.parametrize(
    ("evidence_status", "provide_external", "performs_real_fit", "message"),
    [
        (
            "CONTRACT_EVIDENCE_ONLY",
            True,
            True,
            "requires an explicit non-fit provider",
        ),
        (
            "REAL_FROZEN_COHORT",
            True,
            False,
            "forbids an external candidate provider",
        ),
        (
            "CONTRACT_EVIDENCE_ONLY",
            False,
            False,
            "requires an explicit non-fit provider",
        ),
    ],
)
def test_evidence_status_and_provider_fit_authority_are_bidirectionally_sealed(
    tmp_path: Path,
    evidence_status: str,
    provide_external: bool,
    performs_real_fit: bool,
    message: str,
) -> None:
    rows = _standard_rows()
    provider = _FakeCoefficientProvider()
    provider.performs_real_fit = performs_real_fit
    client = _Client(rows)

    with pytest.raises(ValueError, match=message):
        run_receipt_bound_standard_candidate_export(
            model_sync_receipt=_model_sync_receipt(rows),
            client=client,
            observed_at_utc="2026-08-27T00:00:00Z",
            observed_principal=EXPECTED_MODEL_READER,
            export_request=_export_request(rows, evidence_status=evidence_status),
            output_path=tmp_path / "forbidden-fit-claim.zip",
            candidate_provider=provider if provide_external else None,
            job_config_factory=_job_config,
        )

    assert client.query_calls == 0
    assert provider.calls == []
    assert not (tmp_path / "forbidden-fit-claim.zip").exists()
