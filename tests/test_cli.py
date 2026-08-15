from __future__ import annotations

from pathlib import Path

import pytest

from multisensor_ml import cli
from multisensor_ml.cli import build_parser


def test_public_cli_exposes_all_goal15_workflow_commands() -> None:
    parser = build_parser()
    subparsers = next(
        action for action in parser._actions if action.dest == "command"
    )

    assert set(subparsers.choices) == {
        "evaluate",
        "export-knime",
        "factory",
        "kaggle-model",
        "availability-model",
        "neon-adapter",
        "observational-standard",
        "materialize-synthetic",
        "monitor-v2",
        "model-platform",
        "onnx",
        "phase3",
        "prepare",
        "registry",
        "run-all",
        "train",
    }
    parsed = parser.parse_args(
        [
            "evaluate",
            "--bundle",
            "bundle",
            "--dataset",
            "series",
            "--role",
            "locked_test",
            "--audit-reason",
            "final acceptance",
        ]
    )
    assert parsed.audit_reason == "final acceptance"

    factory = parser.parse_args(
        [
            "factory",
            "run",
            "--config",
            "configs/factory.yaml",
            "--output-receipt",
            "receipt.json",
        ]
    )
    assert factory.factory_command == "run"
    assert str(factory.output_receipt) == "receipt.json"

    registry = parser.parse_args(
        [
            "registry",
            "run-stage",
            "--stage",
            "stage-model",
            "--run-id",
            "run-1",
            "--input-receipt",
            "input.json",
            "--output-receipt",
            "output.json",
        ]
    )
    assert registry.registry_command == "run-stage"
    assert registry.stage == "stage-model"

    monitoring = parser.parse_args(
        ["monitor-v2", "run", "--config", "configs/monitoring_v2_quick.yaml"]
    )
    assert monitoring.monitoring_v2_command == "run"

    observational = parser.parse_args(
        [
            "observational-standard",
            "train",
            "--input",
            "observed.parquet",
            "--output",
            "artifact",
            "--source-domain",
            "real_observed",
        ]
    )
    assert observational.observational_command == "train"
    assert observational.source_domain == "real_observed"

    baseline_export = parser.parse_args(
        [
            "observational-standard",
            "export-baseline-shadow",
            "--output",
            "artifact",
        ]
    )
    assert baseline_export.observational_command == "export-baseline-shadow"

    baseline_verify = parser.parse_args(
        [
            "observational-standard",
            "verify-baseline-shadow",
            "--bundle",
            "artifact",
        ]
    )
    assert baseline_verify.observational_command == "verify-baseline-shadow"

    candidate_export = parser.parse_args(
        [
            "observational-standard",
            "export-candidate-shadow",
            "--input",
            "observed.parquet",
            "--output",
            "candidate",
            "--active-bundle",
            "active",
            "--candidate",
            "ridge",
        ]
    )
    assert candidate_export.observational_command == "export-candidate-shadow"
    assert candidate_export.candidate == "ridge"

    candidate_verify = parser.parse_args(
        [
            "observational-standard",
            "verify-candidate-shadow",
            "--bundle",
            "candidate",
        ]
    )
    assert candidate_verify.observational_command == "verify-candidate-shadow"

    reference_export = parser.parse_args(
        [
            "observational-standard",
            "export-training-reference",
            "--input",
            "observed.parquet",
            "--candidate-bundle",
            "candidate",
            "--output",
            "reference",
            "--generated-at",
            "2026-08-06T09:00:00Z",
        ]
    )
    assert reference_export.observational_command == "export-training-reference"

    reference_verify = parser.parse_args(
        [
            "observational-standard",
            "verify-training-reference",
            "--bundle",
            "reference",
        ]
    )
    assert reference_verify.observational_command == "verify-training-reference"

    platform_preflight = parser.parse_args(
        [
            "model-platform",
            "preflight-bigquery",
            "--output-receipt",
            "preflight.json",
        ]
    )
    assert platform_preflight.model_platform_command == "preflight-bigquery"
    assert str(platform_preflight.output_receipt) == "preflight.json"

    cohort_reader = parser.parse_args(
        [
            "model-platform",
            "read-bigquery-cohort",
            "--training-cohort-uuid",
            "10000000-0000-4000-8000-000000000001",
            "--expected-public-cohort-digest",
            "a" * 64,
            "--expected-public-split-digest",
            "b" * 64,
            "--output-receipt",
            "cohort-readiness.json",
        ]
    )
    assert cohort_reader.model_platform_command == "read-bigquery-cohort"
    assert cohort_reader.training_cohort_uuid == (
        "10000000-0000-4000-8000-000000000001"
    )
    assert cohort_reader.expected_public_cohort_digest == "a" * 64
    assert cohort_reader.expected_public_split_digest == "b" * 64
    assert str(cohort_reader.output_receipt) == "cohort-readiness.json"

    standard_cohort_reader = parser.parse_args(
        [
            "model-platform",
            "read-standard-bigquery-cohort",
            "--standard-cohort-uuid",
            "00000000-0000-4000-8000-000000000099",
            "--expected-public-cohort-digest",
            "c" * 64,
            "--expected-public-split-digest",
            "d" * 64,
            "--output-receipt",
            "standard-cohort-readiness.json",
        ]
    )
    assert (
        standard_cohort_reader.model_platform_command
        == "read-standard-bigquery-cohort"
    )
    assert standard_cohort_reader.standard_cohort_uuid == (
        "00000000-0000-4000-8000-000000000099"
    )
    assert standard_cohort_reader.expected_public_cohort_digest == "c" * 64
    assert standard_cohort_reader.expected_public_split_digest == "d" * 64
    assert str(standard_cohort_reader.output_receipt) == (
        "standard-cohort-readiness.json"
    )


def test_standard_cohort_cli_delegates_to_read_only_reader_without_fit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(cli, "active_gcloud_principal", lambda: "model-reader")

    def fake_reader(output_receipt: Path, **kwargs: object) -> dict[str, object]:
        calls.append({"output_receipt": output_receipt, **kwargs})
        return {
            "status": "BLOCKED_NO_REAL_COHORT",
            "standard_cohort_uuid": kwargs["standard_cohort_uuid"],
            "row_count": 0,
            "training_ready": False,
            "fit_call_count": 0,
            "training_status": "NOT STARTED",
            "evaluation_status": "NOT EVALUABLE",
        }

    monkeypatch.setattr(cli, "run_read_only_standard_cohort_reader", fake_reader)
    output = tmp_path / "standard-receipt.json"
    result = cli.main(
        [
            "model-platform",
            "read-standard-bigquery-cohort",
            "--standard-cohort-uuid",
            "00000000-0000-4000-8000-000000000099",
            "--expected-public-cohort-digest",
            "c" * 64,
            "--expected-public-split-digest",
            "d" * 64,
            "--output-receipt",
            str(output),
            "--observed-at-utc",
            "2026-08-15T05:00:00Z",
        ]
    )

    assert result == 2
    assert calls == [
        {
            "output_receipt": output,
            "standard_cohort_uuid": "00000000-0000-4000-8000-000000000099",
            "expected_public_cohort_digest": "c" * 64,
            "expected_public_split_digest": "d" * 64,
            "observed_at_utc": "2026-08-15T05:00:00Z",
            "observed_principal": "model-reader",
        }
    ]


def test_kaggle_model_commands_parse() -> None:
    parser = build_parser()
    package = parser.parse_args(
        [
            "kaggle-model",
            "package",
            "--config",
            "config.yaml",
            "--wheel",
            "model.whl",
            "--source-project-root",
            "/source/project",
        ]
    )
    verify = parser.parse_args(
        ["kaggle-model", "verify", "--package", "dist/model"]
    )
    reproduce = parser.parse_args(
        [
            "kaggle-model",
            "reproduce",
            "--model-root",
            "payload",
            "--input",
            "sample.parquet",
            "--output",
            "prediction.parquet",
            "--expected",
            "expected.parquet",
        ]
    )

    assert package.kaggle_model_command == "package"
    assert str(package.source_project_root) == "/source/project"
    assert verify.kaggle_model_command == "verify"
    assert reproduce.kaggle_model_command == "reproduce"


def test_availability_model_commands_parse() -> None:
    parser = build_parser()
    package = parser.parse_args(
        [
            "availability-model",
            "package",
            "--project-root",
            "/project",
            "--series",
            "mvp3-oracle-v1",
            "--output",
            "/tmp/package",
            "--wheel",
            "/tmp/multisensor_ml.whl",
            "--max-rows-per-person",
            "1000",
        ]
    )
    verify = parser.parse_args(
        ["availability-model", "verify", "--package", "/tmp/package"]
    )

    assert package.availability_model_command == "package"
    assert str(package.project_root) == "/project"
    assert package.series == "mvp3-oracle-v1"
    assert str(package.wheel) == "/tmp/multisensor_ml.whl"
    assert package.max_rows_per_person == 1000
    assert verify.availability_model_command == "verify"
