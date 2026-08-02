from __future__ import annotations

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
        "materialize-synthetic",
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
