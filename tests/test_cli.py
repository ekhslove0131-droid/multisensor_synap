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
