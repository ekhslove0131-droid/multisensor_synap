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
        "materialize-synthetic",
        "prepare",
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
