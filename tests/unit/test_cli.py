from __future__ import annotations

import json
from pathlib import Path

import pytest

import multisensor_synth.cli as cli_module
from multisensor_synth.cli import run_cli

ROOT = Path(__file__).parents[2]


def test_inspect_config_prints_resolved_config(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = run_cli(
        ["inspect-config", "--config", str(ROOT / "configs" / "quick.yaml")]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["status"] == "PASS"
    assert output["resolved"]["profile"] == "quick"


def test_estimate_reports_exact_quick_size(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = run_cli(["estimate", "--config", str(ROOT / "configs" / "quick.yaml")])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["participants"] == 3
    assert output["duration_sec"] == 21_600
    assert output["canonical_truth_rows"] == 64_800
    assert output["target_event_range"] == {"min": 6, "max": 6}


def test_config_error_returns_exit_code_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("project: wrong\n", encoding="utf-8")

    exit_code = run_cli(["inspect-config", "--config", str(invalid)])

    assert exit_code == 2
    assert "configuration error" in capsys.readouterr().err


def test_generate_requires_truth_only_flag(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = run_cli(
        ["generate", "--config", str(ROOT / "configs" / "quick.yaml")]
    )

    assert exit_code == 2
    assert "--truth-only" in capsys.readouterr().err


def test_unexpected_value_error_is_not_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_: object) -> dict[str, object]:
        raise ValueError("unexpected implementation failure")

    monkeypatch.setattr(cli_module, "estimate_config", fail)

    with pytest.raises(ValueError, match="unexpected implementation failure"):
        run_cli(["estimate", "--config", str(ROOT / "configs" / "quick.yaml")])
