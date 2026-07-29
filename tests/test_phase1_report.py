from __future__ import annotations

import json
from pathlib import Path

REPORT_ROOT = Path("reports")
ARTIFACT_PATH = REPORT_ROOT / "goal15_phase1_interim_ko.artifact.json"
HTML_PATH = REPORT_ROOT / "goal15_phase1_interim_ko.html"


def test_phase1_report_contains_history_failures_and_phase2_plan() -> None:
    html = HTML_PATH.read_text()

    required = {
        "1차 합성데이터",
        "시행착오",
        "Kaggle ML data v2",
        "Kaggle ML benchmark v4",
        "row F1 0.05",
        "2차 실행계획",
        "oracle/sanity",
        "NOT VERIFIED",
    }
    assert all(text in html for text in required)

    artifact = json.loads(ARTIFACT_PATH.read_text())
    report_text = "\n".join(
        block.get("body", "") for block in artifact["manifest"]["blocks"]
    )
    assert "event recall 1.0" in report_text


def test_phase1_report_artifact_preserves_reviewed_metric_values() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text())

    assert artifact["surface"] == "report"
    assert artifact["snapshot"]["status"] == "ready"
    assert artifact["snapshot"]["datasets"]["validation_summary"] == [
        {
            "target": "패턴",
            "aucpr": 0.494423,
            "auroc": 0.81017,
            "f1": 0.049752,
            "support": 2592000,
            "status": "oracle/sanity",
        },
        {
            "target": "5단계 평균",
            "aucpr": 0.63908,
            "auroc": 0.866254,
            "f1": 0.603361,
            "support": 66124,
            "status": "oracle/sanity",
        },
        {
            "target": "행동 평균",
            "aucpr": 0.398993,
            "auroc": 0.540588,
            "f1": 0.379214,
            "support": 74322,
            "status": "oracle/sanity",
        },
    ]


def test_phase1_report_artifact_keeps_title_and_sections_in_reading_order() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text())
    manifest = artifact["manifest"]
    blocks = manifest["blocks"]

    assert blocks[0]["type"] == "markdown"
    assert blocks[0]["body"].startswith(f"# {manifest['title']}")
    assert [block["id"] for block in blocks] == [
        "title",
        "technical_summary",
        "initial_goal",
        "synthetic_data",
        "work_history",
        "trials_and_errors",
        "current_results",
        "validation_chart",
        "model_direction",
        "phase2_plan",
        "limitations",
        "further_questions",
    ]
