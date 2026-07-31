import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def test_phase3_korean_report_records_failure_and_bottleneck() -> None:
    html = (REPORTS / "goal15_phase3_personal_pattern_ko.html").read_text()

    for expected in (
        "3차 개인 기준선·누적 부하 패턴 검증 보고서",
        "12,960,000",
        "24,505",
        "DeadKernelError",
        "병목",
        "시행착오",
        "체크포인트",
        "oracle/sanity",
        "NOT VERIFIED",
        "locked test",
    ):
        assert expected in html


def test_phase3_artifact_does_not_claim_missing_uplift() -> None:
    artifact = json.loads(
        (REPORTS / "goal15_phase3_personal_pattern_ko.artifact.json").read_text()
    )
    rows = artifact["snapshot"]["datasets"]["phase3_status"]
    status = {row["stage"]: row for row in rows}

    assert status["raw source 업로드"]["completion"] == 100
    assert status["05 source 준비"]["completion"] == 100
    assert status["06 후보 학습"] == {
        "stage": "06 후보 학습",
        "completion": 0,
        "status": "실패",
        "evidence": "24,505초 후 DeadKernelError",
    }
    assert status["validation uplift"]["status"] == "미산출"
    assert status["locked test"]["status"] == "미사용"
    assert set(artifact["snapshot"]["datasets"]) == {"phase3_status"}
