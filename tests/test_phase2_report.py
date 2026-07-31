import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def test_phase2_korean_report_records_verified_scope() -> None:
    html = (REPORTS / "goal15_phase2_timeseries_validation_ko.html").read_text()

    for expected in (
        "2차 시계열 머신러닝 검증 보고서",
        "segment-aware",
        "후보 결과 보존",
        "causal 시계열 피처",
        "시행착오",
        "ML 재학습",
        "미실행",
        "oracle/sanity",
        "NOT VERIFIED",
        "locked test",
    ):
        assert expected in html


def test_phase2_artifact_does_not_claim_unrun_model_results() -> None:
    artifact = json.loads(
        (REPORTS / "goal15_phase2_timeseries_validation_ko.artifact.json").read_text()
    )
    rows = artifact["snapshot"]["datasets"]["phase2_status"]
    status = {row["stage"]: row for row in rows}

    assert status["이벤트 평가 계약"]["completion"] == 100
    assert status["후보 결과 보존"]["completion"] == 100
    assert status["causal 피처"]["completion"] == 100
    assert status["ML 재학습"]["completion"] == 0
    t4_dual_gpu = "T4\u00d72 DL 비교"
    assert status[t4_dual_gpu]["completion"] == 0
    assert status["LightGBM 비교"]["completion"] == 0
