#!/usr/bin/env python3
"""Create a fail-closed readiness report for a real Neon shadow test.

The current Neon snapshot is transport/observed data only.  This report never
turns missing corrected clocks or observer labels into synthetic predictions or
accuracy claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_shadow_artifact(snapshot: dict[str, Any]) -> dict[str, Any]:
    observed = snapshot.get("snapshot", {})
    blockers: list[str] = []
    if snapshot.get("model_ready") is not True:
        blockers.append("MODEL_READY_FALSE")
    if int(observed.get("corrected_timestamp_rows", 0)) <= 0:
        blockers.append("CORRECTED_UTC_MISSING")
    if int(observed.get("stage_rows", 0)) <= 0:
        blockers.append("OBSERVER_STAGE_LABELS_MISSING")
    if int(observed.get("model_version_rows", 0)) <= 0:
        blockers.append("MODEL_VERSION_NOT_REGISTERED")
    if snapshot.get("real_data_status") != "VERIFIED":
        blockers.append("REAL_DATA_NOT_VERIFIED")
    status = "READY_FOR_SHADOW" if not blockers else "BLOCKED_NOT_MODEL_READY"
    return {
        "schema_version": "goal1.5/tree-model-real-shadow/v1",
        "status": status,
        "prediction_executed": False,
        "accuracy_status": "NOT VERIFIED",
        "locked_test_read": False,
        "data_status": "observed/neon",
        "real_data_status": "NOT VERIFIED",
        "adapter_status": snapshot.get("adapter_status", "UNKNOWN"),
        "source": snapshot.get("source", {}),
        "observed_snapshot": observed,
        "blockers": blockers,
        "next_safe_action": (
            "corrected_utc와 독립 관찰 onset/peak/recovery/end 라벨을 등록한 뒤, "
            "model-ready schema hash를 확인하고 shadow prediction을 재실행한다."
        ),
    }


def _write_html(artifact: dict[str, Any], output: Path) -> None:
    observed = artifact["observed_snapshot"]
    blockers = "".join(f"<li>{item}</li>" for item in artifact["blockers"])
    html = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>실제 데이터 shadow 준비 점검</title>
<style>body{{font-family:NanumGothic,'Apple SD Gothic Neo',sans-serif;margin:2rem;color:#172033}}
.blocked{{background:#fff1f2;border-left:4px solid #e15759;padding:1rem}}
table{{border-collapse:collapse;width:100%;max-width:760px}}
th,td{{border:1px solid #d9dee8;padding:.5rem;text-align:left}}
</style></head><body>
<h1>최종 모델 실제 데이터 Shadow 준비 점검</h1>
<div class="blocked">상태: <b>{artifact['status']}</b> ·
정확도: <b>{artifact['accuracy_status']}</b> ·
prediction 실행: <b>{artifact['prediction_executed']}</b></div>
<p>이 문서는 Neon 관측 데이터가 실제 모델 입력·라벨 계약을 만족하는지 확인합니다.
결측 계약을 임의로 보정하거나 synthetic truth를 실제 라벨로 사용하지 않았습니다.</p>
<table><tr><th>항목</th><th>관찰값</th></tr>
<tr><td>수집 행</td><td>{observed.get('row_count', 0)}</td></tr>
<tr><td>세션</td><td>{observed.get('session_count', 0)}</td></tr>
<tr><td>corrected UTC 행</td><td>{observed.get('corrected_timestamp_rows', 0)}</td></tr>
<tr><td>단계 라벨 행</td><td>{observed.get('stage_rows', 0)}</td></tr>
<tr><td>모델 버전 행</td><td>{observed.get('model_version_rows', 0)}</td></tr>
</table>
<h2>차단 사유</h2><ul>{blockers or '<li>없음</li>'}</ul>
<p>다음 안전한 조치: {artifact['next_safe_action']}</p>
</body></html>"""
    output.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = build_shadow_artifact(_read_json(args.snapshot))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output.with_suffix(".artifact.json")
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_html(artifact, args.output)
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "report": str(args.output),
                "artifact": str(artifact_path),
            },
            ensure_ascii=False,
        )
    )
    return 0 if artifact["status"] == "READY_FOR_SHADOW" else 2


if __name__ == "__main__":
    raise SystemExit(main())
