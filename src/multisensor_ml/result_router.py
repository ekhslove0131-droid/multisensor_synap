from __future__ import annotations

from typing import cast

STAGE_KO: dict[str, str] = {
    "NO_EVENT": "무이벤트",
    "LOW": "저",
    "MEDIUM": "중",
    "HIGH": "고",
    "DECREASING": "감소",
    "RECOVERY": "회복",
    "NOT_DECISIONABLE": "판단 보류",
}
OOD_KO: dict[str, str] = {
    "IN_DISTRIBUTION": "정상 범위",
    "ADAPTATION_CAPPED": "개인 보정 상한 적용",
    "OOD_MONITOR": "분포 이탈 관찰",
    "RETRAIN_CANDIDATE": "재학습 검토 후보",
    "NOT_DECISIONABLE": "판단 보류",
}
BEHAVIOR_KO: dict[str, str] = {
    "ear_covering": "귀 막기",
    "head_turn_away": "고개 돌리기",
    "withdrawal_movement": "물러나는 움직임",
    "exit_attempt": "이탈 시도",
    "repetitive_hand_movement": "반복 손 움직임",
    "repetitive_body_movement": "반복 몸 움직임",
    "movement_reduction": "움직임 감소",
    "motion_freeze": "움직임 멈춤",
    "repetitive_object_contact": "반복 물체 접촉",
    "sustained_pressure_or_contact": "지속 압박 또는 접촉",
}


def route_prediction_ko(
    prediction: dict[str, object],
    *,
    router_version: str,
) -> dict[str, object]:
    """Add a versioned Korean decision layer while preserving raw probabilities."""

    stage = str(prediction["stage_code"])
    ood = str(prediction["ood_status"])
    behavior_probabilities = cast(
        dict[str, float],
        prediction.get("behavior_probabilities", {}),
    )
    missing_behaviors = sorted(set(behavior_probabilities) - set(BEHAVIOR_KO))
    if stage not in STAGE_KO:
        raise ValueError(f"missing Korean stage translation: {stage}")
    if ood not in OOD_KO:
        raise ValueError(f"missing Korean OOD translation: {ood}")
    if missing_behaviors:
        raise ValueError(
            "missing Korean behavior translations: " + ", ".join(missing_behaviors)
        )
    abstain = ood in {
        "OOD_MONITOR",
        "RETRAIN_CANDIDATE",
        "NOT_DECISIONABLE",
    } or stage == "NOT_DECISIONABLE"
    stage_ko = "판단 보류" if abstain else STAGE_KO[stage]
    decision = (
        "판단 보류"
        if abstain
        else ("무이벤트" if stage == "NO_EVENT" else "패턴 감지")
    )
    return {
        "router_version": router_version,
        "판정": decision,
        "단계": stage_ko,
        "분포상태": OOD_KO[ood],
        "행동확률": {
            BEHAVIOR_KO[code]: probability
            for code, probability in behavior_probabilities.items()
        },
        "raw_event_probability": float(
            cast(str | float | int, prediction["event_probability"])
        ),
        "raw_behavior_probabilities": behavior_probabilities,
        "raw_stage_code": stage,
        "raw_ood_status": ood,
    }
