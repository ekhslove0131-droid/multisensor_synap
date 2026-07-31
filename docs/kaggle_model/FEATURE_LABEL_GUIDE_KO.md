# 합성 계층형 모델 Feature·Label 안내

## 범위

이 문서는 `oracle/sanity` 합성 모델의 입력과 라벨을 설명합니다. 실제 정확도는 `NOT VERIFIED`이며 의료 판단 기준이 아닙니다.

## 기준선과 STD-A

전역 기준선은 train person의 context별 median/MAD입니다. 개인 기준선은 과거 데이터만 사용해 warm-up 1,800초, lookback 21,600초, 60초 갱신으로 계산합니다. 개인 가중치에는 검증된 상한을 적용합니다. GMM K=2~6이 안정성·최소 인원 조건을 통과하지 못해 K=1 `STD-A`로 복귀했으며, 이것은 임상적 사람 유형이 아닙니다.

## 모델 1 라벨

- `NO_EVENT`: 사건 관련 구간 밖
- `LOW`: `pre_early`
- `MEDIUM`: `pre_late`, `onset`
- `HIGH`: `peak`
- `DECREASING`: `recovery_early`
- `RECOVERY`: `recovery_late`, `post`

사건 head가 event probability를 먼저 만들고, 사건 후보에만 5단계 head와 허용 전이 decoder를 적용합니다. OOD·장시간 결측·품질 실패는 `NOT_DECISIONABLE`입니다.

## 모델 2 행동 라벨

`ear_covering`, `head_turn_away`, `withdrawal_movement`, `exit_attempt`, `repetitive_hand_movement`, `repetitive_body_movement`, `movement_reduction`, `motion_freeze`, `repetitive_object_contact`, `sustained_pressure_or_contact`의 다중 라벨 확률을 냅니다. 행동은 객관적 관찰 코드이며 meltdown 같은 요약 태그를 직접 학습하지 않습니다.

## 입력과 누출 방지

입력은 causal robust-z, rolling mean/std/slope, 시간 주기, context, `STD-A` membership, 모델 1의 확률입니다. hidden archetype, intensity truth, event truth ID, `active_target_*`, `event_id`, `stage_code`는 모델 입력에서 차단합니다. 라벨은 결과 비교용 감사 열이며 feature로 섞지 않습니다.

실제 Muse·Polar·Watch 원시 신호는 이 schema가 아닙니다. 실제 사용에는 corrected UTC, 품질값, 장비별 파생변수를 같은 causal 계약으로 만드는 별도 어댑터가 필요합니다.
