# Kidsignal Cloud/BigQuery 인계 v1

## Cloud 팀 구현 목록

1. GCS committed exact-window manifest와 Cloud 내부 projection에는 UUID lineage,
   corrected UTC bounds, source set, quality/readiness와 artifact URI/SHA를 기록한다.
2. BigQuery에 committed window, feature set, label revision, independent review,
   cohort manifest, split assignment projection을 만든다.
3. 모델에는 account/membership/profile/auth, direct person/source/capture UUID와 legacy
   person_key, raw, GCS credential lineage를 노출하지 않는다. 대신 cohort-scoped
   `training_subject_uuid`와 `training_capture_set_uuid`를 제공한다.
4. authorized/materialized view는 `training_cohort_uuid`와 `cohort_digest`가 모두
   일치할 때만 rows를 반환한다.
5. Phone은 원천값을 보내고 Cloud 결과만 표시한다. 현재 Cloud exact composition은
   source manifest scheduling 교집합과 corrected-time provenance만 사용한다.
   cross-device waveform alignment, offset/lag 추정, 보간, 리샘플링은 미구현이며 별도
   연구·승인 계약 전에는 수행하지 않는다. feature, label, inference는 backend 책임이다.
6. Watch-only와 H10-only router를 source readiness로 선택한다. Watch+H10 fused feature는
   현재 `PROPOSED/NOT VERIFIED`이므로 Cloud feature 계약 승인 전 배포 입력으로 요구하지
   않는다. 누락 source를 0/null로 보충하지 않는다.
7. 모델 selection view는 train+validation만 반환한다. locked holdout row/label은 notebook에
   노출하지 않고 sealed evaluator가 최종 immutable bundle만 받아 aggregate 비교한다.
8. locked holdout의 row-level FP/FN은 모델 학습 쪽으로 반환하지 않는다. holdout retirement
   뒤에만 신규 cohort 후보로 만들며 운영/adjudicated error slice와 분리한다.
9. `training_capture_set_uuid`는 동일 cohort 내부의 capture continuity/sequence group에만
   사용한다. 같은 key가 여러 `training_subject_uuid`에 걸치면 handoff를 거부한다.
10. 모델 view의 feature JSON은 feature schema UUID/hash와 정확히 일치하는 flat finite
    numeric object여야 한다. nested/list/null/bool은 모델 동기화 단계에서 거부한다.

## 모델 팀 반환 목록

- `platform_contract.json`
- source schema 2개와 feature schema 3개
- `label_review_schema.json`
- `bigquery_training_contract.json`
- source 조합별 CPU ONNX bundle/manifest/SHA
- evaluation/comparison manifest
- candidate/adapter/threshold 상태와 배포 가능/불가 사유

## 현재 반환 상태

- contract/local ONNX fixture: 생성 가능
- private BigQuery readback: `NOT VERIFIED`
- actual standard/candidate performance: `NOT VERIFIED`
- reviewed FP/FN: `NOT EVALUABLE`
- deployment: `BLOCKED_NO_FROZEN_REAL_COHORT`
