# Kidsignal live BigQuery 학습 preflight 인계

관찰 시각: `2026-08-13T06:39:03.377799Z`  
실행 범위: authorized view 스키마 조회와 `COUNT(*)`만 수행한 읽기 전용 preflight

## 결론

기존 `BLOCKED_NO_FROZEN_TRAINING_VIEW`는 해소됐다. 다음 authorized view가 실제로
존재하고 모델 공개 계약의 24개 필드와 정확히 일치했다.

`multi-app-kidsignal-260801.kidsignal_model_training.training_examples_train_validation_v1`

그러나 관찰된 row가 0개이므로 현재 모델 상태는 아래와 같다.

- preflight: `BLOCKED_NO_REAL_COHORT`
- sync: `NOT STARTED`
- training/Kaggle: `NOT STARTED`
- evaluation: `NOT EVALUABLE`
- 실제 성능: `NOT VERIFIED`
- LOCKED 접근: `false`

schema field digest는
`4b10b69038dfee825d6008b6bee2b22704ae3cfa0cacab595106f2355de96176`이다.
account, membership, provider, 직접 person/source/capture UUID, raw, artifact URI,
LOCKED field는 노출되지 않았다.

## 로컬 principal 경계

이번 readback은 로컬 활성 principal `[REDACTED_PERSONAL_PRINCIPAL]`으로 수행됐다. 배포된 최소권한
reader인
`kidsignal-model-reader@multi-app-kidsignal-260801.iam.gserviceaccount.com`으로 직접
실행한 것은 아니므로 receipt의 `model_reader_identity_verified=false`가 맞다.

따라서 이번 결과는 view 존재·공개 schema·0행 상태의 readback 증거다. 전용 model reader의
실제 로그인/impersonation parity 증거로 사용하면 안 된다. row가 생기더라도 전용 reader
identity가 검증되기 전에는 동기화를 시작하지 않는다.

## 모델 측 변경

- `model-platform preflight-bigquery` CLI 추가
- project/location/view를 상수로 고정하고 다른 dataset을 입력으로 받지 않음
- `bq show --schema`와 authorized view `COUNT(*)`만 실행
- 공개 필드 누락, unknown/private/raw/LOCKED field, 중복 필드를 fail-closed 거부
- 0행에서 create-only `BLOCKED_NO_REAL_COHORT` receipt 생성
- row가 생겨도 명시적 `training_cohort_uuid + cohort_digest` 전에는 sync/training 미시작
- H10-only는 새 hash
  `5ff8e9ceaa534bb8b7fafbdd7f844b06c0a4b21bb206428d5ce9239eec5459d6`와
  `PROPOSED_NOT_IMPLEMENTED` 유지
- Watch+H10은 `PROPOSED_NOT_APPROVED` 유지

## 첫 실제 입력 재개 조건

1. authorized view에 동일 envelope의 실제 row가 존재한다.
2. Cloud가 `training_cohort_uuid`와 일치하는 64자리 `cohort_digest`를 명시한다.
3. `TRAIN`과 `VALIDATION`이 모두 존재하고 다른 split/LOCKED row는 없다.
4. `purge_seconds >= 1800`, 사전등록 split policy, capture-set one-subject invariant를
   만족한다.
5. runtime `feature_schema_uuid/hash`가 source variant의 승인 상태와 일치한다.
6. `TARGET_EVENT`, `VALID_NON_EVENT`, `HARD_NEGATIVE`만 독립 review truth로 들어오고
   label/review UUID가 존재한다.
7. 전용 model reader principal로 같은 view readback을 재현한다.

위 조건을 통과한 뒤에만 전체 row를 `training_cohort_uuid + cohort_digest`로 제한해 읽고,
모델 adapter가 `handoff_digest`를 독립 재계산한다. 그 다음 단계도 sync 검증이며 Kaggle
학습이나 성능 주장은 별도 게이트다.

## 재현 명령

```text
multisensor-ml model-platform preflight-bigquery \
  --output-receipt <new-create-only-receipt.json>
```

현재 검증 receipt:
`docs/evidence/2026-08-13-live-bigquery-training-preflight.json`
