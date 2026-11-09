# Kidsignal Cloud ↔ 모델 handoff 정합성 검토

검토일: 2026-08-13  
Cloud branch/head: `codex/kidsignal-gcs-realtime` / `3c5678c`  
실행 범위: Cloud 읽기 전용, 모델 로컬 계약·테스트만 수정

## 결론

`training_capture_set_uuid`는 모델 sequence group key로 받을 수 있다. Cloud 구현은
`UUIDv5(training_cohort_uuid, capture-session set)` 형태로 cohort 범위 가명을 만들므로
직접 capture UUID를 공개하지 않으면서 동일 cohort 안의 continuity를 묶는다.

모델에서는 다음 제한을 적용한다.

- canonical UUID string만 허용
- 개인 identity로 사용 금지
- 한 capture-set UUID는 정확히 한 `training_subject_uuid`에만 대응
- 같은 cohort 안의 시간 정렬·sequence grouping에만 사용
- cohort가 바뀌면 같은 실제 capture라도 다른 가명이어야 함

## 일치하는 계약

- train/validation만 handoff하고 LOCKED는 제외
- cohort UUID/digest와 split digest 고정
- cohort-scoped subject/capture-set UUID만 공개
- source set, exact-window SHA, schema UUID/hash, label/review UUID 제공
- review가 완료된 POSITIVE/NEGATIVE만 training example에 포함
- 5단계는 시간 단계이며 severity threshold가 아님
- 직접 identity, profile/auth, raw, GCS lineage는 모델 view에서 제외

## 모델 쪽에서 수정한 충돌

1. 기존 모델 view가 직접 `source_uuid`, `capture_session_uuid`, artifact URI를 요구하던
   부분을 제거하고 `training_capture_set_uuid + source_row_digest`로 교체했다.
2. Cloud의 millisecond time bounds와 대문자 `TRAIN/VALIDATION`을 받는다.
3. source set mapping을 고정했다.
   - `watch` → `watch_only`
   - `h10` → `h10_only`
   - `h10,watch` → `watch_h10`
4. `feature_values`는 schema의 ordered names와 정확히 같은 flat finite numeric object만
   허용한다. nested/list/null/bool이나 key 누락·추가는 거부한다.
5. NEGATIVE row에도 Cloud schema상 temporal stage가 존재하지만 모델은 이를 단계 truth로
   사용하지 않고 `NO_EVENT`로 처리한다.

## 아직 남은 차단 및 필드 충돌

1. 모델의 30분 target split에는 최소 1,800초 purge가 필요한데 Cloud cohort schema와
   handoff에 `purge_seconds` 증거가 없다.
2. `TrainingHandoff` dataclass 자체에는 `contract_version`, `split_policy`,
   `sequence_group_key`, `locked_access`, `truth_state`가 없다. 계획 문서의 Gate D envelope와
   실제 직렬화 payload가 하나로 고정되어야 한다.
3. Cloud 내부 feature JSON 검증기는 nested/list/null/bool도 일부 허용한다. handoff 공개
   view 또는 producer에서 schema의 정확한 numeric feature만 보장해야 한다.
4. `temporal_stage`가 NEGATIVE에도 REQUIRED다. 모델은 무시하도록 방어했지만 장기적으로는
   `stage_applicable=false` 또는 nullable stage가 더 명확하다.
5. live GCP에는 아직 training tables/view/producer/real cohort가 없다.

## 현재 판정

- `training_capture_set_uuid`: **ACCEPTED AS COHORT-SCOPED SEQUENCE KEY**
- live cohort sync: `BLOCKED_NO_FROZEN_TRAINING_VIEW`
- model training: `NOT STARTED`
- threshold/adapter: `NOT STARTED`
- locked evaluation: `NOT STARTED`
- actual accuracy: `NOT VERIFIED`

합성 fixture는 계약 회귀검사로만 유지하며 실제 학습 성공 증거로 사용하지 않는다.
