# Kidsignal STEP-021 모델 ingestion 인계

검토일: 2026-08-13
범위: 로컬 공개 계약과 Cloud fixture 교차검증
배포·학습·Git 작업: 수행하지 않음

## 구현 결과

- 모델 validator가 공개 handoff body의 `handoff_digest`를 독립 재계산한다.
- BigQuery authorized flat row를 하나의 반복 envelope로 조립하는
  `synchronize_authorized_training_rows()`를 추가했다.
- source row 순서와 관계없이 Cloud와 같은 canonical row 순서와 digest를 만든다.
- dict와 `google.cloud.bigquery.table.Row`형 객체를 모두 받는다.
- `review_disposition`과 `evaluation_class` 조합을 고정하고 NEGATIVE의 `NO_EVENT`를 강제한다.
- `locked_access=false`, `sequence_group_key=training_capture_set_uuid`를 고정한다.
- 빈 입력, mixed envelope, LOCKED/unknown split, TRAIN/VALIDATION 불완전,
  중복 exact-window/source digest, private/unknown field를 모두 거부한다.

## 직접 교차검증

Cloud가 생성한 최신 Watch authorized-row fixture를 역순으로 모델 adapter에 넣었다.
Cloud adapter와 모델 adapter가 만든 payload 전체와 digest가 동일했다.

- runtime schema: `watch_standard_16_v2`
- schema hash: `2857f8a16cd4450a18f8701c1c9c9f397dee4b291fefd2475279883a0f8a29de`
- digest: `e84be7ebd7b3f345ce4494fbd2db8fe1b1b7df107d178218c477702f23876b47`
- validator: `VALID_SELECTION_HANDOFF`
- split: `TRAIN=1`, `VALIDATION=1`

기존 `d8314b0c...`는 platform description hash이므로 runtime row에서는 거부한다.
H10-only의 모델 측 새 제안은 `h10_standard_13_v1`/
`5ff8e9ceaa534bb8b7fafbdd7f844b06c0a4b21bb206428d5ce9239eec5459d6`지만 Cloud
producer가 아직 없어 `PROPOSED_NOT_IMPLEMENTED`다. Watch+H10은 계속
`PROPOSED_NOT_APPROVED`이며 둘 다 `training_ready=false`다.

## 아직 차단된 부분

reviewed event onset/end가 공개 계약에 없으므로 `window_start_ms/window_end_ms`를 사건
시간으로 오해하지 않는다. event delay와 forecast lead-time은 별도 reviewed event-anchor
계약 전까지 계산하지 않는다.

또한 live BigQuery training tables/view와 실제 reviewed frozen cohort가 없으므로 현재
상태는 `BLOCKED_NO_FROZEN_TRAINING_VIEW / NOT STARTED / NOT EVALUABLE`이다.
