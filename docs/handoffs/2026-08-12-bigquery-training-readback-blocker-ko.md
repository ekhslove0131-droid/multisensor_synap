# Kidsignal BigQuery 학습 readback 및 차단 인계

관찰 시각: 2026-08-12T08:52:54Z  
실행 모드: 읽기 전용

## 결론

GCP BigQuery와 Kaggle API 인증은 실제 readback으로 확인됐다. 그러나 학습에 필요한
authorized/materialized frozen cohort view가 없으므로 실제 동기화와 모델 학습은
fail-closed 중단했다.

현재 상태는 다음과 같다.

- dataset: `multi-app-kidsignal-260801.kidsignal_validation`
- `artifact_monitoring_v1`: 관찰 시점 9,574행, streaming 중
- `model_validation_v1`: 0행
- VIEW: 없음
- 모든 9,574행에 legacy `person_key` 존재
- accuracy/FPR/residual drift가 있는 행: 각각 0
- 집계된 truth 상태: `truth_not_mature`
- source set: Watch, H10, Watch+H10 모두 관찰됨
- Watch+H10 route: `no_model`

따라서 이 projection은 운영 흐름 모니터링 증거로는 쓸 수 있지만 동결 학습 cohort나
독립 평가 truth로 사용할 수 없다.

## Cloud/BigQuery에 필요한 객체

### Canonical projection tables

1. `committed_artifacts_v1`
   - `artifact_uri`, `generation`, `checksum`, `artifact_sha256`
   - `schema_version`, `source_digest`, corrected UTC bounds
   - `source_uuid`, `source_set`, `capture_session_uuid`, `exact_window_id`
   - GCS commit 상태와 quality/readiness
2. `feature_sets_v1`
   - `feature_set_uuid`, `feature_schema_uuid`, `feature_schema_hash`
   - feature artifact URI/generation/SHA와 source variant
3. `label_revisions_v1`
   - `label_uuid`, `label_revision_uuid`, label source, temporal stage
   - `target_event` 여부와 event/time bounds
4. `independent_reviews_v1`
   - `review_uuid`, `label_revision_uuid`, reviewer provenance/time
   - disposition: `target_event`, `valid_non_event`, `hard_negative`,
     `artifact_or_quality_failure`, `correction_required`, `novel_pattern`, `unreviewed`
5. `training_cohort_manifests_v1`
   - `training_cohort_uuid`, `cohort_digest`, materialization URI/SHA
   - feature/label/review lineage digest, source variant, split strategy, purge seconds
6. `training_cohort_windows_v1`
   - `training_cohort_uuid`, `training_subject_uuid`, row lineage UUID
   - `split_role`, window bounds, `exact_window_id`, artifact URI/SHA

위 필드는 Cloud 내부 lineage projection이다. model/Kaggle 공개 view에는
`account_uuid`, `membership_uuid`, legacy `person_key`뿐 아니라 직접 person/source/capture
UUID도 노출하지 않고 cohort-scoped `training_subject_uuid`와
`training_capture_set_uuid`로 변환한다.

### Selection view

`authorized_training_cohort_v1` 또는 동일 목적 materialized view가 필요하다.

- 필수 query key: `training_cohort_uuid + cohort_digest`
- 반환 split: `train`, `validation`만
- split 전략: cohort 동결 전에 `person_group` 또는 `chronological_per_subject` 사전등록
- target 경계 purge: 최소 1,800초
- source variant: `watch_only`, `h10_only`, `watch_h10` 분리
- 누락 source: 생체값 0/null padding 금지
- GCS artifact URI+generation+checksum/SHA lineage 고정

### Locked holdout

locked holdout row와 label은 Kaggle notebook이나 selection view에 노출하지 않는다.
별도 sealed evaluator가 validation 선택이 끝난 immutable final bundle 하나만 받아 동일
cohort digest/evaluation input에서 aggregate 결과만 반환해야 한다. active holdout의 row-level
FP/FN은 모델 학습 쪽으로 반환하지 않으며 holdout retirement 뒤에만 새 cohort 후보가 된다.

## 최소 읽기 권한

모델/Kaggle 전용 service account에 최소 권한을 부여한다.

- 프로젝트: BigQuery Job User (`bigquery.jobs.create`)
- publish dataset/view: BigQuery Data Viewer (`tables.get`, `tables.getData`)
- BigQuery Storage API 사용 시: BigQuery Read Session User
- exact URI로 GCS artifact를 내려받을 때: 해당 committed artifact prefix에
  Storage Object Viewer (`storage.objects.get`; 목록 동기화가 필요할 때만 list)

source dataset 전체 권한이나 BigQuery Data Editor는 필요하지 않다. 개인 ADC를 Kaggle에
복사하지 말고 전용 service account/WIF 또는 승인된 secret 주입을 사용해야 한다.

## 재개 조건

다음 값이 제공된 후에만 동기화와 학습을 다시 시작한다.

1. 실제 `training_cohort_uuid`
2. 일치하는 64자리 `cohort_digest`
3. train+validation 전용 authorized/materialized view 이름
4. immutable split manifest와 feature schema UUID/hash
5. GCS artifact URI/generation/SHA read 권한
6. 독립 label revision/review support 집계
7. locked holdout sealed evaluator endpoint 또는 실행 계약

현재 판정:

- sync: `BLOCKED_NO_FROZEN_TRAINING_VIEW`
- training: `NOT STARTED`
- evaluation: `NOT EVALUABLE`
- actual performance: `NOT VERIFIED`
