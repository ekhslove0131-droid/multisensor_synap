# Kidsignal 모델 플랫폼 v1

## 결론

Canonical 모델 저장소는 `multisensor_ml`이고 `multisensor_synth`는 합성 truth 생성기다.
이번 구현은 Cloud와 맞출 UUID/hash 계약 및 세 센서 조합별 단순 reference를 재현한다.
현재 실데이터 BigQuery cohort와 독립 라벨이 없으므로 실제 표준모델·행동모델 성능은
아직 만들거나 주장할 수 없다.

## 데이터와 모델 흐름

```text
Android raw/vendor observation
  -> GCS committed exact window
  -> BigQuery frozen authorized/materialized cohort
  -> multisensor_ml/Kaggle train + validation selection
  -> immutable CPU ONNX + manifests + hashes
  -> Cloud Run ACTIVE/CANDIDATE shadow inference
```

DuckDB `:memory:`는 이 흐름의 임시 관찰자이고, Neon은 migration-only legacy다.

## Runtime schema identity 정정

Watch runtime의 authoritative feature schema는 다음이다.

- version: `watch_standard_16_v2`
- UUID: `9b842d8c-8889-5259-acca-77baa0c7729d`
- runtime hash: `2857f8a16cd4450a18f8701c1c9c9f397dee4b291fefd2475279883a0f8a29de`
- status: `CANONICAL_RUNTIME`

`d8314b0c...`는 같은 16개 피처 외에 모델 플랫폼의 수식·출처·상태 설명을 포함한
description JSON의 무결성 해시다. Cloud derived feature artifact나 training row의
`feature_schema_hash`로 사용하지 않는다. 모델 validator는 `d8314b0c...`를 runtime
identity로 받으면 거부한다.

H10-only는 실제 `kidsignal_polar_h10_ecg_v2` 필드에 맞춰
`h10_standard_13_v1`/`8ba3dbff-e5c4-5dd3-be1d-69ea6cea8966`/
`5ff8e9ceaa534bb8b7fafbdd7f844b06c0a4b21bb206428d5ce9239eec5459d6`로 다시
고정했다. 기존 `f2d625...` placeholder는 더 이상 canonical hash가 아니다. 다만 Cloud
decoder/producer가 아직 없으므로 H10-only 상태는 `PROPOSED_NOT_IMPLEMENTED`이며,
Watch+H10 `10f0be...`도 `PROPOSED_NOT_APPROVED`다. 두 variant 모두 backend producer와
실제 reviewed cohort가 생기기 전까지 학습·서빙 준비 완료로 표시하지 않는다.

## 원천값과 주요 파생변수

### Galaxy Watch8

- 원천: `heart_rate_bpm`, `eda_us`, `acc_x/y/z_mps2`, corrected UTC, clock/presence/quality
- 파생: EDA/HR/motion robust z, 양의 부하 평균, 30초 평균, 60초 OLS slope,
  60초 load 표준편차, 300초 median, 1,800/21,600 eligible-second EMA

```text
watch_load_raw = mean(max(EDA_z,0), max(HR_z,0), max(motion_z,0))
```

### Polar H10

- 원천: `heart_rate_bpm`, `rr_interval_ms`, `ecg_uv@130Hz`, battery, quality metadata
- H10 ACC는 계약에서 제외
- 파생: HR robust z, `-robust_z(RMSSD)`, 위 두 양의 값의 평균 load와 causal window

```text
h10_load_raw = mean(max(HR_z,0), max(-RMSSD_z,0))
```

ECG는 현재 진단 타깃이 아니라 RR/접촉·신호 품질 provenance다.

### Watch + H10

```text
fused_load_raw = mean(watch_load_raw, h10_load_raw)
```

위 `fused_load_raw`, 그 300초 median, 두 source의 60초 차이를 이용한
`source_agreement_60`은 아직 실제 cohort로 검증하지 않은 `PROPOSED/NOT VERIFIED`
계산이다. Cloud feature 계약 승인 전 배포 입력으로 요구하지 않는다. 하나가 빠지면
fused 값을 만들지 않고 조합 전체를 `NOT_DECISIONABLE`로 처리한다.

현재 exact composition은 source manifest scheduling 교집합과 corrected-time provenance만
사용한다. cross-device waveform alignment, offset/physiological lag 추정, 보간,
리샘플링은 구현되어 있지 않으며 별도 연구·계약 전에는 금지한다.

## baseline과 표준 예측

- 개인 baseline: 이전 900 eligible seconds
- scale: MAD 기반 robust scale
- decision warm-up: 1,800 eligible seconds
- 다음 30분 target: 실제 동결 cohort 계약이 마련된 후 정확히 version 고정
- strongest simple reference 후보: persistence, rolling median 300

reference 선택은 train만 사용한다. validation은 candidate/adapter 선택에 사용한다.
모델 notebook의 selection view는 train+validation만 반환하며 locked holdout row/label은
노출하지 않는다. sealed evaluator가 동결된 최종 immutable bundle 하나만 받아 aggregate
최종 비교를 수행한다. split 전략은 `person_group` 또는 `chronological_per_subject` 중
하나를 cohort 생성 전에 등록한다.

## stage와 행동 패턴

| 단계 | 시간 의미 | 요약 |
|---|---|---|
| 1 | pre_early | before |
| 2 | pre_late/onset | before |
| 3 | peak | during |
| 4 | recovery_early | after |
| 5 | recovery_late/post | after |

단계 숫자는 심각도가 아니다. 합성 pattern/behavior 라벨은 oracle/sanity 용도이며 실제
행동 판정으로 승격하지 않는다. 실제 pattern 학습은 독립 관찰 label revision과 review가
동결 cohort에 포함된 뒤 별도 head/adapter로 만든다.

## Cloud가 구현해야 할 것

### API/ledger 필드

- model handoff: `training_cohort_uuid`, `cohort_digest`, `split_digest`, `handoff_digest`
- model handoff: cohort-scoped `training_subject_uuid`, `training_capture_set_uuid`
- model handoff: `exact_window_id`, `source_set`, corrected window millisecond bounds
- model handoff: `feature_schema_uuid`, `feature_schema_hash`, canonical numeric features
- `label_uuid`, `label_revision_uuid`, `review_uuid`, `review_disposition`
- `evaluation_class`, temporal stage, observation code, split role, source row digest

`handoff_digest`는 자신을 제외한 공개 handoff 전체 body를 UTF-8 canonical JSON
(`sort_keys=true`, `separators=(',', ':')`, `allow_nan=false`)으로 직렬화한 SHA-256이다.
모델은 이를 독립 재계산하고 불일치를 거부한다. BigQuery flat row의 입력 순서와 관계없이
`split_role, training_subject_uuid, window_start_ms, exact_window_id` 순으로 정렬한 뒤 하나의
envelope로 조립한다.

review truth 조합은 `TARGET_EVENT -> POSITIVE`, `VALID_NON_EVENT -> NEGATIVE`,
`HARD_NEGATIVE -> NEGATIVE`만 허용한다. NEGATIVE의 `observation_code`는 `NO_EVENT`이고
`temporal_stage`는 audit 값일 뿐 모델 truth로 사용하지 않는다.

Cloud 내부 projection에는 person/source/capture UUID와 GCS lineage가 존재할 수 있지만,
모델/Kaggle view에는 직접 UUID, profile/auth/account/membership, raw, GCS credential lineage,
LOCKED row/label을 노출하지 않는다. 모델은 `training_capture_set_uuid`를 동일 cohort 안의
capture continuity/sequence grouping에만 사용하며 개인 identity로 사용하지 않는다.

### BigQuery table/view

- Cloud 내부 committed training examples와 frozen cohort manifests
- 모델 공개 view에는 train/validation 행과 cohort-scoped pseudonym만 projection
- `training_cohort_uuid + cohort_digest`로 제한하는 authorized/materialized training view
- 모델 sync adapter는 반복 envelope가 하나인지, TRAIN/VALIDATION이 모두 있는지,
  LOCKED/unknown split·중복 window/digest·private/unknown field가 없는지 fail-closed 검증
- dict뿐 아니라 `google.cloud.bigquery.table.Row`처럼 `items()`를 제공하는 SDK row 지원

## 모델 프로젝트가 보장하는 것

- 입력 feature의 정확한 순서·수식·dtype·shape와 schema UUID/hash
- cohort digest/identity/분할 검증 및 account/membership/legacy key 차단
- `training_capture_set_uuid`의 cohort 범위 sequence grouping과 subject 단일성 검증
- Watch-only runtime 계약의 CPU ONNX bundle과 파일 SHA-256
- H10-only는 `h10_standard_13_v1` exact formula를 제공하되
  `PROPOSED_NOT_IMPLEMENTED`, Watch+H10은 `PROPOSED_NOT_APPROVED` 및
  `training_ready=false`로 명시
- threshold/adapter 선택에 validation만 사용하고 locked holdout 누출 차단
- 독립 truth가 없는 사례를 FP/FN으로 날조하지 않는 평가 manifest
- 운영/adjudicated FP/FN만 모델 개선 후보로 반환하고, active locked holdout row-level
  error slice는 반환하지 않음
- 실제 성능은 증거가 생길 때까지 `NOT VERIFIED`
- reviewed event onset/end 계약이 생기기 전에는 event delay와 forecast lead-time을 계산하지 않음

## 실데이터 전 구현 완료 경계

- 표준 plane은 `standard_baseline_train_validation_v1`의 정확한 24개 필드만 읽고
  `standard_cohort_uuid`와 예상 public cohort/split digest를 필수로 고정한다.
- 행동 plane은 `training_examples_train_validation_v1`의 정확한 26개 필드만 읽고
  `training_cohort_uuid`와 예상 public cohort/split digest를 필수로 고정한다.
- 두 plane의 조회 계약은 각각 `standard_cohort_query_contract()`와
  `behavior_cohort_query_contract()`로 제공한다. 같은 parameterized SQL을 로컬 `bq` CLI와
  Kaggle의 `google.cloud.bigquery` SDK에서 재사용할 수 있지만, 조회 계약 자체는 실행기가
  아니다. 실제 SDK 실행은 각각 `run_read_only_standard_cohort_sdk_reader()`와
  `run_read_only_behavior_cohort_sdk_reader()`가 담당하며 UUIDv4 STRING parameter,
  project `multi-app-kidsignal-260801`, location `asia-southeast1`을 고정한다. private table이나
  LOCKED row를 조회하는 fallback transport는 제공하지 않는다.
- SDK reader는 주입된 `google.cloud.bigquery.Client` 호환 객체의 `query().result()`를
  실행하고 반환 schema의 정확한 24/26 필드 순서와 Row-like `items()`를 검증한다.
  `google-cloud-bigquery`는 라이브러리 import의 필수 의존성이 아니며 실제 SDK 실행 경계에서만
  지연 import한다. 테스트는 외부 GCP 연결 없이 narrow client와 job-config factory를 주입한다.
- SDK가 반환한 Row-like 객체는 각 plane의 동기화·readiness validator를 통과한 뒤에만
  prepared dataset으로 변환한다. 조회 순서와 무관하게 public digest를 독립 재계산한다.
- 표준 prepared dataset은 runtime Watch 16개 배열과 trainer 15개 배열을 별도로 보존한다.
  `watch_load_median_300`은 target source이므로 trainer 배열에서 제외한다.
- person-group은 TRAIN/VALIDATION subject 교집합 0을 요구한다. 동일 사람 chronological
  split은 validation 시작 전 최소 1,800초 purge를 요구한다.
- 0행, schema/digest/identity 불일치, LOCKED, private field, 누출 또는 split 위반은
  trainer 호출 전에 fail-closed 처리하며 receipt의 `fit_call_count`는 0이다.

### Kaggle 실제 cohort intake

- `kaggle/10_kidsignal_bigquery_intake.ipynb`는 기존 01~09 합성·재현·benchmark와 분리된
  실제 cohort 전용 read-only intake다. JSON은 직접 편집하지 않고
  `scripts/build_kaggle_bigquery_intake_notebook.py`로 생성한다.
- 실행 mode는 `standard`(24 fields)와 `behavior`(26 fields)뿐이다. 각 mode는 UUIDv4
  cohort UUID 및 사전에 전달받은 public cohort/split digest를 요구하고, 기존 SDK reader를
  호출한다. SQL과 field 목록을 notebook에 복제하지 않는다.
- `google-cloud-bigquery`는 notebook의 실제 실행 cell에서만 필요한 지연 runtime dependency다.
  `multisensor_ml` wheel/package와 Kaggle runtime ADC는 notebook 밖에서 준비하며, notebook은
  dependency 설치나 credential JSON·token 기록을 수행하지 않는다. 전용 model-reader
  principal을 확인할 수 없으면 즉시 실패한다.
- 출력은 row payload가 아니라 bounded readiness receipt, split count/class count와 배열 shape만
  담는다. 표준 mode는 runtime 16개와 target source를 제외한 trainer 15개 shape를 분리한다.
  account/person/membership UUID, training subject/capture pseudonym, raw URI 및 feature row는
  출력하거나 저장하지 않는다.
- 0행이나 schema/digest/identity/split/class 위반은 `BLOCKED_*`로 끝나며, 성공해도
  `READY_FOR_SYNC_NOT_TRAINED`에서 정지한다. fit·평가·ONNX·bundle·승격은 모두 실행하지 않고
  `fit_call_count=0`을 유지한다.

실제 cohort가 도착하기 전에는 query/schema/digest/shape 계약만 검증한다. fixture는 이
계약의 회귀 테스트이며 모델 fitting, 성능 평가 또는 bundle 생성 입력이 아니다.

## 현재 차단 요인

2026-08-21 app/backend handoff 증거 기준으로 두 authorized view는 존재하지만 표준 및 행동
example/cohort/member와 view가 모두 0행이다. 따라서 현재 차단은 view 부재가 아니라 실제
quality-valid frozen cohort 부재다.

1. 양수 frozen `standard_cohort_uuid`와 public cohort/split digest 미제공
2. 표준 TRAIN/VALIDATION을 구성할 quality-valid no-pattern hour 부재
3. 행동 plane의 독립 reviewed positive/negative truth support 부재
4. H10-only와 Watch+H10 실제 cohort 없음
5. reviewed event onset/end anchor 계약이 없어 delay/lead-time 평가는 차단됨

따라서 이번 local fixture bundle은 인터페이스·ONNX·hash 재현 증거이지 배포 모델이 아니다.
실제 frozen cohort가 제공되기 전에는 “실제 표준모델 생성 완료”가 아니라
`BLOCKED_NO_REAL_COHORT / NOT STARTED / NOT EVALUABLE` 상태이며 `fit_call_count=0`이다.
