# 장비별 개인 기준선 학습 연결부 v1 · 2026-09-06

이해한 범위: source별 authorized receipt로 자료를 읽고 TRAIN 기준선·VALIDATION 잔차·
로컬 모델 export까지 구현한다. 실제 raw/private 접근, Cloud 변경, 실데이터 fit는 하지 않는다.
상위: Cloud Phase06f/06g, commit 2e1e1201db7dd0fb51e5445e5bdee7bedf52154f.
모델 HEAD 5ea78f2 위 로컬 구현이다. Cloud가 다음 계약을 구현/승인했다는 뜻은 아니다.
CONSUMER_CONTRACT_PROPOSED / BLOCKED_NO_REAL_COHORT / TimesFM HOLD.

## 공개 실행 경로

`physiology_baseline_pipeline.run_physiology_baseline()`만 공개 fitting coordinator다.
receipt+schema+BigQuery client → receipt 검증 → fixed view의 UUIDv4 parameter query →
schema/order/row/공개 digest/분할 검증 → immutable PreparedPhysiologyDataset →
내부 TRAIN fit → VALIDATION descriptive residuals → 선택적 create-only JSON export다.
수동 arrays 또는 prepared dataset을 공개 fitting 함수 인자로 받지 않는다.

Watch/H10/Muse width는 3/3/24, domain 포함 hash는 Phase06f 최종 4e949a…/055bfc…/971396…다.
source별 cohort 하나씩 호출한다. 다른 source를 padding/교집합으로 요구하지 않는다.
기존 standard 24필드/behavior 26필드 reader와 legacy 16→15는 변경하지 않는다.

## 신규 authorized view 제안: 정확히 25필드

project `multi-app-kidsignal-260801`, location `asia-southeast1`.
view `kidsignal_model_training.physiology_baseline_train_validation_v1`만 조회한다.
이 이름은 로컬 제안이며 배포/권한 부여/실제 행 존재를 확인하지 않았다.
query parameter `cohort_uuid: STRING(UUIDv4)` 하나만 사용한다.
LOCKED가 노출되면 무시하지 않고 실패시킨다. private fallback은 없다.

| 순서 | 필드 | 타입/검증 |
|---:|---|---|
| 1 | cohort_uuid | STRING canonical UUIDv4 |
| 2 | source_kind | STRING watch/h10/muse, singleton |
| 3 | feature_schema_uuid | STRING pinned schema UUID |
| 4 | feature_schema_hash | STRING pinned lowercase SHA256 |
| 5 | native_hr_policy | STRING H10=polar_native_hr_bpm_v1, 나머지 NOT_APPLICABLE |
| 6 | split_policy | STRING CHRONOLOGICAL_PER_SUBJECT |
| 7 | purge_seconds | INTEGER >=0, receipt가 결정 |
| 8 | eligibility_policy | STRING REVIEWED_NO_PATTERN_V1 (아래 의미) |
| 9 | origin | STRING REAL_OBSERVED, 테스트만 CONTRACT_FIXTURE |
| 10 | public_cohort_digest | STRING lowercase SHA256 |
| 11 | public_split_digest | STRING lowercase SHA256 |
| 12 | training_subject_uuid | STRING cohort-scoped UUIDv5 |
| 13 | training_capture_set_uuid | STRING cohort-scoped UUIDv5, 한 subject 소유 |
| 14 | exact_window_id | STRING SHA256 |
| 15 | source_row_digest | STRING SHA256, Cloud가 검증한 원천 row 식별 |
| 16 | window_start_ms | INTEGER >=0, corrected UTC |
| 17 | window_end_ms | INTEGER, start+10000 |
| 18 | support_start_ms | INTEGER, v1은 window_start_ms와 동일 |
| 19 | support_end_ms | INTEGER, v1은 window_end_ms와 동일 |
| 20 | feature_values | JSON object, exact ordered core 이름의 finite numeric 값 |
| 21 | quality_valid | BOOLEAN true |
| 22 | training_eligible | BOOLEAN true |
| 23 | admission | STRING ELIGIBLE_NO_PATTERN |
| 24 | no_pattern_review_uuid | STRING canonical UUIDv4, 검토 증거 참조 |
| 25 | split_role | STRING TRAIN/VALIDATION |

JSON 문자열을 자동 파싱하는 우회는 없다. SDK에서 feature_values object로 반환해야 한다.
schema field 순서는 위 순서와 일치해야 한다. 반환 행 순서는 자유이며 아래 방식으로 정렬한다.
row는 위 exact set만 허용한다. 계정/person/raw/회원정보와 unknown 필드는 거부한다.
SOURCE_ROW digest는 opaque immutable lineage이며 private 원본을 reader가 역조회하지 않는다.

REVIEWED_NO_PATTERN_V1은 기존 NO_PATTERN evidence를 소비하는 의미 계약 제안이다.
새 서비스 라벨링 워크플로우·자동 정상 판정·신규 검토 권한을 승인하는 이름이 아니다.
기존 hour veto와 최신 제외 의도를 보존하고 unreviewed/REVIEW_REQUIRED/EXCLUDED_*는 제외한다.
Cloud composer는 기존 증거의 exact revision과 immutable content hash를 원천 row 생성에
결합하고, 해당 결합을 source_row_digest가 고정하는 source artifact에서 검증해야 한다.
no_pattern_review_uuid만으로 revision이 고정됐다고 간주하지 않는다. 현재 공개 25필드에는
별도 evidence_revision 필드가 없으므로 이 결합의 producer parity가 실데이터 인계 전 필수다.
기존 증거를 이 참조에 대응시킬 수 없으면 임의 UUID/revision을 생성하지 않고 인계를 차단한다.
모델은 private review를 조회하지 않는다. UUID 문법과 public digest 검사는 증거 검토의
진실성·revision binding 자체를 증명하지 않으며 Cloud의 exact evidence 검증에 의존한다.
동결 이후 invalidation은 Cloud가 해당 receipt를 철회하고 새 cohort를 만들어야 한다.

person/source/capture UUID는 Cloud 저장·전송 identity다. training_subject_uuid 및
training_capture_set_uuid는 해당 cohort 범위 가명 grouping key이며 직접 identity로 대체하지
않는다. source_kind는 장비 family이고 source UUID가 아니다. 가명↔직접 identity 대응은
Cloud가 보관한다. 신규 view 생성·IAM grant·positive cohort freeze는 아직 수행되지 않았다.

## receipt: 정확히 17필드

위 1~9 envelope + 다음 8개다:

| 필드 | 값/타입 |
|---|---|
| contract_version | physiology_baseline_handoff_v1 |
| public_cohort_digest | SHA256 |
| public_split_digest | SHA256 |
| train_row_count | INTEGER >0 |
| validation_row_count | INTEGER >0 |
| locked_access | false |
| fit_call_count | INTEGER 0 |
| handoff_digest | SHA256 |

create-only receipt는 Cloud의 승인 경로에서 받아야 한다. 자기서명 없는 hash는 무결성 검사이지
발급자 인증이 아니다. 운영 호출자는 기존 전용 model-reader principal의 인증된 client와
신뢰한 receipt를 제공해야 한다. 이 모듈은 IAM/principal을 발급하거나 검증했다고 주장하지 않는다.

## digest exact material

모든 hash는 UTF-8, ensure_ascii=false, sort_keys=true, separators=(',', ':'), allow_nan=false.
문자열 출력은 lowercase SHA256이다. 1과 1.0의 표현을 임의 변환하지 않는다.

1. rows를 (training_subject_uuid, window_start_ms, exact_window_id) 오름차순 정렬.
2. members = 각 row에서 public_cohort_digest/public_split_digest만 뺀 전체 dict 목록.
3. envelope = 위 1~9 필드 dict.
4. public_cohort_digest = hash({"envelope": envelope, "rows": members}).
5. public_split_digest = hash([[subject,capture,exact_window_id,split_role,
   support_start_ms,support_end_ms], ...])이며 같은 정렬 순서.
6. handoff_digest = hash(receipt에서 handoff_digest만 제외한 전체 dict).

source_row_digest, review UUID, feature 값, 품질/적격성과 envelope도 public cohort hash에
들어간다. 재실행/행 순서 변경은 같은 digest이며 unknown/mixed/digest mismatch는 fit 전 차단한다.

## 분할 및 적합

개인 baseline이므로 각 subject에 TRAIN과 VALIDATION이 모두 필요하다.
PERSON_GROUP은 이 개인 모델 v1에서 거부한다. unseen subject에 population baseline을
자동 적용하지 않는다. population reference는 별도 후속 모델 계약이다.
동일 subject의 support overlap/duplicate는 capture가 달라도 거부한다.
min(validation.support_start) >= max(train.support_end)+purge_seconds*1000을 요구한다.
purge=0도 시간 경계가 지켜지면 계약상 허용한다. 1800초/3시간 등을 자동 삽입하지 않는다.
positive row count는 인터페이스 최소조건일 뿐 통계적 충분성/임상 성능의 근거가 아니다.

feature별 TRAIN median을 center, 1.4826*MAD를 scale로 계산한다.
MAD=0일 때 TRAIN IQR/1.349 (linear quantile), 여전히 0/nonfinite면
BLOCKED_BASELINE_SCALE_ZERO이며 모델을 export하지 않는다. VALIDATION으로 대체하지 않는다.
VALIDATION은 (x-center)/scale의 feature별 mean/mean absolute residual을 기록한다.
이는 예측 MAE/정확도/스트레스 정답이 아니다. threshold·합산·stress 방향을 학습하지 않는다.

## 로컬 모델 export와 실행 구분

output_path와 export_request={model_release_uuid:UUIDv4, generated_at:UTC Z}를 함께 제공한다.
JSON은 physiology_baseline_artifact_v1이며 schema, receipt, 가명 subject별 center/scale,
TRAIN 행수와 validation 집계, origin/counters, release/time을 보존한다.
content_sha256은 자기 필드를 뺀 canonical document hash, 반환 artifact_sha256은 파일 bytes hash다.
create-only(xb), 같은 요청/자료는 동일 bytes. read_baseline_artifact로 무결성·모델 shape 확인 후
residual()이 같은 subject의 feature vector에 기준선을 적용한다.
계정/person/raw/row-level feature/개별 validation residual은 저장하지 않는다.
이 JSON은 새 모델 형식이며 기존 Cloud ONNX ZIP verifier/등록에 호환된다고 주장하지 않는다.
Cloud residual consumer/export 형식 승인·배포는 별도다. delivery/promotion은 false, stage=null.

CONTRACT_FIXTURE: 수치 적합 테스트만 fixture_fit_call_count=1, fit_call_count=0,
CONTRACT_EVIDENCE_ONLY. 테스트 임시 artifact는 운영 모델로 취급하지 않는다.
REAL_OBSERVED: 신뢰한 real receipt와 일치해야 하고 allow_real_fit=True 명시 시에만 실행한다.
fit_call_count는 내부 적합 호출 횟수이며 실패 후에도 호출 사실을 감추지 않는다.
최종 성능은 NOT VERIFIED이며 현재 실제 양수 cohort가 없어 real fit은 실행하지 않았다.

## Cloud 다음 인계

새 25필드 view와 17필드 receipt의 producer 승인/교차검증 → NO_PATTERN 검토 → source별
기간·purge 동결 → 전용 reader destination readback → 실제 source별 실행 순서다.
새 view 권한/IAM·공개·backfill은 이번 모델 작업에서 실행하지 않는다.
3/3/24 consumer가 있다고 해서 기존 legacy receipt가 신규 schema를 수락하는 것은 아니다.
