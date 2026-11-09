# 개인 관측분포 탐색 기준선 v1

상태: REAL_OBSERVED / UNREVIEWED / EXPLORATORY_PERSONAL_REFERENCE.
이 경로는 기존 REVIEWED_NO_PATTERN_V1 학습·서비스 계약과 별개다.
미검토 자료를 정상·휴식·행동 음성 라벨로 바꾸지 않는다.

## 입력 계약

`exploratory_personal_dataset_v1` feature-only JSON과 외부 전체 파일 SHA256을 함께 받는다.
최상위 exact 필드는 purpose/origin/label/reference 의미, cohort와 가명 subject,
source 종류, 전체 pinned feature schema, snapshot·lineage·scope digest, 시간순 split,
aggregate selection, rows다. self digest는 사용하지 않는다.

row는 가명 capture, exact window/source-row digest, 10초 support, TRAIN/VALIDATION,
exact core feature, quality_valid=true, admission_state=REVIEW_REQUIRED만 허용한다.
직접 person/source/capture UUID, 계정, raw 경로, training_eligible 필드는 거부한다.
source 하나의 실패는 다른 source 적합을 막지 않는다.

split은 `CHRONOLOGICAL_FIRST_70_PERCENT`, purge 10초다. feature 값으로 경계를 고르지 않는다.
모든 support를 시간순 정렬하고 중복·overlap·split 경계 위반을 fit 전에 거부한다.
동일 실제 참여자라는 확인은 Cloud의 scope_digest와 source별 하나의 cohort-scoped subject로
전달된다. source별 가명 UUID가 같지 않으므로 이를 서로 다른 사람 성능으로 해석하지 않는다.

## 적합과 출력

feature별 TRAIN median을 center로, `1.4826*MAD`를 scale로 쓴다. MAD가 0이면
TRAIN IQR/1.349를 사용하며 이것도 0이면 해당 source 적합을 실패시킨다.
VALIDATION은 `(x-center)/scale` 잔차만 계산하며 center/scale 선택에 사용하지 않는다.
랜덤성이 없고 seed 0을 기록한다.

출력은 `exploratory_personal_reference_model_v1` create-only JSON이다. feature 순서,
count/center/scale/scale source, validation residual과 집계, 원천·scope·schema digest,
모델/parameter hash, 실제 fit counter를 보존한다. 파일은 0600이다.
현재 feature 적용 함수는 schema hash와 exact feature set을 확인하고 feature별 현재 잔차를 반환한다.

결측 feature의 연구용 개인 기준 대체에는
`read_verified_exploratory_baseline(model_path, expected_model_sha256, receipt_path,
expected_receipt_sha256, expected_training_subject_uuid, expected_source_kind,
expected_feature_schema_hash)`를 사용한다. 이 함수는 current feature를 받거나 apply를 호출하지 않고,
model·receipt를 각각 한 번 읽어 외부 SHA, 내부 digest, 상호 binding, subject/source/pinned schema를
검증한 뒤 core feature별 `code/unit/baseline_center/baseline_scale`만 반환한다. center는 source
domain, scale은 finite positive 조건을 만족해야 한다. 출력 의미는
`PERSONAL_BASELINE_PARAMETERS`, label은 `UNREVIEWED`, delivery/promotion은 false다.
직접 회귀 명령은 `.venv/bin/pytest -q -p no:cacheprovider
tests/test_verified_exploratory_reference.py`다.

값은 선택한 개인 관측분포 기준 이탈이다. 시간 누적합, 확률, 예측 정확도, 공통 0–100 점수,
확인된 휴식 기준 또는 임상 상태가 아니다. delivery/promotion은 false이고 stage는 null이다.

## 실제 실행 · 2026-09-07

- Watch: TRAIN 119 / VALIDATION 52, fit_call_count=1, readback verified.
- H10: TRAIN 380 / VALIDATION 162 / PURGED 1, fit_call_count=1, readback verified.
- Muse: native time/sampling 미검증으로 입력 dataset이 없어 미실행.
- 두 입력의 scope_digest는 같아 같은 실제 참여자 범위다. source별 cohort 가명은 별도다.
- TimesFM, Cloud 쓰기, IAM, 배포, commit/push, 기존 artifact overwrite는 수행하지 않았다.

세부 수치와 SHA는
`reports/exploratory_personal_reference/2026-09-07-real-observed/REPORT_KO.md`와
`summary.json`에 고정한다.
