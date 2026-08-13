# Kidsignal 모델 플랫폼 계약 설계

작성일: 2026-08-12
상태: 모델 저장소 구현 기준

## 1. 저장소 소유권

- `multisensor_ml`이 feature contract, 동결 cohort 검증, 기준 reference, 학습 후보,
  평가 증거와 Cloud 인계 bundle을 소유한다.
- `multisensor_synth`는 truth/observed/model_ready를 분리해 생성하는 합성 truth
  엔진이다. 운영 모델 registry나 Cloud serving의 소유 저장소가 아니다.
- 기존 Goal 1.5 및 monitoring-v2 결과는 `oracle/sanity`다. 실제 표준 성능이나
  실제 행동 성능으로 승격하지 않는다.
- 기존 observational Watch bundle도 독립 실데이터 cohort와 locked holdout 증거가
  없으므로 `NOT VERIFIED`를 유지한다.

## 2. 플랫폼 경계

- Android/Watch/H10: 원천 신호, vendor-native observation, capture metadata만 수집한다.
- Cloud Run: corrected UTC, 1 Hz 조립, 품질, 개인 baseline, deviation, pattern/stage,
  ACTIVE/CANDIDATE inference를 담당한다.
- GCS: raw 및 committed artifact의 canonical lake다.
- DuckDB `:memory:`: 즉시 exact-window 흐름·품질·지연 관찰에만 쓴다.
- BigQuery: GCS projection, 장기 모니터링, 검토 truth, 동결 training cohort를 제공한다.
- Kaggle/모델 프로젝트: `training_cohort_uuid + cohort_digest`로 동결된 view만 읽는다.
- Neon: migration-only legacy이며 신규 학습·serving 원본으로 쓰지 않는다.

## 3. 모델 identity

모델에는 `account_uuid`, `membership_uuid`, legacy `person_key`가 유입되면 안 된다.
코호트 범위 가명인 `training_subject_uuid`만 사용한다. 데이터 identity는
`exact_window_id=SHA-256(canonical corrected-time window bytes)`로 고정한다.

## 4. 센서 조합

세 모델 입력은 서로 다른 immutable schema다.

1. `watch_only`: Watch EDA, HR, ACC 기반 16개 causal feature
2. `h10_only`: H10 HR/RR/ECG 품질 기반 13개 causal feature; H10 ACC는 사용하지 않음
3. `watch_h10`: 두 schema와 fused load/agreement를 합친 32개 feature

필수 source가 없거나 품질이 실패하면 `NOT_DECISIONABLE`이다. 생체 신호를 0 또는
null로 채워 다른 조합 모델에 넣지 않는다.

단, fused load/agreement는 현재 `PROPOSED/NOT VERIFIED`이며 승인된 Cloud serving
feature가 아니다. 현재 exact composition은 source manifest scheduling 교집합과
corrected-time provenance만 수행한다. cross-device waveform alignment, offset/lag 추정,
보간, 리샘플링은 별도 연구·계약 전까지 금지한다.

## 5. 기준선과 후보의 분리

- 개인 baseline: 과거 900 eligible seconds의 robust center/scale
- decision warm-up: 총 1,800 eligible seconds
- persistence: 현재 load를 다음 30분 평균의 단순 예측으로 사용
- rolling-median-300: 과거 300초 load median을 단순 예측으로 사용
- strongest simple reference: 오직 train 구간 MAE로 위 둘 중 하나를 사전 선택
- candidate/adapter: validation에서만 선택·보정하며 locked holdout은 최종 1회 비교

selection view에는 train+validation만 노출한다. split은 `person_group` 또는
`chronological_per_subject` 중 하나를 사전등록한다. locked holdout row/label은 notebook에
노출하지 않으며 sealed evaluator가 최종 immutable bundle만 평가해 aggregate 비교만
반환한다. row-level FP/FN은 holdout retirement 뒤 신규 cohort 후보로만 사용할 수 있다.

현재 실제 동결 cohort가 없으므로 이 구현은 contract fixture ONNX만 만든다. 실제
candidate/adapter, threshold, FP/FN 사례는 만들지 않는다.

## 6. 라벨과 평가

stage 1~5는 severity가 아니라 시간 순서다. `stage >= k`를 severity truth로 쓰지 않는다.
FPR은 독립 review가 끝난 `valid_non_event`와 `hard_negative`만 분모로 사용하고, FNR은
`target_event`만 사용한다. `novel_pattern`, `unreviewed`, 품질 실패, correction 대상은
오탐으로 세지 않는다.

실제 라벨·review·locked holdout이 없을 때 결과는 `NOT EVALUABLE`, 실제 정확도는
`NOT VERIFIED`다.
