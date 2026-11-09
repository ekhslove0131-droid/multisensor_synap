# Kidsignal 현재 피처 명세와 학습 준비 상태

> 기준일: 2026-08-14  
> 범위: 실제 관측 데이터용 모델 계약. 합성 `oracle/sanity` 성능과 분리한다.  
> 현재 결론: **모델 reader 코드는 준비됐지만 실제 학습은 불가능하다.**

## 1. 결론

- 현재 학습 입력의 canonical source는 BigQuery authorized view
  `multi-app-kidsignal-260801.kidsignal_model_training.training_examples_train_validation_v1`
  하나다.
- 모델은 `person_uuid + 날짜`로 직접 조회하지 않는다. 모델 reader의 조회 키는 정확히
  하나의 `training_cohort_uuid`다.
- 날짜는 cohort row의 `window_start_ms`, `window_end_ms`에 보존되며 시간순 분할과 purge
  검증에 사용한다.
- Cloud 내부에서 `person_uuid + 기간 + reviewed truth`로 cohort를 동결하고, 모델에는
  cohort 범위 가명인 `training_subject_uuid`, `training_capture_set_uuid`만 제공해야 한다.
- 최신 live readback은 authorized view **0행, 24필드**다. 26필드 migration은 코드에
  반영됐지만 live 미적용이므로 현재 reader는 fail-closed한다.
- 현재 상태는 `BLOCKED_NO_REAL_COHORT / NOT STARTED / NOT EVALUABLE`이며 fit/train
  호출 수는 0이다.

## 2. 센서 variant와 고정 identity

| variant | schema version | schema UUID | runtime hash | 상태 |
|---|---|---|---|---|
| Watch-only | `watch_standard_16_v2` | `9b842d8c-8889-5259-acca-77baa0c7729d` | `2857f8a16cd4450a18f8701c1c9c9f397dee4b291fefd2475279883a0f8a29de` | `CANONICAL_RUNTIME` |
| H10-only | `h10_standard_13_v1` | `8ba3dbff-e5c4-5dd3-be1d-69ea6cea8966` | `5ff8e9ceaa534bb8b7fafbdd7f844b06c0a4b21bb206428d5ce9239eec5459d6` | `PROPOSED_NOT_IMPLEMENTED` |
| Watch+H10 | `PROPOSED_NO_RUNTIME_SCHEMA_VERSION` | `6802252a-f5ef-5a04-92ba-37378d75cd44` | `10f0be5737cea4965f8ca9552f4f8ae27154b3c195da7ff266e5703ae7d55b66` | `PROPOSED_NOT_APPROVED` |

현재 BigQuery 학습 readiness가 허용하는 variant는 Watch-only뿐이다. H10-only와
Watch+H10 row는 producer·parity·승인 완료 전까지 reader가 거부한다.

## 3. Watch 원천값과 공통 계산 규칙

Watch 16-vector를 만들기 위한 핵심 관측값은 다음과 같다.

- `corrected_utc`: UTC로 교정된 1 Hz 시간축
- `observed__eda_us`: EDA
- `observed__heart_rate_bpm`: 심박
- `observed__motion_magnitude`: ACC 3축에서 만든 움직임 크기
- `quality_confidence`: HR·EDA·ACC 현재 window의 coverage와 validity를 보수적으로
  결합한 값

공통 계약:

- HR·EDA·ACC가 모두 있어야 하며 `quality_confidence >= 0.70`이어야 eligible이다.
- 품질 누락, 비유한 생체값, clock/presence/source-quality 실패는
  `NOT_DECISIONABLE`이다.
- 보간, hold-forward, 생체값 0/null padding은 금지한다.
- 개인 기준선은 최대 1,000 wall seconds 안에서 확보한 900 eligible seconds다.
- 기준선 center는 median, scale은 `1.4826 × MAD`, 실패 시 `IQR/1.349`, 다시
  실패하면 `BASELINE_SCALE_ZERO`다.
- 최초 기준선은 session 동안 freeze하며 총 1,800 eligible seconds 전에는 모델 판단을
  생성하지 않는다.
- rolling window는 현재 시점을 제외한 `[t-window,t)` causal window다.
- 품질값은 load에 곱하지 않고 gate 및 독립 feature로만 사용한다.

## 4. Watch-only ordered 16 features

ONNX/model 입력은 아래 순서의 `float32[N,16]`이다. 순서를 바꾸면 다른 schema다.

| # | 피처 | 수식 | 최소 과거 이력 |
|---:|---|---|---|
| 1 | `watch_eda_z` | `clip((EDA_t-median(EDA_baseline))/scale(EDA_baseline),-20,20)` | 현재 eligible second |
| 2 | `watch_hr_z` | `clip((HR_t-median(HR_baseline))/scale(HR_baseline),-20,20)` | 현재 eligible second |
| 3 | `watch_motion_z` | `clip((motion_t-median(motion_baseline))/scale(motion_baseline),-20,20)` | 현재 eligible second |
| 4 | `watch_load_raw` | `mean(max(EDA_z,0),max(HR_z,0),max(motion_z,0))` | 현재 eligible second |
| 5 | `watch_eda_mean_30` | `mean(watch_eda_z[t-30:t])` | 27 eligible seconds 이상 |
| 6 | `watch_hr_mean_30` | `mean(watch_hr_z[t-30:t])` | 27 eligible seconds 이상 |
| 7 | `watch_motion_mean_30` | `mean(watch_motion_z[t-30:t])` | 27 eligible seconds 이상 |
| 8 | `watch_eda_slope_60` | `corrected_utc_ols_slope(watch_eda_z[t-60:t])` | 54 eligible seconds 이상 |
| 9 | `watch_hr_slope_60` | `corrected_utc_ols_slope(watch_hr_z[t-60:t])` | 54 eligible seconds 이상 |
| 10 | `watch_motion_slope_60` | `corrected_utc_ols_slope(watch_motion_z[t-60:t])` | 54 eligible seconds 이상 |
| 11 | `watch_load_std_60` | `population_std(watch_load_raw[t-60:t],ddof=0)` | 54 eligible seconds 이상 |
| 12 | `watch_load_median_300` | `median(watch_load_raw[t-300:t])` | 270 eligible seconds 이상 |
| 13 | `watch_load_ema_1800` | `EMA(load, alpha=1-2^(-1/1800))` | eligible-second replay |
| 14 | `watch_load_ema_21600` | `EMA(load, alpha=1-2^(-1/21600))` | eligible-second replay |
| 15 | `quality_confidence` | `min(HR_quality,EDA_quality,ACC_quality)` | 현재 second, 누락 금지 |
| 16 | `watch_ineligible_fraction_60` | `1-sum(eligible[t-60:t])/60` | 직전 60 corrected-UTC seconds |

Watch 표준 예측 target은
`observational_future_mean_1800_valid90_gap5_v2`다.

```text
target(t) = mean(watch_load_raw[t+1 : t+1800])
```

미래 1,800초 중 90% 이상이 유효하고 최대 gap이 5초 이하여야 `target_valid=true`다.
이 target은 연속형 관측 표준 예측값이며, 행동명이나 5단계 stage truth가 아니다.

## 5. H10-only ordered 13 features

H10 계약은 수식과 model-side golden parity까지 고정됐지만 Cloud producer와 실제 cohort가
없어 아직 학습 대상이 아니다.

| # | 피처 | 수식 |
|---:|---|---|
| 1 | `h10_hr_z` | `clip((HR_t-median(HR_baseline))/scale(HR_baseline),-20,20)` |
| 2 | `h10_rmssd_inverse_z` | `clip(-(RMSSD_t-median(RMSSD_baseline))/scale(RMSSD_baseline),-20,20)` |
| 3 | `h10_load_raw` | `mean(max(h10_hr_z,0),max(h10_rmssd_inverse_z,0))` |
| 4 | `h10_hr_mean_30` | `time_weighted_mean(h10_hr_z,[t-30,t))` |
| 5 | `h10_rmssd_inverse_mean_30` | `time_weighted_mean(h10_rmssd_inverse_z,[t-30,t))` |
| 6 | `h10_hr_slope_60` | `corrected_utc_ols_slope(h10_hr_z,[t-60,t))` |
| 7 | `h10_rmssd_inverse_slope_60` | `corrected_utc_ols_slope(h10_rmssd_inverse_z,[t-60,t))` |
| 8 | `h10_load_std_60` | `time_weighted_population_std(h10_load_raw,[t-60,t),ddof=0)` |
| 9 | `h10_load_median_300` | `time_weighted_median(h10_load_raw,[t-300,t))` |
| 10 | `h10_load_ema_1800` | `bounded_ema_eligible_second_replay(load,half_life=1800)` |
| 11 | `h10_load_ema_21600` | `bounded_ema_eligible_second_replay(load,half_life=21600)` |
| 12 | `h10_quality_confidence` | `hard_gate*min(ecg_coverage,contact_fraction,rr_fraction)` |
| 13 | `h10_ineligible_fraction_60` | `1-eligible_seconds([t-60,t))/60` |

## 6. BigQuery model view 26-field 계약

### 반복 envelope 10필드

1. `training_cohort_uuid`
2. `cohort_digest`
3. `split_digest`
4. `public_cohort_digest`
5. `public_split_digest`
6. `feature_schema_uuid`
7. `feature_schema_hash`
8. `split_policy`
9. `purge_seconds`
10. `truth_state`

### row 16필드

1. `training_subject_uuid`
2. `training_capture_set_uuid`
3. `exact_window_id`
4. `source_set`
5. `window_start_ms`
6. `window_end_ms`
7. `feature_values`
8. `label_uuid`
9. `label_revision_uuid`
10. `review_uuid`
11. `review_disposition`
12. `evaluation_class`
13. `temporal_stage`
14. `observation_code`
15. `split_role`
16. `source_row_digest`

Reader는 필드 누락·추가, mixed/null digest, `LOCKED`, 중복 window/digest, 비유한 feature,
잘못된 review/evaluation 조합을 거부한다. sequence grouping key는
`training_capture_set_uuid`다. `account_uuid`, `person_uuid`, profile,
`capture_session_uuid`, raw payload는 모델 view에 노출하지 않는다.

허용 truth 조합:

```text
TARGET_EVENT    -> POSITIVE
VALID_NON_EVENT -> NEGATIVE
HARD_NEGATIVE   -> NEGATIVE
```

NEGATIVE의 `observation_code`는 `NO_EVENT`여야 한다. `temporal_stage` 1~5는 시간 단계
감사용이며 severity truth가 아니다.

## 7. UUID와 날짜 조회 방식

### 현재 모델 reader가 지원하는 방식

```text
input: training_cohort_uuid 1개
query: WHERE training_cohort_uuid=@training_cohort_uuid
output: TRAIN + VALIDATION frozen rows
```

```bash
multisensor-ml model-platform read-bigquery-cohort \
  --training-cohort-uuid <uuid> \
  --output-receipt <create-only-json>
```

### 지원하지 않는 방식

```text
person_uuid + 시작일 + 종료일로 Kaggle/모델이 live raw row를 직접 선택
```

이 방식은 대상자 누출, 날짜별 사후 선택, 분할 변경, label revision 혼합을 일으킬 수 있어
금지한다.

### 올바른 날짜 기반 흐름

```text
Cloud private plane:
person_uuid + from_utc + to_utc + source_set + reviewed-only
  -> exact windows와 label/review revision 고정
  -> split 정책과 1,800초 purge 사전 적용
  -> immutable training_cohort_uuid + digests 생성

Model/Kaggle public plane:
training_cohort_uuid
  -> 26필드 TRAIN/VALIDATION view read
  -> digest/schema/split/truth 검증
  -> 검증 성공 후에만 fit
```

따라서 “UUID와 날짜로 가져온다”는 기능은 Cloud의 cohort materializer에는 필요하지만,
모델 reader에는 날짜 인자를 추가하지 않는 것이 현재 설계와 맞다.

## 8. 현재 학습 가능성 판정

| 확인 항목 | 현재 상태 | 판정 |
|---|---|---|
| model-reader principal/view 경계 | read-only 경계 검증됨 | 통과 |
| live authorized view row 수 | 0 | 차단 |
| live view schema | 24필드 | 26필드 reader와 불일치 |
| 26필드 migration | 코드 push, live 미적용 | 차단 |
| 실제 `training_cohort_uuid` | model view에 없음 | 차단 |
| `REVIEWED_REAL` label/review | row 0이므로 없음 | 차단 |
| TRAIN/VALIDATION 분할 | 실제 cohort 없음 | 차단 |
| Watch schema | 코드상 canonical | 입력 row가 없어 미검증 |
| H10/Watch+H10 | producer/승인 미완료 | 차단 |
| fit/train 실행 | 0회 | 정상적인 fail-closed |

현재는 **UUID 계층의 코드 계약은 존재하지만 model-visible training hierarchy는 아직
materialize되지 않았다.** 즉 account/person/source/capture UUID가 일부 운영 데이터에
존재하는 것과, 학습 가능한 `training_cohort_uuid -> training_subject_uuid ->
training_capture_set_uuid -> exact_window_id` row가 존재하는 것은 별개다.

## 9. 실제 학습 시작 게이트

다음 조건을 모두 충족한 후에만 학습을 시작한다.

1. 26필드 migration을 live 적용하고 destination schema readback이 정확히 일치한다.
2. Watch derived feature producer가 canonical UUID/hash로 유한한 16-vector를 생성한다.
3. 독립 검토가 끝난 positive/negative label revision을 만든다.
4. Cloud가 날짜 범위와 대상자를 private plane에서 선택해 frozen cohort를 생성한다.
5. authorized view에 해당 cohort의 TRAIN과 VALIDATION row가 모두 나타난다.
6. 각 split의 POSITIVE/NEGATIVE가 최소 1개 이상이라는 기술적 gate를 통과한다.
   이는 통계적으로 충분한 학습량을 뜻하지 않는다.
7. `purge_seconds >= 1800`, public/private digest 단일성, 중복 0, LOCKED 접근 0을
   reader가 검증한다.
8. readiness receipt가 `training_ready=true`가 된 뒤에만 fit을 별도 승인한다.

독립 실제 라벨과 충분한 사람·날짜 범위가 확보되기 전에는 accuracy, AUCPR, FPR/FNR 또는
임상 성능을 계산하거나 주장하지 않는다.
