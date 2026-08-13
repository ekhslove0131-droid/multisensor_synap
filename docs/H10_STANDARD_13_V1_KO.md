# Polar H10 단독 13-피처 계약 v1

## 상태와 식별자

- 입력 원본: `kidsignal_polar_h10_ecg_v2`
- 장치 프로필: `polar_h10_v1`
- 피처 스키마: `h10_standard_13_v1`
- 스키마 UUID: `8ba3dbff-e5c4-5dd3-be1d-69ea6cea8966`
- canonical JSON SHA-256: `5ff8e9ceaa534bb8b7fafbdd7f844b06c0a4b21bb206428d5ce9239eec5459d6`
- producer 상태: `PROPOSED_NOT_IMPLEMENTED`
- 실제 데이터 성능: `NOT VERIFIED`
- Watch+H10 융합: `PROPOSED_NOT_APPROVED`

이 계약은 모델 쪽에서 실제 Polar observation v2 필드와 수식을 고정한 제안이다. 현재
Cloud Inference에 H10 v2 decoder/feature producer가 있다는 뜻이 아니며, Kaggle 학습 가능
상태나 실제 성능을 의미하지 않는다.

canonical JSON은
`multisensor_ml.platform_contract.h10_runtime_feature_schema()`가 반환하는
`canonical_json`이다. UTF-8, key 정렬, 공백 없는 구분자, NaN 금지로 직렬화하며 위
SHA-256은 그 바이트에 대해 계산한다.

## 원시 window와 시간축

H10 v1 모델 입력 단위는 완결된 10초 source window다. ECG의
`ecg_samples.corrected_utc_ms`가 H10 내부 corrected-time 축이며, window 출력 시각은
마지막 ECG sample의 corrected UTC다.

`hr_observations.phone_utc_ms`는 HR/RR callback이 해당 source window에 들어왔다는
근거로만 쓴다. 이것을 ECG beat의 corrected UTC로 간주하거나 보간·재정렬하지 않는다.
RR별 corrected timestamp가 없으므로 RMSSD는 같은 source window 내부에서 callback
순서대로 펼친 `rr_ms`에 대해서만 계산하고 window 경계를 넘기지 않는다.

`voltage_uv`는 v1에서 finite 여부·개수·corrected-time 순서 등 원시 무결성 확인에만
쓴다. R-peak, ECG 진단, 행동 라벨 또는 직접 모델 피처를 여기서 만들지 않는다.

## 원시 품질과 eligibility

한 10초 window가 eligible이려면 아래를 모두 만족해야 한다.

1. payload/schema/device가 각각 `kidsignal_polar_h10_ecg_v2`,
   `real_observed_h10`, `polar_h10_v1`이다.
2. sample rate는 130 Hz, 기대 sample은 1,300개이며 선언된 관측 개수와 실제 ECG
   배열 길이가 같다.
3. ECG coverage가 90% 이상이고 `is_sufficient=true`, `gap_count=0`,
   `timestamp_regression_count=0`이다.
4. 모든 `corrected_utc_ms`가 유한하고 엄격히 증가하며 모든 `voltage_uv`가 유한하다.
5. `hr_stream_state=observed`이고, HR callback 중
   `contact_status_supported=true AND contact_status=true AND corrected_hr_bpm이 양의 유한값`
   인 비율이 90% 이상이다.
6. `rr_available=true`이며 양의 유한 `rr_ms`가 있는 callback 비율이 90% 이상이고,
   contact-valid RR interval이 최소 2개다.

품질 신뢰도는 hard integrity gate가 통과했을 때만

`min(ECG coverage, contact-valid fraction, RR-valid fraction)`

으로 계산한다. hard gate 실패나 품질 필드 누락은 0이며 `NOT_DECISIONABLE`이다. 생체값
0/null 대체, 보간, hold-forward, 이전 window 재사용, `load × quality`는 금지한다.

## 현재 window 요약과 기준선

- `HR_t = median(contact-valid corrected_hr_bpm)`
- `RMSSD_t = sqrt(mean(diff(flatten(rr_ms in callback order))²))`
- 기준선은 과거 eligible 900초를 최대 wall time 1,000초 안에서 확보한 최초 구간이다.
- center는 median, scale은 `1.4826 × MAD`, MAD가 0이면 `IQR/1.349`, 이것도 0이면
  `BASELINE_SCALE_ZERO`로 판단을 중단한다.
- 최초 기준선은 session 동안 freeze한다.
- 총 eligible 1,800초 전에는 13개 피처가 모두 있어도 모델 판단을 내지 않는다.

## 정렬된 13개 피처

| 순서 | 피처 | 수식 |
|---:|---|---|
| 1 | `h10_hr_z` | `clip((HR_t-median(HR_baseline_900))/scale(HR_baseline_900),-20,20)` |
| 2 | `h10_rmssd_inverse_z` | `clip(-(RMSSD_t-median(RMSSD_baseline_900))/scale(RMSSD_baseline_900),-20,20)` |
| 3 | `h10_load_raw` | `mean(max(h10_hr_z,0),max(h10_rmssd_inverse_z,0))` |
| 4 | `h10_hr_mean_30` | `time_weighted_mean(h10_hr_z,[t-30,t))` |
| 5 | `h10_rmssd_inverse_mean_30` | `time_weighted_mean(h10_rmssd_inverse_z,[t-30,t))` |
| 6 | `h10_hr_slope_60` | `corrected_utc_ols_slope(h10_hr_z,[t-60,t))` |
| 7 | `h10_rmssd_inverse_slope_60` | `corrected_utc_ols_slope(h10_rmssd_inverse_z,[t-60,t))` |
| 8 | `h10_load_std_60` | `time_weighted_population_std(h10_load_raw,[t-60,t),ddof=0)` |
| 9 | `h10_load_median_300` | `time_weighted_median(h10_load_raw,[t-300,t))` |
| 10 | `h10_load_ema_1800` | `bounded_ema_eligible_second_replay(h10_load_raw,half_life_seconds=1800)` |
| 11 | `h10_load_ema_21600` | `bounded_ema_eligible_second_replay(h10_load_raw,half_life_seconds=21600)` |
| 12 | `h10_quality_confidence` | `hard_integrity_gate*min(ecg_coverage,contact_valid_fraction,rr_valid_fraction)` |
| 13 | `h10_ineligible_fraction_60` | `1-eligible_seconds([t-60,t))/60` |

모든 rolling window는 현재 값을 제외한 `[t-window,t)`이다. 30/60/300초 window는 각각
eligible coverage 90% 이상이어야 한다. EMA의 1초 alpha는
`1-exp(-ln(2)/half_life_seconds)`이고, eligible 10초 window 요약값을 1초씩 10회 replay한다.
결측 동안 EMA state는 freeze하며 decay나 값 대체를 하지 않는다.

## 다음 구현 게이트

1. Cloud H10 v2 decoder가 위 원시 품질 provenance를 보존한다.
2. 같은 raw fixture에서 Cloud와 모델의 window summary·13-vector·schema hash가 일치한다.
3. 실제 reviewed frozen H10 cohort가 BigQuery authorized view에 생성된다.
4. 그 전까지 H10-only는 학습·승격·배포 불가이며 Watch+H10 융합도 승인하지 않는다.

## 독립 golden parity 결과

모델 저장소는 Cloud producer 코드를 복사하지 않고 이 문서의 수식으로 summary와
stateful 13-vector 계산기를 구현했다.

- summary fixture: payload 180,535 bytes, SHA-256
  `f75039ec05b80a72e60b326a5f9ab50a1c9b75c768f982b774b4522a08cd3877`
- summary: emission `19992`, HR median `72.0`, RMSSD `12.5`, quality `1.0`,
  eligible 10초, reason 없음
- vector fixture: 180개 eligible 10초 summary, 900초 baseline freeze, 1,800초 decision
- 13-vector Cloud 비교 최대 절대 오차: `6.661338147750939e-16`
- status/reason: `DECISIONABLE / DECISIONABLE`

이는 deterministic local parity 증거이며 실제 학습이나 성능 검증이 아니다. BigQuery 실제
cohort가 0행이므로 training은 계속 `BLOCKED_NO_REAL_COHORT`다.
