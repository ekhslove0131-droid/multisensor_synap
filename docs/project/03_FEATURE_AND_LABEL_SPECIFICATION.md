# 피처와 라벨 명세서

## 1. 목적

이 문서는 합성 신호에서 어떤 피처를 계산하고, 특정 시간 패턴에 어떤 라벨을 붙이는지 정의한다.

핵심 규칙:

```text
숨겨진 사건·상태 생성
→ 장비별 원시 관측 신호 생성
→ 품질과 결측 적용
→ 과거 데이터만 사용해 파생변수 계산
→ 숨겨진 사건 타임라인에서 라벨 생성
```

라벨을 피처 임계값으로 만들지 않는다. 파생변수를 직접 조작하지 않는다.

---

# 2. 시간 해상도

## 2.1 Native streams

| 장비 | 스트림 | 기본 native rate |
|---|---|---:|
| Muse S | EEG TP9/AF7/AF8/TP10 | 256 Hz |
| Muse S | head ACC XYZ | 52 Hz |
| Muse S | PPG | 64 Hz, 기본 비활성 |
| Polar H10 | ECG | 130 Hz |
| Polar H10 | chest ACC XYZ | 100 Hz, 설정 가능 |
| Polar H10 | device HR/RR | 1 Hz |
| Galaxy Watch8 | wrist ACC XYZ | 25 Hz |
| Galaxy Watch8 | EDA | 1 Hz |
| Galaxy Watch8 | PPG G/R/IR | 25 Hz |
| Galaxy Watch8 | HR/IBI | 1 Hz |
| Galaxy Watch8 | skin/ambient temperature | event-driven, MVP 1 Hz 관측 |

## 2.2 Canonical timeline

- 기준: `1 Hz`
- 키: `person_id`, `canonical_time_ns`
- UTC 저장
- 표시용 `timestamp_utc` 추가
- 장비별 1초 커버리지와 품질을 별도 열로 유지

## 2.3 윈도

| 윈도 | 주요 목적 |
|---:|---|
| 5초 | 동작·아티팩트·급격한 변화 |
| 15초 | 짧은 onset과 signal-quality 변화 |
| 30초 | 단기 자율·EEG 변화 |
| 60초 | 사건 전후 경향과 HRV 시간영역 |
| 180초 | 전조 패턴과 멀티모달 지연 |
| 300초 | 안정적 baseline·회복·HRV 일부 |
| 600초 선택 | 긴 회복·저주파 문맥 |

각 피처에는 `window_sec`, `minimum_coverage`, `causal=true`를 메타데이터로 기록한다.

---

# 3. 공통 컬럼

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `run_id` | string | 생성 실행 ID |
| `person_id` | string | 가상 참여자 ID |
| `canonical_time_ns` | int64 | 정렬된 UTC nanosecond |
| `date_utc` | date | Parquet partition |
| `day_index` | int8 | 0~4 |
| `seconds_from_start` | int64 | 실행 시작 후 경과시간 |
| `context_state` | category | sleep/rest/activity 등 관측 가능한 문맥 |
| `is_awake` | bool | 수면 문맥 |
| `feature_window_sec` | int16 | long-form 사용 시 윈도 |

`truth_*` 접두어 열은 model-ready feature 테이블에 허용하지 않는다.

---

# 4. EEG 피처

## 4.1 전처리·품질

각 native clip에 다음 순서를 적용한다.

1. 단위 확인 및 µV 변환
2. 채널 이름·순서 확인
3. notch 50/60 Hz는 설정 기반
4. 분석 목적별 bandpass
5. flatline, clipping, extreme amplitude 검사
6. 머리 ACC와의 동작 결합 검사
7. 채널별 유효 구간 표시

MVP에서는 ICA를 품질 정답 생성의 필수 단계로 두지 않는다. 4채널에서 ICA의 안정성이 제한되므로 아티팩트 정답과 간단한 robust quality feature를 우선한다.

## 4.2 채널별 피처

각 채널 `ch ∈ {tp9, af7, af8, tp10}`에 대해 생성한다.

### 스펙트럼

- `eeg_{ch}_delta_abs_power`
- `eeg_{ch}_theta_abs_power`
- `eeg_{ch}_alpha_abs_power`
- `eeg_{ch}_beta_abs_power`
- `eeg_{ch}_gamma_abs_power`
- `eeg_{ch}_{band}_relative_power`
- `eeg_{ch}_theta_beta_ratio`
- `eeg_{ch}_beta_alpha_ratio`
- `eeg_{ch}_spectral_entropy`
- `eeg_{ch}_spectral_edge_95hz`
- `eeg_{ch}_median_frequency_hz`
- `eeg_{ch}_aperiodic_exponent` 선택
- `eeg_{ch}_aperiodic_offset` 선택

주파수 band 경계는 설정 파일에서 고정한다. gamma는 Muse 위치와 근전도 혼입 가능성이 크므로 `gamma_power`와 `high_freq_artifact_ratio`를 함께 제공한다.

### 시간영역·복잡도

- `eeg_{ch}_rms_uv`
- `eeg_{ch}_peak_to_peak_uv`
- `eeg_{ch}_line_length`
- `eeg_{ch}_zero_crossing_rate`
- `eeg_{ch}_hjorth_activity`
- `eeg_{ch}_hjorth_mobility`
- `eeg_{ch}_hjorth_complexity`
- `eeg_{ch}_sample_entropy`
- `eeg_{ch}_permutation_entropy`

복잡도 피처는 최소 샘플 수를 충족하지 못하면 NaN과 `feature_valid=false`를 반환한다.

## 4.3 채널 간 피처

- `eeg_frontal_alpha_asymmetry_af7_af8`
- `eeg_temporal_alpha_asymmetry_tp9_tp10`
- `eeg_pair_{a}_{b}_coherence_{band}`
- `eeg_pair_{a}_{b}_plv_{band}` 선택
- `eeg_global_field_power`
- `eeg_channel_correlation_mean`

4채널 소비자 EEG이므로 연결성 피처는 탐색용이며 높은 임상 의미를 부여하지 않는다.

## 4.4 EEG 품질 피처

- `eeg_{ch}_coverage_ratio`
- `eeg_{ch}_flatline_ratio`
- `eeg_{ch}_clipping_ratio`
- `eeg_{ch}_extreme_amplitude_ratio`
- `eeg_{ch}_line_noise_ratio`
- `eeg_{ch}_high_freq_artifact_ratio`
- `eeg_{ch}_motion_correlation`
- `eeg_valid_channel_count`
- `eeg_quality_score`
- `eeg_artifact_dominant`

---

# 5. 심장·HRV 피처

Polar H10 ECG를 1차 원천으로 한다.

## 5.1 R-peak와 RR 품질

- `ecg_rpeak_count`
- `ecg_rpeak_confidence_mean`
- `ecg_rr_valid_ratio`
- `ecg_rr_missed_peak_suspect_count`
- `ecg_rr_extra_peak_suspect_count`
- `ecg_baseline_wander_score`
- `ecg_motion_artifact_score`
- `ecg_quality_score`

장비 RR과 ECG 재계산 RR은 분리한다.

- `polar_rr_device_median_ms`
- `polar_rr_ecg_median_ms`
- `polar_rr_source_disagreement_ms`

## 5.2 심박·시간영역 HRV

- `hr_mean_bpm`
- `hr_median_bpm`
- `hr_min_bpm`
- `hr_max_bpm`
- `hr_std_bpm`
- `hr_slope_bpm_per_min`
- `hr_acceleration_bpm_per_sec`
- `rr_mean_ms`
- `rr_median_ms`
- `rr_mad_ms`
- `sdnn_ms`
- `rmssd_ms`
- `sdsd_ms`
- `pnn20`
- `pnn50`
- `cvnn`
- `poincare_sd1_ms`
- `poincare_sd2_ms`
- `poincare_sd1_sd2_ratio`
- `rr_sample_entropy`

### 최소 윈도 정책

| 피처군 | 최소 권장 윈도 | 미달 시 |
|---|---:|---|
| HR 평균·slope | 15~30초 | coverage 기준 |
| RMSSD·SDNN 단기 | 60초 | `feature_valid=false` 가능 |
| Poincaré·entropy | 180초 | NaN + invalid |
| frequency-domain HRV | 기본 300초 | 더 짧으면 계산 금지 |

## 5.3 주파수영역 HRV

- `hrv_vlf_power` 선택
- `hrv_lf_power`
- `hrv_hf_power`
- `hrv_lf_hf_ratio`
- `hrv_total_power`

주의:

- 호흡을 직접 측정하지 않으므로 `HF=부교감`처럼 단정하지 않는다.
- 300초 미만 윈도에서 기본 계산하지 않는다.
- 활동과 비정상 RR 구간은 `feature_valid=false` 처리한다.

---

# 6. EDA 피처

Watch8의 1 Hz 관측률에 맞춰 고주파 EDA처럼 과도한 형태를 만들지 않는다.

- `eda_mean_us`
- `eda_median_us`
- `eda_min_us`
- `eda_max_us`
- `eda_slope_us_per_min`
- `eda_tonic_level_us`
- `eda_phasic_level_us`
- `eda_scr_count`
- `eda_scr_amplitude_mean_us`
- `eda_scr_amplitude_max_us`
- `eda_scr_area_us_sec`
- `eda_time_since_last_scr_sec`
- `eda_variability`
- `eda_flatline_ratio`
- `eda_spike_ratio`
- `eda_coverage_ratio`
- `eda_quality_score`

SCR 관련 피처는 1 Hz 해상도에서 근사값임을 메타데이터에 기록한다.

---

# 7. 움직임 피처

세 위치를 분리한다.

- `wrist_*`: Galaxy Watch8
- `head_*`: Muse S
- `chest_*`: Polar H10

각 위치의 기본 피처:

- `{loc}_acc_vm_mean_g`
- `{loc}_acc_vm_std_g`
- `{loc}_enmo_mean_g`
- `{loc}_sma_g`
- `{loc}_jerk_mean_g_per_sec`
- `{loc}_jerk_p95_g_per_sec`
- `{loc}_dominant_frequency_hz`
- `{loc}_spectral_entropy`
- `{loc}_autocorr_peak`
- `{loc}_periodicity_score`
- `{loc}_stillness_ratio`
- `{loc}_activity_burst_count`
- `{loc}_orientation_change_rate`
- `{loc}_coverage_ratio`
- `{loc}_quality_score`

반복 움직임 후보:

- `wrist_repetition_score`
- `head_repetition_score`
- `cross_location_repetition_coherence`
- `repetition_duration_sec`
- `repetition_frequency_stability`

`repetition_score`만으로 행동 의미를 확정하지 않는다.

---

# 8. PPG·온도 피처

## 8.1 Watch PPG

- `watch_ppg_green_dc`
- `watch_ppg_green_ac_amplitude`
- `watch_ppg_ir_ac_amplitude`
- `watch_ppg_red_ac_amplitude`
- `watch_ppg_pulse_rate_bpm`
- `watch_ppg_pulse_amplitude_variability`
- `watch_ppg_motion_correlation`
- `watch_ppg_saturation_ratio`
- `watch_ppg_quality_score`

## 8.2 온도

- `watch_skin_temp_c`
- `watch_ambient_temp_c`
- `watch_skin_ambient_gap_c`
- `watch_skin_temp_slope_c_per_min`
- `watch_temp_coverage_ratio`

피부 온도를 체온으로 명명하지 않는다.

---

# 9. 개인 기준선 피처

## 9.1 Truth baseline

`truth/person_profiles.parquet`에는 시뮬레이션에 사용한 실제 개인 파라미터가 들어간다. 모델 입력에는 사용하지 않는다.

## 9.2 Online baseline

모델 입력 기준선은 과거 데이터로만 추정한다.

권장 방식:

1. 최소 warm-up 30분 또는 설정값
2. 수면·운동·저품질 구간 제외
3. rolling 또는 exponentially weighted median/MAD
4. 현재 시점 이후 값 미사용

피처 예:

- `hr_personal_robust_z`
- `rmssd_personal_robust_z`
- `eda_personal_robust_z`
- `eeg_alpha_personal_robust_z`
- `eeg_beta_alpha_personal_robust_z`
- `wrist_motion_personal_robust_z`
- `baseline_age_sec`
- `baseline_sample_count`
- `baseline_confidence`

Robust z-score:

```text
z = (x - rolling_median_past) / (1.4826 × rolling_MAD_past + epsilon)
```

## 9.3 고정 baseline 실험

합성 truth baseline을 이용한 `oracle` 실험은 별도 테이블로만 허용한다.

- `oracle_features/`에 저장
- 일반 모델 성능과 혼합 금지
- 개인 baseline 추정 알고리즘의 상한 비교용

---

# 10. Cross-modal 피처

- `watch_hr_vs_polar_hr_abs_diff_bpm`
- `watch_ibi_vs_polar_rr_abs_diff_ms`
- `ppg_ecg_pulse_transit_proxy_ms`
- `eeg_motion_lag_corr_max`
- `eda_hr_lag_corr_max`
- `eda_motion_lag_corr_max`
- `hr_motion_conditioned_residual`
- `eda_motion_conditioned_residual`
- `multisensor_valid_axis_count`
- `multisensor_quality_min`
- `multisensor_quality_mean`
- `multisensor_deviation_score`
- `modality_onset_order_code`
- `cross_modal_synchrony_score`
- `cross_device_disagreement_score`

`multisensor_deviation_score`는 한 임계값으로 라벨을 복제하지 않도록 학습 입력의 한 피처로만 둔다.

---

# 11. 라벨 명세

## 11.1 식별 라벨

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `event_id` | nullable string | 동일 사건 연결 |
| `event_type` | category | 중립 사건 유형 |
| `event_phase` | category | 단계 |
| `event_binary` | int8 | target 사건 구간 0/1 |
| `event_instance_index` | int32 | 개인 내 사건 순번 |
| `label_source` | string | 항상 `synthetic_rule_v*` |
| `label_confidence` | float32 | 합성 정답 자체의 확실성 |

`label_confidence=1.0`은 실제 세계에서 사건 의미가 확실하다는 뜻이 아니라, 생성 엔진이 주입한 구간을 정확히 안다는 뜻이다.

## 11.2 단계 라벨

허용값:

```text
baseline
pre_early
pre_late
onset
peak
recovery_early
recovery_late
post
```

별도 정수 매핑:

```yaml
baseline: 0
pre_early: 1
pre_late: 2
onset: 3
peak: 4
recovery_early: 5
recovery_late: 6
post: 7
```

## 11.3 시간 라벨

- `time_to_onset_sec`
- `time_since_onset_sec`
- `time_to_peak_sec`
- `time_since_peak_sec`
- `time_to_event_end_sec`

해당 사건과 관련 없는 시점은 null이다.

## 11.4 미래 예측 라벨

각 시점 `t`에 대해 미래에 target onset이 있는지 표시한다.

- `will_target_start_within_30s`
- `will_target_start_within_60s`
- `will_target_start_within_120s`
- `will_target_start_within_180s`
- `next_target_onset_sec`

주의:

- 피처는 `t`까지의 데이터만 사용한다.
- 미래 라벨은 타깃으로만 사용한다.
- recovery 중 곧 다른 사건이 발생하는 겹침 사례를 설정 가능하게 한다.

## 11.5 강도 라벨

- `event_intensity_class`: `low|medium|high`
- `event_intensity_truth`: 0~1

연속 truth 강도는 기본적으로 회귀 타깃 또는 검증용이며 일반 입력에서 제외한다.

## 11.6 품질·아티팩트 라벨

- `artifact_present`
- `artifact_type_primary`
- `artifact_dominant`
- `valid_for_eeg_model`
- `valid_for_cardiac_model`
- `valid_for_eda_model`
- `valid_for_multimodal_model`
- `missing_modality_count`

아티팩트 구간을 모두 삭제하지 않는다. 품질 감지와 실제 환경 견고성 평가를 위해 일부를 학습에 남긴다.

---

# 12. 사건 겹침 규칙

기본 MVP에서는 target event끼리 peak가 겹치지 않게 한다. 문맥·활동·아티팩트는 target event와 겹칠 수 있다.

우선순위:

```text
sensor_artifact는 별도 축
ordinary_activity는 context 축
primary target event는 event 축
sleep/wake는 context 축
```

따라서 한 행에 다음이 동시에 가능하다.

```text
context_state = light_activity
event_type = multimodal_arousal_episode
event_phase = pre_late
artifact_present = true
```

하나의 `state` 열로 모든 것을 압축하지 않는다.

---

# 13. Hard negative 설계

모델이 단순 규칙을 외우지 않도록 다음을 포함한다.

## 13.1 Ordinary activity

- HR 증가
- HRV 감소
- EDA 증가 가능
- 움직임 크게 증가
- EEG 아티팩트 증가
- target label은 0

## 13.2 Quiet cognitive load

- 움직임 적음
- EEG와 일부 자율 피처 변화
- target label은 0 또는 별도 사건

## 13.3 Sensor artifact only

- 신호는 크게 흔들리지만 숨겨진 생리 상태는 안정
- artifact label 1
- target label 0

## 13.4 Partial precursor

- EDA만 증가 또는 EEG만 변화
- onset 없이 baseline으로 복귀
- `recovery_without_peak`

## 13.5 Spontaneous normal extremes

- 정상 구간에도 개인 분포의 극단값 발생
- 특정 z-score 임계값이 라벨을 완벽히 분리하지 못하게 함

---

# 14. 라벨 누출 방지

금지:

```python
event_label = (multisensor_deviation_score > 0.8)
```

허용:

```text
event engine이 E001의 onset을 10:32:15로 정함
→ 생리 잠재요인 변화
→ 신호와 파생변수 변화
→ E001의 truth timeline에서 라벨 생성
```

추가 검사:

1. label 생성 모듈이 feature table을 import하지 않는지 확인
2. `truth_*`, `event_intensity_truth`가 feature schema에 없는지 확인
3. 미래 윈도 또는 centered rolling 사용 금지
4. 사람 ID에서 라벨 빈도가 직접 추론되지 않도록 사건 수 분산
5. 날짜·시간이 target을 완벽히 설명하지 않도록 발생 시간 다양화
6. split은 row가 아니라 사람 단위

---

# 15. Split 정책

MVP 12명 예시:

- train: 8명
- validation: 2명
- test: 2명

split은 seed로 결정하고 manifest에 고정한다.

추가 평가:

- `unseen_person`: 사람 완전 분리
- `known_person_future_day`: 같은 사람의 과거 3일 학습, 이후 2일 평가
- `cross_day`: 날짜 drift 견고성
- `artifact_stress`: 고아티팩트 test subset

합성기 프로젝트는 모델 성능을 최종 산출물로 요구하지 않지만, 데이터가 지나치게 쉬운지 검사하기 위한 간단한 sanity model을 허용한다.

---

# 16. Feature metadata 계약

모든 파생변수는 registry에 등록한다.

```yaml
name: rmssd_ms
modality: cardiac
source_streams: [polar_h10.ecg]
window_sec: 60
minimum_coverage: 0.9
causal: true
unit: ms
valid_range: [0, 500]
requires:
  min_valid_rr_count: 30
leakage_risk: low
notes: short-window estimate
```

registry에서 코드·문서·스키마를 생성할 수 있게 한다.

---

# 17. 승인 기준

- 모든 feature가 registry에 존재한다.
- raw clip에서 feature를 재계산했을 때 저장값과 허용 오차 내 일치한다.
- `feature_valid=false` 조건이 테스트된다.
- label phase의 순서·경계가 event manifest와 일치한다.
- forecast label이 정확한 미래 범위를 반영한다.
- target과 hard negative의 단일 피처 분리가 완벽하지 않다.
- 품질 피처 없이도 아티팩트가 target으로 오인될 수 있는 현실적 사례가 존재한다.
- 사람 단위 split 누출이 없다.
