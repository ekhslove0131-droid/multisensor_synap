# 설정 스키마 초안

## 1. 원칙

- YAML은 사람이 수정한다.
- Pydantic 모델과 JSON Schema로 검증한다.
- 알 수 없는 키는 기본적으로 오류로 처리한다.
- 단위와 샘플링률은 명시한다.
- 모든 확률은 0~1 범위를 검증한다.
- 실행 시 설정 전체를 `config_snapshot/`에 보존한다.
- 설정 버전은 `schema_version`으로 관리한다.

---

# 2. 권장 파일 분리

```text
configs/
├── quick.yaml
├── mvp.yaml
├── devices.yaml
├── population.yaml
├── events.yaml
├── artifacts.yaml
├── features.yaml
├── labels.yaml
└── schemas/
    ├── run.schema.json
    ├── devices.schema.json
    ├── events.schema.json
    └── features.schema.json
```

`quick.yaml`과 `mvp.yaml`은 다른 파일을 include하거나 병합한 최종 실행 설정이다. 초기 구현에서는 복잡한 YAML include 대신 Python loader에서 명시적 `extends`만 지원한다.

---

# 3. `mvp.yaml` 통합 예시

```yaml
schema_version: "1.0"
project: multisensor_synth
profile: mvp

run:
  seed: 20260725
  start_time_utc: "2026-01-01T00:00:00Z"
  duration_days: 5
  participant_count: 12
  canonical_rate_hz: 1
  timezone_context: "Asia/Seoul"
  chunk_duration_sec: 3600
  deterministic: true
  fail_on_warning: false

storage:
  output_root: "output"
  parquet_compression: zstd
  partition_by: [person_id, date_utc]
  write_truth: true
  write_native_clips: true
  write_full_native_streams: false
  write_mne_fif_for_eeg: true
  estimated_size_guard_gb: 20

devices:
  muse_s:
    enabled: true
    hardware_profile: muse_s_gen2_default
    capability_check_required: true
    eeg:
      enabled: true
      sample_rate_hz: 256
      channels: [TP9, AF7, AF8, TP10]
      unit: uV
      adc_bits: 12
      reference: FPz_CMS_DRL
      mains_frequency_hz: 60
    accelerometer:
      enabled: true
      sample_rate_hz: 52
      range_g: 4
      axes: [x, y, z]
    ppg:
      enabled: false
      sample_rate_hz: 64
      channels: [ir_1, ir_2, red]
    aux:
      enabled: false

  polar_h10:
    enabled: true
    ecg:
      enabled: true
      sample_rate_hz: 130
      unit: uV
      leads: [chest_single_lead]
    hr_rr:
      enabled: true
      sample_rate_hz: 1
      rr_unit: ms
    accelerometer:
      enabled: true
      sample_rate_hz: 100
      allowed_sample_rates_hz: [25, 50, 100, 200]
      range_g: 8
      allowed_ranges_g: [2, 4, 8]
      axes: [x, y, z]

  galaxy_watch8:
    enabled: true
    sdk_profile: samsung_health_sensor_sdk
    capability_check_required: true
    accelerometer:
      enabled: true
      mode: continuous
      sample_rate_hz: 25
      axes: [x, y, z]
    eda:
      enabled: true
      mode: continuous
      sample_rate_hz: 1
      unit: uS
    heart_rate_ibi:
      enabled: true
      mode: continuous
      sample_rate_hz: 1
    ppg:
      enabled: true
      mode: continuous
      sample_rate_hz: 25
      channels: [green, ir, red]
    skin_temperature:
      enabled: true
      mode: continuous_event
      synthetic_observation_rate_hz: 1
      units: C
    on_demand:
      ecg_500hz:
        enabled: false
        policy: calibration_only
      ppg_100hz:
        enabled: false
        policy: calibration_only

population:
  participant_id_prefix: P
  age_years:
    distribution: truncated_normal
    mean: 10.0
    std: 2.5
    min: 6
    max: 17
  biological_sex:
    categories: [female, male]
    probabilities: [0.35, 0.65]
  baseline_profiles:
    cardiac:
      resting_hr_bpm:
        distribution: truncated_normal
        mean: 82
        std: 11
        min: 50
        max: 125
      rmssd_ms:
        distribution: lognormal
        median: 35
        geometric_std: 1.6
    eeg:
      aperiodic_exponent:
        distribution: truncated_normal
        mean: 1.6
        std: 0.25
        min: 0.8
        max: 2.5
      relative_alpha:
        distribution: beta
        alpha: 4.0
        beta: 8.0
      relative_gamma:
        distribution: beta
        alpha: 2.0
        beta: 18.0
    eda:
      tonic_level_us:
        distribution: lognormal
        median: 3.0
        geometric_std: 1.8
    recovery_capacity:
      distribution: beta
      alpha: 4.0
      beta: 2.5
  response_archetypes:
    eeg_first: 0.20
    autonomic_first: 0.25
    motor_first: 0.20
    quiet_internal: 0.15
    motor_dominant: 0.10
    partial_response: 0.08
    non_responder: 0.02
  research_prior_strength: 0.20
  allow_opposite_direction_probability: 0.15
  allow_no_response_probability: 0.10

context:
  circadian:
    enabled: true
    amplitude_variation: 0.25
  sleep:
    enabled: true
    bedtime_local_hour:
      mean: 22.0
      std: 1.0
    duration_hours:
      mean: 8.8
      std: 0.9
  ordinary_activity:
    daily_bouts:
      min: 4
      max: 12
  meals:
    enabled: true
    daily_count: 3

events:
  target_type: multimodal_arousal_episode
  target_events_per_day:
    distribution: discrete_uniform
    min: 1
    max: 4
  minimum_gap_between_target_sec: 1800
  clip_window:
    pre_sec: 300
    post_sec: 600
  phases:
    pre_early_sec: {min: 60, max: 180}
    pre_late_sec: {min: 15, max: 60}
    onset_sec: {min: 5, max: 30}
    peak_sec: {min: 10, max: 120}
    recovery_early_sec: {min: 30, max: 180}
    recovery_late_sec: {min: 60, max: 600}
    post_sec: {min: 0, max: 600}
  intensity:
    distribution: beta
    alpha: 2.2
    beta: 2.0
  modality_delays_sec:
    eeg: {min: -20, max: 20}
    cardiac: {min: -10, max: 30}
    eda: {min: 0, max: 45}
    motor: {min: -10, max: 60}
  hard_negatives:
    ratio_to_target: 1.0
    types:
      ordinary_physical_activity: 0.35
      quiet_cognitive_load: 0.20
      sensor_artifact_episode: 0.20
      recovery_without_peak: 0.15
      false_alarm_like_episode: 0.10
  matched_baseline_clips_per_day: 2

artifacts:
  global:
    enabled: true
    random_missing_probability_per_hour: 0.02
    block_dropout_events_per_day: {min: 0, max: 3}
  clocks:
    initial_offset_ms:
      muse_s: {min: -300, max: 300}
      polar_h10: {min: -250, max: 250}
      galaxy_watch8: {min: -500, max: 500}
    drift_ppm:
      muse_s: {min: -30, max: 30}
      polar_h10: {min: -20, max: 20}
      galaxy_watch8: {min: -40, max: 40}
    jitter_ms_std:
      muse_s: 4
      polar_h10: 6
      galaxy_watch8: 12
  eeg:
    blink_events_per_min: {min: 5, max: 25}
    jaw_emg_probability_per_min: 0.08
    electrode_pop_probability_per_hour: 0.20
    channel_dropout_probability_per_hour: 0.05
    line_noise_hz: 60
  ecg:
    baseline_wander_strength: {min: 0.0, max: 0.3}
    motion_artifact_probability_per_min: 0.04
    missed_peak_probability: 0.002
    extra_peak_probability: 0.001
  ppg:
    motion_coupling_strength: {min: 0.2, max: 0.9}
    contact_loss_probability_per_hour: 0.05
  eda:
    motion_spike_probability_per_min: 0.03
    flatline_probability_per_hour: 0.02
  accelerometer:
    bias_std_g: 0.01
    scale_error_std: 0.015
    clipping_probability_per_hour: 0.005

features:
  windows_sec: [5, 15, 30, 60, 180, 300]
  causal_only: true
  minimum_coverage_default: 0.8
  personal_baseline:
    enabled: true
    warmup_sec: 1800
    method: rolling_robust
    lookback_sec: 21600
    exclude_contexts: [moderate_activity]
    exclude_low_quality: true
  eeg:
    frequency_bands_hz:
      delta: [1, 4]
      theta: [4, 8]
      alpha: [8, 13]
      beta: [13, 30]
      gamma: [30, 45]
    connectivity_enabled: true
    aperiodic_enabled: true
  hrv:
    frequency_domain_min_window_sec: 300
    entropy_min_window_sec: 180
  registry_path: configs/feature_registry.yaml

labels:
  phases:
    baseline: 0
    pre_early: 1
    pre_late: 2
    onset: 3
    peak: 4
    recovery_early: 5
    recovery_late: 6
    post: 7
  forecast_horizons_sec: [30, 60, 120, 180]
  include_intensity_target: true
  include_artifact_labels: true
  label_source: synthetic_rule_v1
  overlap_policy: multi_axis

splits:
  strategy: by_person
  train_fraction: 0.67
  validation_fraction: 0.17
  test_fraction: 0.16
  seed: 20260725

validation:
  schema: true
  invariants: true
  feature_recomputation: true
  leakage_checks: true
  statistical_checks: true
  simple_model_sanity_check: true
  generate_html_report: true
```

---

# 4. 핵심 Pydantic 모델 초안

```python
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RangeFloat(StrictModel):
    min: float
    max: float

    @model_validator(mode="after")
    def validate_order(self) -> "RangeFloat":
        if self.min > self.max:
            raise ValueError("min must be <= max")
        return self


class StreamConfig(StrictModel):
    enabled: bool = True
    sample_rate_hz: float = Field(gt=0)
    unit: str | None = None


class RunConfig(StrictModel):
    seed: int
    start_time_utc: str
    duration_days: int = Field(gt=0)
    participant_count: int = Field(gt=0)
    canonical_rate_hz: int = Field(default=1, gt=0)
    chunk_duration_sec: int = Field(default=3600, gt=0)
    deterministic: bool = True
```

구현 시 각 장비와 사건 설정을 별도 모델로 분리한다. `dict[str, Any]`로 우회하지 않는다.

---

# 5. 장비 계약 검증 규칙

## Muse S

- EEG channel은 기본 `[TP9, AF7, AF8, TP10]`
- sample rate 256 Hz
- ACC 52 Hz
- PPG를 켜려면 capability 확인 표시 필요

## Polar H10

- ECG 130 Hz
- ACC rate는 `[25, 50, 100, 200]` 중 하나
- ACC range는 `[2, 4, 8]` 중 하나

## Galaxy Watch8

- continuous ACC 25 Hz
- continuous EDA 1 Hz
- continuous HR/IBI 1 Hz
- continuous PPG 25 Hz
- on-demand ECG와 100 Hz PPG는 `calibration_only` 외 설정 금지
- on-demand 두 개 동시 활성 금지
- on-demand와 continuous core 동시 생성은 기본 금지

---

# 6. Event pattern 스키마 예시

```yaml
name: multimodal_arousal_episode
version: 1
is_target: true

phase_curve:
  pre_early: smoothstep
  pre_late: exponential_rise
  onset: sigmoid
  peak: plateau_with_variation
  recovery_early: biexponential_decay
  recovery_late: slow_decay

latent_effects:
  autonomic_arousal:
    direction_probabilities:
      increase: 0.80
      decrease: 0.05
      unchanged: 0.15
    magnitude:
      distribution: beta
      alpha: 2.0
      beta: 2.5
  motor_activation:
    direction_probabilities:
      increase: 0.65
      decrease: 0.05
      unchanged: 0.30
  cognitive_load:
    direction_probabilities:
      increase: 0.70
      decrease: 0.05
      unchanged: 0.25

observation_effects:
  eeg:
    lag_sec: [-20, 20]
    response_strength: [0.0, 1.0]
  cardiac:
    lag_sec: [-10, 30]
    response_strength: [0.0, 1.0]
  eda:
    lag_sec: [0, 45]
    response_strength: [0.0, 1.0]
  motor:
    lag_sec: [-10, 60]
    response_strength: [0.0, 1.0]

exceptions:
  no_precursor_probability: 0.10
  opposite_response_probability: 0.10
  missing_modality_probability: 0.15
```

방향 확률은 연구 근거와 실제 데이터 교정 전까지 보수적으로 넓게 둔다.

---

# 7. Feature registry 예시

```yaml
features:
  - name: eeg_af7_alpha_relative_power
    modality: eeg
    source_streams: [muse_s.eeg.AF7]
    window_sec: 30
    unit: ratio
    minimum_coverage: 0.8
    causal: true
    valid_range: [0.0, 1.0]

  - name: rmssd_ms
    modality: cardiac
    source_streams: [polar_h10.ecg]
    window_sec: 60
    unit: ms
    minimum_coverage: 0.9
    causal: true
    requirements:
      min_valid_rr_count: 30

  - name: eda_scr_count
    modality: eda
    source_streams: [galaxy_watch8.eda]
    window_sec: 60
    unit: count
    minimum_coverage: 0.8
    causal: true
    notes: approximate at 1 Hz

  - name: eda_hr_lag_corr_max
    modality: cross_modal
    source_streams: [galaxy_watch8.eda, polar_h10.ecg]
    window_sec: 180
    unit: correlation
    minimum_coverage: 0.8
    causal: true
```

---

# 8. 출력 컬럼 스키마 초안

## `truth/events.parquet`

| 컬럼 | 타입 |
|---|---|
| `run_id` | string |
| `person_id` | string |
| `event_id` | string |
| `event_type` | string |
| `is_target` | bool |
| `start_time_ns` | int64 |
| `onset_time_ns` | int64 |
| `peak_start_time_ns` | int64 |
| `peak_end_time_ns` | int64 |
| `end_time_ns` | int64 |
| `intensity_truth` | float32 |
| `archetype` | string |
| `seed` | uint64 |

## `observed/native/*`

공통:

| 컬럼 | 타입 |
|---|---|
| `run_id` | string |
| `person_id` | string |
| `device_id` | string |
| `stream_name` | string |
| `truth_time_ns` | int64, truth export에서만 |
| `device_time_ns` | int64 |
| `receive_time_ns` | int64 |
| `sample_index` | int64 |
| `packet_id` | int64 |
| `quality_native` | float32 |

신호값 컬럼은 장비 stream별 wide schema를 사용한다.

## `model_ready/timeline_1hz.parquet`

- key: `run_id`, `person_id`, `canonical_time_ns`
- 장비별 1초 요약과 품질
- label은 별도 파일과 1:1 join 가능
- raw truth 컬럼 금지

## `model_ready/labels.parquet`

- key 동일
- `event_id`, `event_type`, `event_phase`
- forecast horizons
- validity flags

---

# 9. 설정 변경 관리

설정 변경마다 manifest에 기록한다.

```json
{
  "schema_version": "1.0",
  "config_sha256": "...",
  "generator_version": "0.1.0",
  "git_commit": "...",
  "created_at_utc": "...",
  "seed": 20260725,
  "profile": "mvp"
}
```

동일한 `schema_version`에서 의미가 바뀌는 변경은 금지한다. 컬럼 의미나 라벨 정의가 바뀌면 minor 또는 major schema version을 올린다.
