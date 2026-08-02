# Neon 실제 데이터 매핑 점검 보고서

작성 기준: 2026-08-02 17:47 KST 읽기 전용 스냅샷

## 결론

현재 Neon 운영 브랜치의 `public.sensor_batches`는 **전송 계약 수준에서는 매핑 가능**하다. 다섯 개
Galaxy Watch 경로의 `bytea` payload가 Android `DataOutputStream` big-endian v1 계약으로 고정되어
있고, 경로별 디코더를 ML 저장소에 추가했다.

반면 현재 데이터는 **실제 `model_ready` 학습 입력으로 바로 사용할 수 없다**.

- payload 내부 센서 시각을 교정할 clock-sync 테이블이 운영 브랜치에 없다.
- 최신 스냅샷에서는 `corrected_timestamp_ms`가 0건이고, 새로 들어온 121건은
  `insufficient_signal`로 분류되며 품질 사유가 `clock_unsynchronized`다.
- `analysis_status`는 3,163건이 `not_configured`, 121건이 `insufficient_signal`이고,
  `stage`·`model_version`은 모두 비어 있다.
- 관찰자 onset/peak/recovery 라벨이 Neon에 없으므로 실제 행동·사건 정확도를 산출할 수 없다.
- 현재 운영 브랜치에는 Watch 데이터만 있고 Muse·Polar·ECG 기준 신호는 없다.

따라서 이번 결과는 `oracle/sanity`와 전송 매핑 검증으로만 기록한다. 실제 정확도와 장비 동기화는
`NOT VERIFIED`다.

## 관찰된 운영 계약

2026-08-02 17:47 KST 읽기 전용 조회에서는 3,284개 배치, 11개 세션, 5개 Watch 경로가
적재되어 있었다. 상태별로 `not_configured` 3,163건, `insufficient_signal` 121건이며,
`corrected_timestamp_ms`, `stage`, `model_version`은 모두 미지정이었다. 수집 중인 테이블이므로
행 수는 조회 시점에 따라 변할 수 있다. 운영 테이블의 주요 컬럼은
다음과 같다.

| 영역 | 컬럼 | 의미 |
|---|---|---|
| 식별 | `sensor_path`, `batch_id`, `session_id`, `sequence` | 센서 경로·세션·배치 순서 |
| 전송 | `payload`, `payload_sha256`, `received_at` | 불투명 바이너리와 수신 시각 |
| 분석 | `analysis_status`, `stage`, `model_version`, `analyzed_at` | 미설정 또는 clock 동기화 실패 |
| 동기화 | `source_timestamp_ms`, `corrected_timestamp_ms`, `quality`, `quality_sufficient` | 원천 시각·교정 시각·품질 사유 |

경로별 관찰 계약은 다음과 같다.

| Android 경로 | 모델 종류 | 관찰 주기/내용 |
|---|---|---|
| `/kidsignal/v1/sensor-batch` | `heart_rate_ibi` | HR/IBI, 대체로 1 Hz |
| `/kidsignal/v1/eda-batch` | `eda` | 피부전도, 대체로 1 Hz, 품질 상태 포함 |
| `/kidsignal/v1/accelerometer-batch` | `accelerometer` | raw XYZ, 25 Hz 배치 |
| `/kidsignal/v1/ppg-batch` | `ppg` | green/IR/red raw 채널, 25 Hz 배치 |
| `/kidsignal/v1/skin-temperature-batch` | `skin_temperature` | object/ambient, 희소 샘플 |

`src/multisensor_ml/neon_mapping.py`의 디코더는 버전·개수·trailing bytes·UTF-8·payload 길이를
검사하고, 다음 시각을 모두 보존한다.

- `sensor_timestamp_ms`, `sensor_timestamp_utc`: 생리 신호의 원천 시각
- `received_at`: 서버 수신 시각
- `session_id`, `batch_id`, `sequence`, `sensor_path`: 재현·감사 식별자

생리 시각으로 `received_at`를 대체하지 않는다. clock correction 이후의 `corrected_utc`만
파생변수 계산과 모델 입력의 공통 시간축으로 사용한다.

## 모델 입력으로의 매핑 계약

디코드된 장비 관측값은 다음 `model_ready` 파생변수로 내려보내는 것이 적절하다. 원시 고주파
payload 전체를 Timescale에 장기 보관하는 것이 목적이 아니다.

| 그룹 | 파생변수 예시 | 품질/동기화 메타데이터 |
|---|---|---|
| 심박 | `hr_bpm`, `ibi_ms`, `hr_mean_5s`, `hr_slope_30s`, `rmssd_60s` | HR/IBI status, 유효 샘플 수 |
| EDA | `eda_us`, `eda_mean_30s`, `eda_slope_60s`, phasic proxy | 센서 status, 결측 초 |
| 움직임 | `acc_norm`, `acc_jerk`, `motion_mean_5s`, `motion_energy_30s` | 축별 유효율, dropout |
| PPG | 채널별 robust level/quality, pulse-rate proxy | green/IR/red status, 품질 |
| 온도 | `skin_temp_object_c`, `skin_temp_ambient_c`, 차이·추세 | status, 유효 샘플 수 |
| 공통 | `corrected_utc`, `clock_offset_ms`, `drift_ppm`, `jitter_ms`, `missing_seconds`, `quality_confidence`, `adapter_version` | 세션·장비 pair·reference device |

이 파생변수에서 기존 Oracle 모델의 causal time/context/history 피처를 만들고, 가용 장비에 맞는
`watch_only`, `watch_polar`, `watch_muse`, `watch_polar_muse` 등의 adapter profile을 선택한다.
실제 장비가 측정하지 않는 latent truth, hidden archetype, event truth를 관측 입력으로 복사해서는
안 된다.

## 동기화 판정

초기 세션의 일부 배치에서 `sensor_timestamp_ms - received_at`가 약 `-60,413,112 ms`였고,
최근 수집분도 품질 사유가 `clock_unsynchronized`이며 `corrected_timestamp_ms`가 채워지지
않았다. 이는 단순한 네트워크 지연으로 가정할 수 없는 시계 기준 불일치다.

권장 공통 시간축은 `corrected UTC`이며, 이후 별도 기록할 항목은 다음과 같다.

`dataset_id`, `session_id`, `person_key`, device pair, reference device, offset ms, drift ppm,
jitter ms, physiological lag ms, overlap seconds, correlation, status.

현재는 교정값을 꾸미지 않고 `NOT_AVAILABLE_TRUTH_ONLY`로 남긴다. Polar H10을 cardiac reference로
사용하고 Watch ECG는 calibration-only로 다루며, ECG–PPG 생리적 시차와 clock error를 분리한다.

## 저장 권장안

1. `sensor_batches`의 원시 payload와 SHA-256은 보존하되, 고주파 원본 장기 보관은 object storage로
   보낸다.
2. Timescale hypertable에는 세션·장비·`corrected_utc`별 1 Hz 정규화 관측과 5/15/30/60초 causal
   파생변수, 품질·결측·동기화 메타데이터를 저장한다.
3. 예측 결과는 `model_version`, `profile_id`, `feature_schema_hash`, `event_probability`,
   `predicted_stage`, `decision_state`와 함께 별도 테이블에 append-only로 기록한다.
4. 관찰자 라벨이 들어오면 onset/peak/recovery/end와 `label_source`·신뢰도를 별도 outcome 테이블에
   저장한다. 예측 결과를 라벨로 덮어쓰지 않는다.

운영 브랜치에는 현재 `public.sensor_batches`만 있고, 준비된 Timescale telemetry schema는 임시
브랜치에만 있다. 운영 마이그레이션은 별도 승인 전까지 적용하지 않는다.

## 검증 파일

- 디코더/매핑: `src/multisensor_ml/neon_mapping.py`
- 매핑 테스트: `tests/test_neon_mapping.py`
- 운영 ML·모델 상태: `not_configured`/`insufficient_signal`, `stage/model_version` 미지정,
  실제 accuracy `NOT VERIFIED`
- 장비 동기화: `NOT_AVAILABLE_TRUTH_ONLY`
