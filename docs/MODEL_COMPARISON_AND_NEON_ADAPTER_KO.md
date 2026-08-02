# 센서 조합 비교와 Neon 개인화 어댑터

작성 기준: 2026-08-02 · synthetic `oracle/sanity` validation · locked test 미사용

그래프용 한글 폰트는 `scripts/install_nanum_font.sh`로 사용자 폰트 폴더에 설치한다. 현재
보고서는 `NanumGothic`을 우선 사용하고, 설치되지 않은 환경에서는 Apple SD Gothic Neo로
대체한다.

## 1. 비교 결과를 보는 방법

`reports/sensor_model_comparison_ko.html`은 현재 저장된 validation metric을 같은 표로
읽어 다음 모델을 비교한다.

- `Watch 단독`, `Polar 단독`, `Muse 단독`
- `Watch + Polar`, `Watch + Muse`, `Polar + Muse`
- `Watch + Polar + Muse`
- 기존 전체 계층형 `Logistic`, `HGB`

그래프는 사건 AUCPR, 사건 recall, 사건 F1, 시간당 false alerts, Brier/ECE, 5단계
macro-F1을 분리한다. 사건 지표와 5단계 지표를 하나의 점수로 합치지 않는다. false
alerts·Brier·ECE는 낮을수록 좋고, 나머지 사건·단계 지표는 높을수록 좋다.

현재 산출물의 중요한 해석은 다음과 같다.

1. 합성 데이터의 센서 조합 차이는 확인할 수 있지만, 실제 Galaxy Watch8 신호의 우수성을
   증명하지 않는다.
2. profile 모델의 validation threshold는 profile별로 선택됐으므로, 확률 자체를 모델 간
   직접 비교하지 않고 AUCPR·event recall/F1·오탐·보정 오차를 함께 본다.
3. 실제 Neon accuracy, 장비 동기화, 관찰자 라벨은 아직 `NOT VERIFIED`다.

## 2. Kaggle 3번 오류 상태

과거 availability v3의 Cell 3은 Kaggle scikit-learn 1.6.1에서 로컬 1.9.0으로 저장된
`LogisticRegression`을 읽을 때 `multi_class` 속성이 없어 실패했다. 현재 v4 kernel은
그 호환 처리를 포함한 model v3를 읽어 7개 profile × 600행을 모두 `REPRODUCED`로
완료했다. v4 receipt에서 `run_training=false`, `run_locked_test=false`, 실제 데이터
`NOT VERIFIED`를 확인한다.

따라서 같은 이전 버전 노트북을 다시 실행하면 오류가 재현될 수 있다. 재현할 때는 다음
최신 리소스를 사용한다.

- Model: `bjcoding/multisensor-goal15-availability` variation `ScikitLearn/oracle-sanity-v1/3`
- Kernel: `bjcoding/goal15-availability-reproduction` version 4

## 3. Neon 관측 데이터 연결 경계

현재 production branch의 `public.sensor_batches`는 Watch payload가 들어오는 transport
테이블이다. 2026-08-02 17:47 KST 읽기 전용 확인에서는 3,284행·11 session·5 sensor path였고,
상태는 `not_configured` 3,163건과 `insufficient_signal` 121건이었다. `corrected_timestamp_ms`,
`stage`, `model_version`은 모두 비어 있고, 최근 `insufficient_signal` 행의 품질 사유는
`clock_unsynchronized`다. 이는 전송·디코드 상태이지 모델 입력 또는 실제 성능 라벨이 아니다.

Neon에서 읽은 행은 다음 순서로 처리한다.

```text
sensor_batches (read-only)
  → neon_mapping.py (opaque bytea decode)
  → 외부 clock correction (corrected_utc; 현재 교정값 0건)
  → 1 Hz observed features + quality/missing
  → personal baseline adapter
  → 관찰자 라벨이 등록된 뒤에만 model-ready projection 검토
```

`received_at`을 생리 시각으로 사용하지 않는다. 초기 데이터에는 sensor timestamp와
수신 시각의 큰 차이가 관찰됐으므로, `clock_offset_ms`는 동기화 보고서에서 외부 측정한
값만 입력한다. 네트워크 지연과 ECG–PPG 생리적 시차를 하나의 clock offset으로 합치지
않는다.

## 4. 개인 기준선 어댑터 사용법

먼저 Neon의 읽기 전용 결과를 parquet으로 저장하고 `neon_mapping.map_sensor_batches`로
long table을 만든다. 이후 외부 보정 offset을 넣어 1 Hz 관측 feature를 만든다.

```bash
multisensor-ml neon-adapter derive \
  --input mapped_watch.parquet \
  --output observed_1hz.parquet \
  --clock-offset-ms 1234.0 \
  --physiological-lag-ms 0

multisensor-ml neon-adapter fit \
  --input observed_1hz.parquet \
  --output adapter_p1.json \
  --person-id child-001 \
  --model-version hierarchical-v5
```

어댑터가 만드는 값은 관측 feature의 robust center/scale, `personal_weight`, 품질 신뢰도,
결측·동기화 상태다. 개인 weight는 다음 형태로 상한을 둔다.

```text
n_eff / (n_eff + 1800) × quality_confidence
```

현재 기본 상한은 `0.75`다. 이 값은 `.skops`의 Logistic/HGB 계수를 덮어쓰지 않는다.
어댑터 manifest에는 항상 `model_weights_changed=false`, `promotable=false`,
`promotion_block_reason=NO_OBSERVED_LABELS`가 남는다. 따라서 라벨 없는 실제 데이터가
합성 모델을 조용히 재학습하거나 승격시키지 않는다.

## 5. 다음 실제 데이터 단계

1. corrected UTC와 품질·결측·OOD 컬럼을 Neon 파생 시계열에 등록한다.
2. 관찰자가 기록한 onset/peak/recovery/end를 별도 label set으로 등록한다.
3. Watch 단독·Watch+Polar 등 profile별 model-ready projection을 동일 split으로 평가한다.
4. 개인 기준선 weight cap 후보는 validation에서만 비교하고, 여러 날·여러 session의
   반복 이탈과 라벨 성능 저하가 확인될 때만 재학습 후보로 기록한다.
5. 실제 라벨과 동기화가 검증되기 전에는 정확도·AUCPR을 `PASS`로 표현하지 않는다.
