# Monitoring v2: 표준선 30분 예측 + 가상 패턴 모델

문서 버전: `goal1.5/monitoring-v2/v1`
현재 범위: 합성 `oracle/sanity`
실제 Neon 정확도: `NOT VERIFIED`

## 1. 무엇을 만들었나

기존 Goal 1.5 Oracle 번들은 보존하고, 별도 v2 번들을 추가했다.

```text
truth-only multisensor_synth
  → synthetic observed proxy (Watch / H10)
  → 개인 기준선·causal 파생변수·누적 부하
  → M0: 다음 30분 정상 부하 평균 예측
  → M2: 표준 예측 잔차 + 패턴 파생변수
  → 5개 virtual pattern probability
  → ONNX Cloud Run API
```

합성 관측값은 실제 Galaxy Watch8/Polar H10 SDK 신호가 아니다. 실제 장비의
교정·동기화·관찰자 라벨이 연결되기 전까지 모델의 상태는 반드시
`oracle/sanity`, 정확도는 `NOT VERIFIED`로 표시한다.

## 2. 두 모델의 역할

### M0 표준선 모델

각 사람의 초기 900초 warm-up으로 신호별 중심·변동폭을 계산한다.

```text
z_j(t) = (x_j(t) - median_person_j) / (1.4826 × MAD_person_j + ε)
```

현재 행을 제외한 다음 1,800초(30분)의 정상 행 평균을 타깃으로 한다.
사건 행은 표준 타깃 평균에서 제외하고, 미래 행을 입력에 사용하지 않는다.

```text
y_standard(t) = mean(load(t+1:t+1800) | event_binary = 0)
```

M0는 LightGBM 회귀 모델이다. validation에서는 현재 300초 causal 평균을
persistence 기준선으로 같이 계산한다. 최종 manifest에 두 MAE/RMSE를 모두
기록하여 표준선 예측이 기준선보다 실제로 나아졌는지 확인한다.

### M2 가상 패턴 모델

M0 예측값을 먼저 계산한 뒤 패턴 입력에 붙인다.

```text
standard_residual(t) = current_watch_load(t) - M0_forecast(t)
S_k(t) = f_k(
  personal_robust_z,
  causal_mean/std/slope,
  instantaneous_load,
  cumulative_load,
  context,
  quality,
  standard_residual
)
p_k(t) = sigmoid(S_k(t))
```

패턴 head는 다음 5개다.

- `meltdown_like`
- `tantrum_like`
- `sensory_seeking_like`
- `stereotypy_like`
- `shutdown_like`

각 head는 LightGBM one-vs-rest 분류 모델로 학습하고 validation F1 최대
threshold를 고정한다. 행동명은 진단·자동 개입 명령이 아니라 연구용 후보
라벨이다. 특히 `tantrum_like`는 센서만으로 확정할 수 없으며, 실제 시스템에서는
요구·의사소통·관찰자 맥락 라벨이 있어야 `DECISIONABLE`이 된다.

## 3. 센서 조합

| variant | 입력 관측 축 | 출력 |
|---|---|---|
| `galaxy_watch` | Watch EDA·심박·움직임·온도에서 만든 causal 피처 | M0 + 5개 패턴 확률 |
| `galaxy_watch_h10` | Watch 축 + H10 심박·HRV·흉부 움직임 | M0 + 5개 패턴 확률 |

모델은 사람별 별도 모델을 만들지 않는다. 개인 기준선은 입력 피처에 반영하고,
실제 사용자 라벨이 축적되면 별도 calibration adapter를 validation으로 비교한다.

## 4. 생성·학습 재현

현재 로컬 v2 quick 데이터는 3개 seed × 12명 = 36명이며 split은 고정된
24/6/6 사람 단위다. raw truth와 model-ready parquet는 생성 캐시이므로 Git에는
넣지 않는다.

```bash
cd /Users/baital/dev/multisensor_ml/.worktrees/kaggle-ml-dl-benchmark
./scripts/build_monitoring_v2.sh
```

개별 단계가 필요하면:

```bash
.venv/bin/multisensor-ml materialize-synthetic \
  --config configs/standard_pattern_v2_quick.yaml
.venv/bin/multisensor-ml monitor-v2 run \
  --config configs/monitoring_v2_quick.yaml
```

결과 위치:

- model-ready manifest: `data/model_ready/standard-pattern-v2-quick/manifest.json`
- 결과 metrics: `artifacts/monitoring-v2/standard-pattern-v2-quick/result.json`
- ONNX bundles: `services/onnx_api/models/goal15-monitor-v2/{galaxy_watch,galaxy_watch_h10}`

각 bundle에는 `manifest.json`, `standard_30m.onnx`, 패턴별 ONNX 5개,
SHA-256, split 계약, validation metrics가 있다. `.pkl`·`.joblib`는 사용하지
않는다.

## 5. Cloud Run 로컬 실행

v2 전용 entrypoint와 Dockerfile을 사용한다.

```bash
MODEL_ROOT_V2=services/onnx_api/models/goal15-monitor-v2 \
.venv/bin/uvicorn services.onnx_api.app_v2:app --host 0.0.0.0 --port 8080
```

```text
GET  /healthz
POST /v2/predict
```

요청에는 해당 variant manifest의 `feature_names`가 필요하다. 응답은
`standard_forecast_30m`, `pattern_probabilities`, `pattern_predicted`를 반환한다.
응답의 `model_scope=oracle/sanity`, `real_data_status=NOT VERIFIED`,
`locked_test_read=false`를 운영 승격 전까지 보존한다.

## 6. 현재 검증 결과와 제한

현재 합성 validation에서 H10을 추가한 표준선 M0의 MAE가 Watch 단독보다 낮게
나오는지 manifest에 기록되며, 패턴별 AUCPR·event recall·F1·false alerts/hour·
Brier를 모두 보존한다. 이 수치는 합성 prior의 재현성 확인이지 실제 행동 예측
성능이 아니다.

다음 단계는 실제 Neon corrected UTC, 장비별 품질·동기화, 독립 관찰자
onset/peak/recovery/end 라벨을 model-ready 계약으로 등록한 뒤 같은 두 variant의
shadow 평가를 수행하는 것이다. 그 전에는 자동 개입·임상 판정·모델 자동 승격을
하지 않는다.
