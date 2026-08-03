# 최종 Optuna·ONNX 모델 사용 안내

## 결론

현재 합성 MVP3의 validation AUCPR 기준으로 최종 후보 두 개를 만들었다.

| 후보 | 입력 | 피처 수 | AUCPR | event recall | event F1 | false alerts/hour | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `galaxy_watch` | Galaxy Watch 단일 | 127 | 0.8074 | 0.4050 | 0.5670 | 1.813 | 0.2191 | 0.1925 |
| `galaxy_watch_h10` | Galaxy Watch + Polar H10 | 146 | **0.8345** | **0.4953** | **0.6451** | 3.089 | **0.1774** | **0.1083** |

두 모델은 같은 6명 validation 전체 타임라인에서 비교했다. H10을 추가한 후보는 AUCPR이
0.0271(상대 약 3.36%) 높고 recall·F1·확률 보정도 좋아졌지만, false alerts/hour는
증가했다. 따라서 실제 운영 threshold는 실제 관찰자 라벨이 확보된 뒤 validation에서
다시 정하고 locked test에는 적용하지 않는다.

## 재현 조건

- 데이터: `mvp3-oracle-v1`, 합성 `oracle/sanity`만 사용
- 분할: 사람 기준 train 24명 / validation 6명 / locked test 6명
- 학습 행: train 597,466행, validation 120,000행
- Optuna: TPE sampler, 12 trials/후보, seed `20260725`, CPU LightGBM `n_jobs=1`
- 목적함수: validation `average_precision_score`(AUCPR)
- ONNX: opset 15, Cloud Run CPU `CPUExecutionProvider`
- threshold: 현재 0.5(테스트 데이터로 변경하지 않음)

재실행:

```bash
cd /Users/baital/dev/multisensor_ml/.worktrees/kaggle-ml-dl-benchmark
PYTHONPATH=src:. .venv/bin/python scripts/tune_dual_sensor_onnx.py \
  --project-root /Users/baital/dev/multisensor_ml \
  --series mvp3-oracle-v1 \
  --output services/onnx_api/models/goal15-final-v1 \
  --trials 12 --max-train-rows 600000 --max-validation-rows 120000 \
  --seed 20260725
```

## 산출물

- `services/onnx_api/models/goal15-final-v1/galaxy_watch/model.onnx`
- `services/onnx_api/models/goal15-final-v1/galaxy_watch_h10/model.onnx`
- 각 디렉터리의 `manifest.json`, `feature_schema.json`, `optuna_study.json`
- 상위 `experiment_manifest.json`의 variant·파일 SHA-256

모델 SHA-256:

| 후보 | SHA-256 |
|---|---|
| `galaxy_watch` | `5d378837a8d363af4d7114adb0925b325fdd207cc40dd971149fdb42c21eb370` |
| `galaxy_watch_h10` | `ddfa4bc277c7e565899f11f34c089279ed5f633a8b6e9efc8ba928388cb91044` |

입력 피처는 현재 latent 기반 파생변수의 명시적 synthetic proxy다. Galaxy Watch/H10의
실제 센서 컬럼을 의미한다고 간주하면 안 되며, 실제 adapter가 같은 `feature_schema.json`
을 채우기 전에는 실전 정확도나 의료 판단에 사용하지 않는다.

## Cloud Run API

서비스 코드는 `services/onnx_api/`에 있다. 두 bundle이 모두 검증되어야 앱이 뜨며,
`/healthz`와 `/v1/predict`만 제공한다.

```bash
MODEL_ROOT=services/onnx_api/models/goal15-final-v1 \
.venv/bin/uvicorn services.onnx_api.app:app --host 0.0.0.0 --port 8080
```

실제 요청에는 해당 variant manifest의 **모든** 피처가 필요하다. 응답은 확률, 0.5
threshold 판정, 모델 ID, `oracle/sanity`, `NOT VERIFIED`, `locked_test_read=false`를
함께 반환한다. 배포 스크립트는 `ALLOW_CLOUD_RUN_DEPLOY=1`을 명시하기 전에는 거부하며,
아직 Google Cloud Run에 배포하지 않았다.

## 실제 데이터 경계

Neon 관측 데이터는 현재 3,284행·11 session이 확인되었지만 corrected UTC 0행,
관찰자 단계 라벨 0행, 등록 모델 버전 0행으로 `model_ready=false`다. 따라서 이
커밋의 정확도는 합성 sanity 결과이며, 실제 데이터 성능은 `NOT VERIFIED`다.
실전 테스트 전에는 corrected UTC, 독립 관찰 onset/peak/recovery/end 라벨, device
adapter의 피처 생성, 모델 버전 등록을 먼저 통과시켜야 한다.
