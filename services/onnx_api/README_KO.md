# Cloud Run ONNX 추론 API

이 서비스는 두 개의 고정 ONNX 입력 계약을 제공한다.

- `galaxy_watch`: Galaxy Watch 단일 입력
- `galaxy_watch_h10`: Galaxy Watch + Polar H10 입력

현재 모델은 합성 `oracle/sanity`로 튜닝된 후보이며, 실제 Neon 데이터 정확도는
`NOT VERIFIED`다. 실제 데이터가 model-ready가 되기 전에는 의료적 판단이나 운영
승격으로 사용하지 않는다.

## API

```text
GET  /healthz
POST /v1/predict
```

요청 예:

```json
{
  "variant": "galaxy_watch",
  "request_id": "demo-001",
  "features": {
    "motor_activation__robust_z": 0.2,
    "cognitive_load__mean_5s": 0.1,
    "time_sin": 0.3
  }
}
```

실제 요청에는 해당 variant의 `manifest.json`에 있는 모든 `feature_names`가 필요하다.
응답에는 `probability`, `event_predicted`, `threshold`, 모델 ID, 데이터 상태,
`locked_test_read=false`가 포함된다.

## 로컬 실행

```bash
MODEL_ROOT=services/onnx_api/models/goal15-final-v1 \
.venv/bin/uvicorn services.onnx_api.app:app --host 0.0.0.0 --port 8080
```

## Cloud Run 이미지

저장소 루트에서 다음처럼 빌드한다.

```bash
docker build -f services/onnx_api/Dockerfile -t multisensor-goal15-onnx .
docker run --rm -p 8080:8080 multisensor-goal15-onnx
```

실제 배포는 `scripts/deploy_cloud_run_onnx.sh`가 `ALLOW_CLOUD_RUN_DEPLOY=1`일 때만
실행하도록 잠겨 있다. 배포 전 이미지의 두 manifest hash와 `/healthz` readback을
확인해야 한다.
