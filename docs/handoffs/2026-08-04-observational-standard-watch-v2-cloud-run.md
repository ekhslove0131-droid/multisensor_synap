# Kidsignal Cloud Run handoff — Watch standard v2

## 전달 경계

이 handoff는 모델 artifact와 streaming parity 계약만 제공한다. Kidsignal 저장소,
Neon, GCP/Cloud Run에는 이 작업에서 변경이나 배포를 하지 않았다.

## 전달 파일

`artifacts/observational-standard-30m-watch-baseline-shadow-v1/`의 다음 파일을 하나의
불변 디렉터리로 복사한다.

```text
model.onnx
manifest.json
feature_schema.json
golden_fixture.json
runtime.json
runtime_requirements.txt
SHA256SUMS.json
```

Cloud build는 복사 후 `SHA256SUMS.json`과 `manifest.json.artifacts`를 모두 검증해야
한다. `feature_schema.json`의 canonical SHA-256은 다음과 같다.

```text
2857f8a16cd4450a18f8701c1c9c9f397dee4b291fefd2475279883a0f8a29de
```

## 환경변수

```text
KIDSIGNAL_MODEL_DIR=/app/model/observational-standard-30m-watch-baseline-shadow-v1
KIDSIGNAL_MODEL_RELEASE=observational_standard_30m_watch_baseline_shadow_v1
KIDSIGNAL_FEATURE_SCHEMA_VERSION=watch_standard_16_v2
KIDSIGNAL_FEATURE_SCHEMA_SHA256=2857f8a16cd4450a18f8701c1c9c9f397dee4b291fefd2475279883a0f8a29de
KIDSIGNAL_BASELINE_VERSION=personal_robust_baseline_900s_v2
KIDSIGNAL_LOAD_FORMULA_VERSION=watch_positive_z_bounded_ema_v2
KIDSIGNAL_QUALITY_FORMULA_VERSION=watch_core_quality_min_v1
KIDSIGNAL_TARGET_VERSION=observational_future_mean_1800_valid90_gap5_v2
KIDSIGNAL_REAL_DATA_STATUS=NOT VERIFIED
KIDSIGNAL_ONNX_INTRA_OP_THREADS=1
KIDSIGNAL_ONNX_INTER_OP_THREADS=1
```

Cloud Run이 환경변수와 manifest 값이 다르면 시작을 거부해야 한다. feature vector에
NaN/Inf가 있거나 `feature_decisionable=false`이면 ONNX를 호출하지 않는다.

## Python runtime와 ONNX 호출

bundle의 `runtime_requirements.txt`를 Cloud 이미지에 설치한다. ONNX Runtime은 CPU
provider와 intra/inter-op thread 1을 사용한다.

```python
options = onnxruntime.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
session = onnxruntime.InferenceSession(
    f"{model_dir}/model.onnx",
    sess_options=options,
    providers=["CPUExecutionProvider"],
)
prediction = session.run(
    ["prediction"],
    {"features": ordered_vector.astype("float32").reshape(1, 16)},
)[0][0, 0]
```

현재 selector는 `watch_load_median_300`, 즉 0-based index 11만 반환한다.

## Cloud Run entrypoint

모델 저장소는 ingest/worker HTTP 앱을 소유하지 않는다. Kidsignal private inference
service의 기존 FastAPI 앱을 사용한다.

```text
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
```

서비스 시작 단계에서 다음을 한 번 실행하는 loader를 연결한다.

1. manifest 및 모든 SHA-256 검증
2. schema version/hash와 환경변수 비교
3. ONNX input `float32[N,16]`, output `[N,1]` 확인
4. `golden_fixture.json`의 1,800 input rows를 streaming feature builder로 replay
5. ordered vector와 ONNX prediction을 golden 값과 비교

하나라도 다르면 revision은 readiness 실패로 남기고 traffic을 받지 않는다.

## 응답 경계

baseline-only shadow는 다음만 반환·저장한다.

```text
standard_30m
current_load
ema_30m
ema_6h
quality_confidence
feature_decisionable
decision_reason
baseline_maturity
real_data_status=NOT VERIFIED
stage=null
```

행동, pattern, 5단계 stage, 실제 정확도/AUCPR는 반환하지 않는다. Cloud가 prediction
ledger를 한 번 생성하고 Phone은 delivery/display receipt 또는 독립 관찰 라벨만
되돌린다는 기존 계약을 유지한다.

## 로컬 재현 명령

```bash
.venv/bin/multisensor-ml observational-standard export-baseline-shadow \
  --output artifacts/observational-standard-30m-watch-baseline-shadow-v1

.venv/bin/multisensor-ml observational-standard verify-baseline-shadow \
  --bundle artifacts/observational-standard-30m-watch-baseline-shadow-v1
```

배포 승인 전 완료 조건은 model repo 검증, Kidsignal golden parity, idempotency 및
missing/OOD 테스트다. 비트래픽 shadow revision 이후에도 production traffic은 별도
승인 없이는 전환하지 않는다.
