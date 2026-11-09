# Observational Standard Dual-Shadow CANDIDATE v1 인계서

상태: **EXPLORATORY / CANDIDATE**

이 문서는 모델 worktree의 immutable artifact를 Kidsignal shadow 검증에 전달하기 위한 것이다. Kidsignal, Neon, GCP, Git에는 이 작업에서 변경을 가하지 않았다.

## 1. 릴리스와 경계

| 항목 | 값 |
|---|---|
| CANDIDATE release | `observational_standard_30m_watch_baseline_shadow_candidate_v1` |
| artifact 경로 | `/Users/baital/dev/multisensor_ml/.worktrees/kaggle-ml-dl-benchmark/artifacts/observational-standard-30m-watch-baseline-shadow-candidate-v1` |
| 알고리즘 | `ridge` (CPU, 연속 표준 예측 후보) |
| ACTIVE reference | `observational_standard_30m_watch_baseline_shadow_v1` |
| ACTIVE model SHA-256 | `5bf7d697bb8bb3310afa8db805e4d1ac9eaa6159f84cfff5b843f2ad3c511bd3` |
| CANDIDATE `model.onnx` SHA-256 | `c815457ea5d68160adc67315f784a62a3f01d78e1dacf890836c2fd8de16f0aa` |
| CANDIDATE `manifest.json` SHA-256 | `fb642ffec2fae42ff52483421819de48d008d0a09341047c224b95c6efb4713b` |
| CANDIDATE `feature_schema.json` SHA-256 | `fc5f4741c1182236a0944ef6e2043d27b838920e2f29870ff9ddbcaf9b14c7da` |
| canonical `feature_schema_hash` | `2857f8a16cd4450a18f8701c1c9c9f397dee4b291fefd2475279883a0f8a29de` |
| CANDIDATE `SHA256SUMS.json` SHA-256 | `a55806c021f5e099ec84a4238d11e1da21185b322acd5a2026dc5dbcb7b7bee4` |
| device profile | `galaxy_watch8_v1` |
| delivery | `delivery_eligible=false`; Phone delivery 금지, shadow ledger만 허용 |
| real data | `real_data_status=NOT VERIFIED` |
| stage/pattern/behavior | 모두 `null` |

이 후보는 ACTIVE를 대체하거나 승격하도록 추천하지 않는다. `EXPLORATORY`에서 동일 window·동일 16-vector를 비교하고, 별도 실제 라벨과 사용자 명시 승인이 있을 때만 다음 gate를 검토한다.

## 2. ONNX 입출력 계약

- Runtime: `onnxruntime==1.28.0`, `CPUExecutionProvider`만 사용, intra/inter-op threads 각 1.
- Input: 이름 `features`, `float32[N,16]`.
- Output: 이름 `variable`, `float32[N,1]`.
- 반환값: `prediction_value`(연속 watch-load 표준 예측값). 1,800 eligible second가 누적되기 전에는 inference 결과를 만들지 않고 `NOT_DECISIONABLE`로 남긴다.
- CANDIDATE에는 stage, 행동 코드, 임상 판단, t2/t3/t4/t5 threshold가 없다.

Canonical ordered feature names:

```text
watch_eda_z
watch_hr_z
watch_motion_z
watch_load_raw
watch_eda_mean_30
watch_hr_mean_30
watch_motion_mean_30
watch_eda_slope_60
watch_hr_slope_60
watch_motion_slope_60
watch_load_std_60
watch_load_median_300
watch_load_ema_1800
watch_load_ema_21600
quality_confidence
watch_ineligible_fraction_60
```

전처리는 corrected UTC 1 Hz, HR+EDA+ACC core 품질 gate, 900초 baseline과 1,800 eligible-second decision warm-up을 사용한다. `quality_confidence` 누락·core 품질 실패·비유한 feature는 fail-closed이며, quality를 load에 곱하지 않는다.

## 3. Threshold와 calibration 상태

```json
{
  "t2": null,
  "t3": null,
  "t4": null,
  "t5": null,
  "threshold_state": "UNSET",
  "blocker": "NO_SEPARATE_CALIBRATION_TRAINING_EVIDENCE"
}
```

평가 라벨을 threshold fitting에 사용하지 않았다(`threshold_fit_evaluation_labels=false`, `locked_test_read=false`). 따라서 현재 CANDIDATE는 패턴 단계나 행동을 출력할 수 없다.

## 4. ACTIVE와 golden parity

동일한 `golden_fixture.json`(1,800 raw rows, canonical ordered 16-vector)을 ACTIVE와 CANDIDATE에 복사해 사용한다.

| 확인 항목 | 결과 |
|---|---|
| ordered feature names | 16개, canonical hash 일치 |
| ACTIVE golden output | `prediction` output, fixture와 일치 |
| CANDIDATE golden output | `variable` output, `prediction_value=0.1460956186056137` |
| candidate golden file | `candidate_golden_output.json` |
| feature decisionable | 마지막 row에서 true (1,800 eligible seconds 이후) |
| stage | `null` |

## 5. 모델 레지스트리 최소 handoff 파일

필수 전달 파일:

```text
model.onnx
manifest.json
feature_schema.json
golden_fixture.json
candidate_golden_output.json
runtime.json
runtime_requirements.txt
SHA256SUMS.json
```

재현·감사 보관 파일:

```text
model.skops
training_evidence.json
metrics_validation.parquet
predictions_validation.parquet
split_assignments.parquet
baseline.parquet
```

권장 환경변수:

```text
KIDSIGNAL_MODEL_RELEASE=observational_standard_30m_watch_baseline_shadow_candidate_v1
KIDSIGNAL_MODEL_DIR=/app/model/observational-standard-30m-watch-baseline-shadow-candidate-v1
KIDSIGNAL_MODEL_DEVICE_PROFILE=galaxy_watch8_v1
KIDSIGNAL_MODEL_DELIVERY_ELIGIBLE=false
KIDSIGNAL_MODEL_INPUT_NAME=features
KIDSIGNAL_MODEL_OUTPUT_NAME=variable
KIDSIGNAL_MODEL_FEATURE_SCHEMA_HASH=2857f8a16cd4450a18f8701c1c9c9f397dee4b291fefd2475279883a0f8a29de
```

Entrypoint는 `onnxruntime.InferenceSession(model.onnx, providers=["CPUExecutionProvider"], sess_options(intra_op_num_threads=1, inter_op_num_threads=1))`로 고정한다. 모델 로드 전 `SHA256SUMS.json`, manifest release/status/delivery flag, feature schema hash와 golden parity를 검사한다. CANDIDATE 결과는 prediction ledger의 shadow 행으로만 기록하고 Phone delivery outbox에는 넣지 않는다.

## 6. Gate와 차단 요인

- 현재 gate: `EXPLORATORY`.
- `COMPARABLE` 이전에 필요한 것: 동일 corrected-UTC window의 ACTIVE/CANDIDATE pair, 결측/OOD/fail-closed parity, 세션·사람별 bounded metrics, 별도 calibration 근거.
- 현재 차단: Production actual labels=0, prediction_value도 아직 0; synthetic source만 있어 실제 성능과 threshold를 검증할 수 없음.
- 실제 관찰 라벨이 수집되기 전에는 추천·승격·임상/행동 해석을 하지 않는다.

## 7. bounded 검증 결과

| 검사 | 상태 | 증거 |
|---|---|---|
| candidate/observational 표적 테스트 | PASS | `24 passed, 2 warnings` |
| mypy | PASS | `Success: no issues found in 46 source files` |
| Ruff | PASS | 대상 source/test 파일 `All checks passed!` |
| `uv lock --check` | PASS | `Resolved 109 packages in 6ms` |
| CANDIDATE hash·schema·ONNX·golden | PASS | `verify-candidate-shadow` → `VERIFIED`, prediction `0.1460956186056137` |
| 전체 pytest (`PYTHONPATH=src`) | NOT PASS | 기존 `scripts` 모듈 2건 collection import error |
| 전체 pytest (`PYTHONPATH=src:.`) | NOT VERIFIED | 장시간 실행 중 실패 표시를 확인한 뒤 bounded 중단; 전체 결과를 PASS로 확정하지 않음 |

전체 pytest의 실패/미완료 상태는 CANDIDATE bundle 검증 PASS와 별도다. 다음 검증에서는 실패한 전체-suite 항목만 재현해 원인을 분리해야 한다.
