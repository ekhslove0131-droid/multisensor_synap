# Kidsignal 표준 후보 bundle 계약

상위 계약: [KIDSIGNAL_MODEL_PLATFORM_CONTRACT_KO.md](KIDSIGNAL_MODEL_PLATFORM_CONTRACT_KO.md)

## 목적

노패턴 표준 24-field BigQuery cohort를 단일 model-sync receipt로 선택하고,
TRAIN/VALIDATION 경계를 지킨 Watch-only CPU ONNX CANDIDATE bundle을 만든다.
행동 26-field plane과 LOCKED holdout은 이 진입점의 입력이 아니다.

## 공개 진입점

`run_receipt_bound_standard_candidate_export()`는 다음 값만 받는다.

- create-only model-sync receipt 한 건
- read-only BigQuery client와 관측 principal/time
- UUID·시간·참조 hash를 고정한 canonical export request
- create-only output path
- 계약 fixture에서만 쓰는 명시적 non-fit coefficient provider

임의 배열, 수동 cohort UUID, 수동 public digest, `PreparedStandardDataset`은 공개 입력으로
받지 않는다. 내부에서 receipt를 검증하고 24-field reader를 한 번 호출한 뒤
receipt↔readiness↔prepared UUID·digest·schema·split·purge·행 수를 다시 결합한다.

## 학습·선택 경계

- 사전등록 후보: `mean_constant_v1`, `ridge_l2_alpha_1_v1`
- TRAIN만 후보 계수 생성에 사용한다.
- VALIDATION은 MAE 우선, RMSE 차순 후보 선택에만 사용한다.
- validation 재학습과 LOCKED 조회·평가는 금지한다.
- `CONTRACT_EVIDENCE_ONLY`는 명시적 non-fit provider만 허용한다.
- `REAL_FROZEN_COHORT`는 외부 provider 주입을 거부하고 내부 사전등록 trainer만 사용한다.
- cohort 0행, behavior receipt, lineage 불일치는 query 또는 provider 전에 가능한 가장 이른
  지점에서 fail-closed한다.

## runtime·누출 계약

- 입력: float32 `features[N,16]`
- ONNX 내부 `Gather`: runtime index 11 `watch_load_median_300` 제외
- trainer 입력: float32 `[N,15]`
- 출력: float32 `prediction[N,1]`
- ONNX opset 17, `CPUExecutionProvider`
- target-source 값만 바꾼 두 probe의 prediction은 byte-exact 동일해야 한다.
- `target_version`은 `stable-stress-standard-v1`만 허용한다. legacy 미래 1,800초
  target version을 표준 bundle 계보에 연결하지 않는다.

## immutable archive

ZIP은 다음 13개 member만 허용한다.

- `manifest.json`, `model.onnx`, `feature_schema.json`
- `label_review_schema.json`, `runtime.json`, `runtime_requirements.txt`
- `golden_fixture.json`, `golden_output.json`, `split_manifest.json`
- `training_evidence.json`, `training_projection.json`
- `sealed_evaluation_request.json`, `SHA256SUMS.json`

ZIP entry 시간·권한·플랫폼 metadata와 정렬·압축 수준을 고정해 같은 canonical member
bytes가 같은 archive bytes/SHA를 만든다. output은 create-only다. archive에는 private
subject/capture/hour UUID, raw URI, 행 payload를 넣지 않는다.

manifest는 `FROZEN_CONTRACT + TRAINED_NOT_EVALUATED`, `delivery_eligible=false`,
`promotion_eligible=false`, `sealed_evaluation_status=NOT_REQUESTED`, `stage=null`,
`real_data_status=NOT VERIFIED`를 유지한다. validation 선택은 sealed/final 평가가 아니다.

## 현재 증거와 차단 상태

- focused: 13 passed
- 관련 계약: 194 passed
- current worktree 전체: 641 passed
- exact clean source: 640 passed, clean-checkout 조건의 synthetic materialization 1건 expected skip
- model-side verifier와 실제 Cloud bundle/golden verifier: 통과
- 계약 fixture의 실제 estimator fit: 0

교차검증 archive는 가짜 계수와 비개인 fixture를 쓴 `CONTRACT_EVIDENCE_ONLY` 증거다.
BigQuery standard view는 0행이므로 현재 상태는 계속
`BLOCKED_NO_REAL_COHORT / NOT EVALUABLE / fit_call_count=0`이다. 실제 모델 성능,
shadow traffic, delivery 또는 promotion을 증명하지 않는다.
