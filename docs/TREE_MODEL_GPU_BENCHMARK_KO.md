# GPU 트리 모델·ExtraTrees 비교 보고서

## 결론

Kaggle GPU 커널에서 XGBoost와 LightGBM을 `cuda`로 강제하고, ExtraTrees를 CPU 기준
후보로 함께 학습했다. 합성 `oracle/sanity`의 train 24명·validation 6명만 사용한
검증에서 LightGBM이 AUCPR, 사건 F1, 시간당 오탐의 균형이 가장 좋아 앵커 후보로
선택되었다. 소프트 앙상블은 보조 후보로 보존했으며 자동 승격하지 않았다.

이 결과는 실제 Galaxy Watch8 데이터나 행동 예측 정확도가 아니다. 실제 Neon 데이터는
현재 shadow 준비 조건을 충족하지 않아 [실전 테스트 차단 보고서](../reports/tree_model_real_shadow_ko.html)에
`BLOCKED_NOT_MODEL_READY`로 기록되어 있다.

## 실행·검증 계약

- Kaggle 커널: `bjcoding/multisensor-goal15-gpu-tree-benchmark-extratrees` v4
- 데이터셋: `bjcoding/multisensor-goal15-tree-gpu-current-bounded`
- prepared schema: `goal1.5/prepared/v1`
- 분할: train 24명 / validation 6명 / locked test 0명
- validation 행: 120,000개, positive 55,289개
- 모델: XGBoost·LightGBM 3 seed·160 trees, ExtraTrees 3 seed·64 trees
- XGBoost·LightGBM: `use_gpu=true`, `gpu_required=true`, 실행 장치 `cuda`
- ExtraTrees: scikit-learn CUDA 구현이 없어 `cpu`; GPU 모델과 실행 장치를 혼합하지 않음
- locked test: `locked_test_read=false`; test 임계값 재조정 없음
- 실행 시간: Kaggle 로그 기준 약 1,100.7초; 산출물과 로그 hash를 artifact에 기록
- 폰트: `NanumGothic-Regular.ttf`를 사용해 그래프의 한글 라벨 생성

Kaggle 출력의 장치 메타데이터는 XGBoost·LightGBM에 `cuda`, GPU 요청·필수 `true`,
ExtraTrees에 `cpu`를 기록한다. 이는 실행 계약과 로그 증거이며, 실제 장치 사용률
측정값을 대신하지 않는다.

## 전체 validation 결과

| 모델 | 장치 | AUCPR | 사건 recall | 사건 F1 | 시간당 오탐 | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|---:|
| LightGBM | cuda | 0.830078 | 0.567165 | 0.678253 | 8.085417 | 0.169417 | 0.042910 |
| 소프트 앙상블 | mixed | 0.829873 | 0.565158 | 0.675282 | 8.347985 | 0.168358 | 0.033006 |
| XGBoost | cuda | 0.829549 | 0.566225 | 0.674204 | 8.714746 | 0.167791 | 0.032392 |
| ExtraTrees | cpu | 0.784629 | 0.562101 | 0.636864 | 15.601243 | 0.192284 | 0.073178 |

LightGBM은 이 bounded validation에서 AUCPR·F1·시간당 오탐이 모두 가장 좋았다.
XGBoost는 Brier와 ECE가 가장 낮았고, 앙상블은 ExtraTrees를 포함했음에도 LightGBM보다
AUCPR가 소폭 낮았다. 따라서 현재 합성 기준의 선택은 `LightGBM anchor 후보`이며,
실제 운영 모델이 아니다.

## 시간 피처 제거(ablation)

| 모델 | 전체 AUCPR | 시간 제거 AUCPR | AUCPR 변화 | 전체 recall | 시간 제거 recall |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.829549 | 0.835854 | +0.006306 | 0.566225 | 0.589195 |
| LightGBM | 0.830078 | 0.834864 | +0.004786 | 0.567165 | 0.589448 |
| ExtraTrees | 0.784629 | 0.783553 | -0.001076 | 0.562101 | 0.570837 |
| 소프트 앙상블 | 0.829873 | 0.834176 | +0.004303 | 0.565158 | 0.588978 |

XGBoost·LightGBM·앙상블은 시간 피처를 제거해도 성능이 오히려 좋아졌다. ExtraTrees는
AUCPR가 0.001076 낮아졌지만 게이트의 5% 상대 저하 한계에는 훨씬 못 미친다. 이는
시간 피처가 합성 데이터의 필수 근거가 아니라는 신호이지, 실제 생리적 인과성을
증명하는 결과는 아니다.

## 파생변수 임포턴스 감사

`derived_signal`은 autonomic arousal, motor activation, cognitive load, sleep pressure,
sensory context, recovery capacity, social context의 causal window 파생변수 묶음이다.

| 모델 | 평균 AUCPR drop | 양수 사람 비율 | 반복 양수 비율 | 게이트 |
|---|---:|---:|---:|---|
| LightGBM | 0.366647 | 1.00 | 1.00 | 통과 |
| XGBoost | 0.365198 | 1.00 | 1.00 | 통과 |
| ExtraTrees | 0.327759 | 1.00 | 1.00 | 통과 |

세 후보 모두 validation 6명 전원에서 파생변수 그룹을 섞으면 성능이 반복적으로
감소했다. 반면 `context`와 `time` 그룹은 사람별 반복 통과 조건을 충족하지 못했다.
따라서 단일 피처의 gain 순위보다 사람별 그룹 permutation과 시간 제거 결과를 앵커
판단의 중심에 둔다.

## 실제 데이터 shadow 결과

현재 Neon 관측 snapshot은 3,284행·11세션이지만 다음이 모두 비어 있다.

- corrected UTC 행: 0
- 독립 관찰 단계 라벨 행: 0
- 등록된 모델 버전 행: 0
- `model_ready`: `false`

따라서 실제 데이터에 예측을 실행하거나 정확도를 계산하지 않았다. 다음 단계는
corrected UTC, observer onset/peak/recovery/end 라벨, model-ready schema hash, 모델 버전을
등록한 뒤 locked shadow prediction을 다시 실행하는 것이다. 그 전까지 실제 성능은
`NOT VERIFIED`로 유지한다.

## 재현·산출물

노트북 생성과 제출:

```bash
UV_CACHE_DIR=/private/tmp/multisensor-uv-cache \
.venv/bin/python scripts/build_kaggle_tree_benchmark_notebook.py
kaggle kernels status bjcoding/multisensor-goal15-gpu-tree-benchmark-extratrees
```

검증된 보고서와 그래프:

- [한글 GPU 비교 HTML](../reports/tree_model_gpu_kaggle_ko.html)
- [canonical artifact JSON](../reports/tree_model_gpu_kaggle_ko.artifact.json)
- [지표 그래프](../reports/tree_model_gpu_kaggle_ko_metrics.png)
- [피처 중요도 그래프](../reports/tree_model_gpu_kaggle_ko_importance.png)
- [안정성 그래프](../reports/tree_model_gpu_kaggle_ko_stability.png)
- [시간 제거 그래프](../reports/tree_model_gpu_kaggle_ko_ablation.png)
- [permutation 그래프](../reports/tree_model_gpu_kaggle_ko_permutation.png)
- [실제 데이터 shadow 차단 artifact](../reports/tree_model_real_shadow_ko.artifact.json)

합성 결과의 `LightGBM`은 모델 후보로만 보존한다. 실제 운영 승격은 여러 날의 실제
관찰 라벨, 장치 동기화·결측 품질, 개인 기준선 안정성, false alerts/hour·ECE guardrail을
확인한 뒤 수동 audit reason과 함께 결정한다.
