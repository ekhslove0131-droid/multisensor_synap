# 트리 모델 비교·피처 중요도 감사 보고서

## 결론

이번 단계의 핵심은 “피처 중요도가 큰 모델”을 고르는 것이 아니라, 우리가 만든
생리·상황 파생변수가 **사람이 바뀌어도 반복해서 실제 예측에 기여하는지**를 확인하는
것이다. 따라서 피처 중요도는 필요한 증거이지만 단독 채택 기준으로 사용하지 않는다.
다음 세 조건을 모두 통과한 후보만 앵커 후보가 된다.

1. 시간 주기 피처를 제거해도 AUCPR 상대 저하가 5% 이하이고 사건 recall 절대 저하가
   5%p 이하일 것.
2. `derived_signal` 그룹을 사람별로 섞었을 때 AUCPR가 평균적으로 양수만큼 감소할 것.
3. 6명 각각에서 양의 permutation 감소가 반복되고, 양수 사람 비율이 80% 이상이며
   3회 반복 중 양수 비율 평균이 67% 이상일 것.

이번 `mvp3-oracle-v1`에서는 XGBoost와 LightGBM이 모두 위 감사 게이트를 통과했다.
게이트 통과 후보 중 seed 간 중요도 분산이 더 작은 LightGBM을 `challenger anchor`로
기록했다. 이것은 **합성 데이터에서의 앵커 후보 선정**이며 운영 승격이나 실제 행동
예측 정확도 승인을 뜻하지 않는다.

## 왜 중요도만으로 고르지 않는가

강한 permutation 중요도는 해당 그룹을 섞으면 예측 성능이 떨어진다는 뜻이므로,
파생변수가 모델의 판단에 실제로 사용되었다는 좋은 신호다. 그러나 다음 경우에는
중요도가 커도 잘못된 앵커가 될 수 있다.

- 시간·세션 ID·생성 스케줄처럼 라벨을 우회하는 누출 또는 합성 데이터 전용 규칙
- 여러 상관 피처 중 하나에 중요도가 몰리는 현상
- 한 사람에서만 강하고 다른 사람에게 재현되지 않는 개인 특화 우연
- AUCPR는 높지만 사건 recall, 오탐률, Brier/ECE가 나쁜 모델

그래서 이번 구현은 전체 피처 중요도 그래프, 사람 단위 permutation, 시간 피처 제거
ablation, AUCPR·event recall·F1·false alerts/hour·Brier·ECE를 함께 저장한다.
개별 파생변수 하나의 중요도보다 `derived_signal` 그룹을 사람별로 반복 측정한 결과를
앵커 판단의 중심으로 삼았다.

## 실행 계약

- 데이터: `mvp3-oracle-v1`, 상태 `oracle/sanity`
- 사람 단위 train/validation 분할, train 최대 600,000행·validation 최대 300,000행
- 도전자: XGBoost, LightGBM; 각 3개 결정적 seed, 160개 트리, `n_jobs=1`
- 앙상블: 두 도전자의 validation 확률 단순 평균
- 도전자 비교 임계값: 0.5; locked test는 읽지 않음
- ExtraTrees: 실행하지 않고 마지막 후보로 보류
- 실제 센서·실제 행동 정확도: `NOT VERIFIED`
- 그래프: 코드에서 설치된 `NanumGothic-Regular.ttf`를 등록해 한글로 생성

데이터를 사람 단위로 cap할 때 validation 6명이 모두 남도록 사람별 상한을 먼저
분배했다. permutation은 각 사람의 validation 행 안에서만 섞어 person split을
보존한다. 따라서 다른 사람의 분포를 빌려 중요도를 부풀리지 않는다.

## 전체 모델 결과

| 모델 | AUCPR | 사건 recall | 사건 F1 | 시간당 오탐 | Brier | ECE | 평가 범위 |
|---|---:|---:|---:|---:|---:|---:|---|
| 기존 Logistic | 0.246723 | 1.000000 | 0.663462 | 0.097222 | 0.174620 | 0.365122 | 기존 전체 validation |
| 기존 HGB | 0.511067 | 0.623188 | 0.741379 | 0.005556 | 0.064288 | 0.183716 | 기존 전체 validation |
| XGBoost | 0.659047 | 0.496283 | 0.595736 | 13.042150 | 0.114248 | 0.104470 | bounded validation sample |
| LightGBM | 0.658956 | 0.498001 | 0.596174 | 13.258824 | 0.109654 | 0.085909 | bounded validation sample |
| 소프트 앙상블 | 0.661107 | 0.499593 | 0.600172 | 12.689359 | 0.111222 | 0.095475 | bounded validation sample |

기존 HGB와 도전자 행은 임계값·평가 범위가 다르므로 표의 수치만으로 운영 우열을
판정하지 않는다. 앙상블은 도전자 중 AUCPR와 F1이 가장 높고 오탐도 조금 낮지만,
이번 앵커 게이트의 대상은 개별 도전자 모델이며 앙상블은 보조 후보로 보존했다.

## 시간 피처 제거(ablation)

| 모델 | 전체 AUCPR | 시간 제거 AUCPR | 상대 저하 | 전체 recall | 시간 제거 recall | 절대 저하 |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.659047 | 0.668217 | 0.00% | 0.496283 | 0.512200 | 0.00% |
| LightGBM | 0.658956 | 0.667903 | 0.00% | 0.498001 | 0.511277 | 0.00% |

두 모델 모두 시간 피처를 없앴을 때 성능이 오히려 높아졌다. 즉 이번 합성 실행에서는
`time_cos` 같은 시간 shortcut에 의존하지 않아도 되며, 시간 피처가 성능의 필수 근거라는
증거가 없다. 그래프의 0%는 성능이 같다는 뜻이 아니라, “제거로 인한 저하가 없고
오히려 개선되어 저하율을 0으로 표시했다”는 뜻이다.

## 사람별 파생변수 그룹 permutation

`derived_signal`은 autonomic arousal, motor activation, cognitive load, sleep pressure,
sensory context, recovery capacity, social context의 파생변수와 causal window 통계를
묶은 그룹이다. 각 사람의 validation 행에서 이 그룹만 섞고 AUCPR 감소량을 계산했다.

| 모델 | 평균 AUCPR 감소 | 중앙값 | 양수 사람 비율 | 반복 양수 비율 | 유효 사람 수 | 게이트 |
|---|---:|---:|---:|---:|---:|---|
| XGBoost | 0.473604 | 0.468952 | 1.00 | 1.00 | 6 | 통과 |
| LightGBM | 0.473159 | 0.464917 | 1.00 | 1.00 | 6 | 통과 |

두 모델에서 6명 전원이 3회 반복 모두 양수였다. 이것은 “생리·상황 파생변수가
현재 합성 oracle 예측에 강하게 사용된다”는 근거다. 반면 시간·맥락 그룹은 양수
사람 비율이 게이트 기준에 못 미쳤으므로 앵커 근거로 승격하지 않았다.

## 피처 중요도와 안정성 해석

전체 중요도 그래프에서는 XGBoost의 상위권이 감각 맥락·자율 각성 등 파생변수였고,
LightGBM에서는 `time_cos`가 상위에 나타났다. 그러나 시간 제거 결과와 사람별
permutation을 함께 보면 최종 앵커를 시간 피처에 맡길 이유는 없다. seed 간 중요도
분산은 LightGBM이 XGBoost보다 작아, 게이트 통과 후 LightGBM을 선택하는 안정성
tie-break로 사용했다.

중요도는 인과적 원인이나 의료적 의미가 아니다. 실제 Galaxy Watch8 데이터에서는
장치 동기화, 결측·노이즈, 개인 기준선 보정, 라벨 품질을 먼저 확인한 뒤 동일 게이트를
다시 실행해야 한다.

## 재현 방법

로컬에서 동일한 비교와 감사 산출물을 재생성한다.

```bash
MPLCONFIGDIR=/private/tmp/multisensor-mpl-cache \
PYTHONPATH=src .venv/bin/python scripts/run_tree_model_benchmark.py \
  --project-root /Users/baital/dev/multisensor_ml \
  --series mvp3-oracle-v1 \
  --output reports/tree_model_benchmark_ko.html \
  --max-train-rows 600000 \
  --max-validation-rows 300000 \
  --seed-count 3 \
  --n-estimators 160
```

주요 산출물:

- [한글 HTML 보고서](../reports/tree_model_benchmark_ko.html)
- [감사 manifest JSON](../reports/tree_model_benchmark_ko.artifact.json)
- `tree_model_benchmark_ko.metrics.parquet`
- `tree_model_benchmark_ko.importance.parquet`
- `tree_model_benchmark_ko.predictions.parquet`
- `tree_model_benchmark_ko.ablation_metrics.parquet`
- `tree_model_benchmark_ko.permutation_detail.parquet`
- `tree_model_benchmark_ko.permutation_summary.parquet`
- `_metrics.png`, `_importance.png`, `_stability.png`, `_ablation.png`, `_permutation.png`

캐글 재현용 코드는 [09_tree_model_benchmark.ipynb](../kaggle/09_tree_model_benchmark.ipynb)에
있다. `RUN_LOCKED_TEST=False`, `USE_GPU=False`, `N_JOBS=1`이며, prepared manifest·새
wheel·NanumGothic 폰트를 입력 Dataset으로 연결해야 한다. 이번 변경에서는 캐글
커널을 새로 시작하거나 업로드하지 않았다.

## 다음 단계

1. 실제 데이터가 들어오기 전까지는 이번 게이트를 합성 데이터 내부 회귀 테스트로
   사용한다. synthetic 결과는 계속 `oracle/sanity`로 표시한다.
2. Galaxy Watch8 실측에서 센서별 파생변수 품질·결측·동기화 상태를 확인하고, 동일한
   사람 단위 permutation과 시간 제거 검사를 반복한다.
3. 도전자와 기존 HGB에 동일한 validation 임계값·calibration·오탐 상한을 적용한 뒤
   운영 비교를 한다. 그 전에는 LightGBM을 자동 운영 모델로 승격하지 않는다.
4. 실제 검토 완료 라벨이 여러 날 반복해서 쌓인 경우에만 개인 calibration과 재학습
   후보를 검토한다. 실제 정확도와 장치 동기화 상태는 그때까지 `NOT VERIFIED`다.
