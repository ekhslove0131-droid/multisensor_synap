# XGBoost·LightGBM 도전자 비교 보고서

## 결론

이번 단계에서는 ExtraTrees를 실행 목록에서 제외하고 XGBoost와 LightGBM을 같은
합성 train/validation 계약으로 비교했다. 세 개의 결정적 seed를 평균한 소프트
앙상블도 함께 기록했다. 도전자만 놓고 보면 LightGBM을 **challenger anchor**로
선정했다. AUCPR가 XGBoost와 거의 같으면서 피처 중요도 분산과 상위 피처 집중도가
더 안정적이기 때문이다.

다만 운영 모델을 LightGBM으로 자동 교체하지 않았다. 기존 HGB는 튜닝된 임계값과
전체 validation 범위로 평가되었고, 도전자는 bounded validation sample에서 고정
임계값 0.5로 평가되었다. 이 두 결과는 운영 승격을 결정할 만큼 동일한 조건이
아니므로 현재 운영 기준 모델은 기존 HGB로 남기고, 다음 단계에서 동일한 내부
validation 임계값·캘리브레이션 절차를 적용한 뒤 비교한다.

## 실행 계약

- 데이터: `mvp3-oracle-v1`, `oracle/sanity` 합성 데이터
- train/validation: 사람 단위 분할을 유지하며 각각 최대 600,000/300,000행
- 모델: XGBoost와 LightGBM, 각 3 seed, 160개 트리, 학습 작업 수 1
- 앙상블: 두 도전자의 validation 확률 단순 평균
- 임계값: 도전자 비교에서는 0.5 고정; locked test는 읽지 않음
- ExtraTrees: 실행하지 않고 마지막 후보로 보류
- 실제 센서 정확도: `NOT VERIFIED`
- 그래프 폰트: 설치된 `NanumGothic-Regular.ttf`를 코드로 등록

그래프는 `scripts/run_tree_model_benchmark.py`가 직접 생성한다. macOS에서 GUI
백엔드가 중단되는 문제를 피하기 위해 `matplotlib`를 `Agg`로 고정했고, 결과에는
지표·피처 중요도·seed 간 안정성 그래프를 함께 저장한다.

## 지표 결과

| 모델 | AUCPR | 사건 recall | 사건 F1 | 시간당 오탐 | Brier | ECE | 평가 범위 |
|---|---:|---:|---:|---:|---:|---:|---|
| 기존 Logistic | 0.246723 | 1.000000 | 0.663462 | 0.097222 | 0.174620 | 0.365122 | 기존 전체 validation |
| 기존 HGB | 0.511067 | 0.623188 | 0.741379 | 0.005556 | 0.064288 | 0.183716 | 기존 전체 validation |
| XGBoost | 0.652711 | 0.488922 | 0.585615 | 13.888261 | 0.114248 | 0.109882 | bounded validation sample |
| LightGBM | 0.659188 | 0.502704 | 0.596771 | 13.979933 | 0.109654 | 0.092973 | bounded validation sample |
| 소프트 앙상블 | 0.658315 | 0.497459 | 0.594681 | 13.482683 | 0.111222 | 0.101359 | bounded validation sample |

도전자 범위에서는 LightGBM이 AUCPR와 ECE가 가장 좋았고, 앙상블은 LightGBM보다
AUCPR가 소폭 낮지만 시간당 오탐은 약간 낮았다. 따라서 앙상블은 보조 후보로
보존하고, 현재의 앵커는 LightGBM으로 기록했다. 기존 HGB의 시간당 오탐이 매우
낮게 나온 것은 평가 범위와 임계값이 다르기 때문이며, 이 표만으로 어느 모델이
실제 운영에 우월하다고 결론내리지 않는다.

## 피처 중요도 안정성

| 도전자 | 대표 상위 피처 | 상위 3개 집중도 | 중요도 평균 분산 | 최대 분산 |
|---|---|---:|---:|---:|
| XGBoost | `autonomic_arousal__mean_300s` | 0.090057 | 0.000001712 | 0.000054075 |
| LightGBM | `time_cos` | 0.140417 | 0.000000584 | 0.000003733 |

LightGBM은 seed에 따른 중요도 분산이 더 작아 앵커 후보로 선택되었다. 다만
`time_cos`가 가장 큰 피처로 나타난 것은 합성 데이터의 시간 주기 구조를 반영한
것일 수 있으므로, 실제 Galaxy Watch8 데이터에서는 장치별 시간축 보정과 누락·노이즈
스트레스를 다시 거쳐야 한다. 피처 중요도는 인과적 원인이나 의료적 의미로 해석하지
않는다.

## 재현 방법

로컬에서는 다음 명령으로 동일한 bounded 비교를 다시 실행한다.

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

다음 파일이 함께 생성된다.

- `reports/tree_model_benchmark_ko.html`
- `reports/tree_model_benchmark_ko.artifact.json`
- `reports/tree_model_benchmark_ko.metrics.parquet`
- `reports/tree_model_benchmark_ko.importance.parquet`
- `reports/tree_model_benchmark_ko.predictions.parquet`
- `_metrics.png`, `_importance.png`, `_stability.png`

캐글용 코드는 `kaggle/09_tree_model_benchmark.ipynb`와
`kaggle/09_tree_model_benchmark.kernel-metadata.json`에 남겼다. 실행 설정은
`RUN_LOCKED_TEST=False`, `USE_GPU=False`, `N_JOBS=1`이며, prepared manifest와
새 wheel, NanumGothic 폰트 Dataset을 입력으로 연결해야 한다. 이번 변경에서는
캐글 커널을 새로 시작하거나 업로드하지 않았다.

## 다음 단계

1. XGBoost·LightGBM·앙상블 모두에 동일한 사람 단위 내부 validation으로 임계값과
   calibration을 다시 선택한다.
2. 시간당 오탐 상한, event recall, AUCPR, Brier/ECE를 함께 만족하는지 확인하고,
   후보 한 개만 locked test를 읽는다.
3. 개인 기준선과 누적 스트레스 파생변수를 추가한 뒤, 단일·2축·3축 센서 가용성별로
   같은 평가를 반복한다.
4. Galaxy Watch8 실측 데이터가 충분히 쌓인 뒤에만 `NOT VERIFIED` 경계를 해제하고,
   자동 승격 없이 검토 가능한 release로 등록한다.
