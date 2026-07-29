# 시계열 머신러닝 중간보고서 설계

## 목적

현재 Kaggle validation 결과를 근거로 Goal 1.5에서 어떤 머신러닝
아키텍처에 집중할지 결정하고, 다음 실험을 단계별로 고정한다. 독자는
기획자이자 개발 책임자인 사용자이며, 결과는 한 페이지 한국어 HTML로
전달한다.

## 핵심 결론

- 현 champion인 `HistGradientBoosting`을 주 기준 모델로 유지한다.
- 시계열성은 딥러닝에 맡기지 않고 causal lag, rolling 통계, slope,
  기준선 편차와 상태 전이 decoder로 표현한다.
- `LightGBM`을 다음 challenger로 추가하고, Logistic Regression은
  설명·calibration·회귀 감지용 기준 모델로 유지한다.
- 패턴 모델, 5단계 모델, 행동별 one-vs-rest 모델을 분리한다.
- 현재 이벤트 지표의 모순과 후보별 metrics 미보존을 먼저 바로잡은 뒤
  모델을 확장한다.

## 근거 범위

- 데이터 상태: `oracle/sanity`
- 학습/검증 분리: 사람 기준 train 24명, validation 6명
- locked test: 미사용
- 실제 데이터 정확도: `NOT VERIFIED`
- champion: `HistGradientBoosting`
- 패턴: AUCPR 0.494, AUROC 0.810, row F1 0.050, ECE 0.207
- 5단계 macro: AUCPR 0.639, AUROC 0.866, F1 0.603, ECE 0.060
- 행동 10종 macro: AUCPR 0.399, AUROC 0.541, F1 0.379, ECE 0.179

패턴의 event recall 1.0과 false alerts/hour 0은 낮은 row F1과 함께
나왔으므로 평가 구현을 감사하기 전에는 의사결정 지표로 사용하지 않는다.

## 권장 모델 구조

1. **기준선 계층**
   전역 context별 median/MAD와 causal 개인 기준선을 계산하고
   robust-z 및 개인 기준선 편차를 출력한다.
2. **패턴 발생 head**
   HistGradientBoosting으로 `NO_EVENT` 대비 event probability를
   계산한다. LightGBM은 동일 split·feature·metric으로만 비교한다.
3. **5단계 head**
   event 후보 구간에 대해 LOW, MEDIUM, HIGH, DECREASING, RECOVERY
   one-vs-rest 확률을 계산한다.
4. **상태 decoder**
   허용된 단계 전이, 최소 지속시간, hysteresis를 적용한다.
   품질 실패와 OOD에서는 `NOT_DECISIONABLE`로 abstain한다.
5. **행동 head**
   행동 코드별 독립 one-vs-rest 모델을 사용한다. 모델 1의 확률은
   person-grouped OOF prediction으로만 학습 입력에 넣는다.
6. **calibration 계층**
   validation에서 isotonic과 sigmoid calibration을 비교하되
   person split을 유지하고 ECE·Brier가 개선될 때만 채택한다.

## 비교할 모델

| 모델 | 역할 | 장점 | 제한 |
|---|---|---|---|
| HistGradientBoosting | 현재 champion | 비선형성, 결측 대응, 현재 의존성에 포함 | 긴 시계열 순서를 직접 기억하지 않음 |
| LightGBM | 다음 challenger | 대규모 표형 데이터 속도와 성능 | 새 의존성, calibration 별도 검증 |
| Logistic Regression | 기준선·calibration | 설명 가능, 안정적 회귀 감지 | 복잡한 상호작용 표현이 약함 |
| Random Forest/Extra Trees | 진단 후보 | feature importance와 비선형 비교 | 메모리·추론비용 대비 우선순위 낮음 |

CatBoost는 현재 context가 이미 수치형 one-hot으로 변환되어 있어
LightGBM 이후의 보조 후보로만 둔다.

## HTML 구성

한 페이지 단일 열 구조로 다음 순서에 따른다.

1. 결론과 현재 판정
2. 현재 validation 핵심 지표
3. 왜 HistGradientBoosting + decoder인가
4. 모델 후보 비교표
5. 권장 계층형 아키텍처
6. 단계별 실행 계획과 승인 기준
7. 제한·금지 해석·추가 질문

지표 비교는 접근 가능한 HTML/CSS bar chart와 정확한 수치 표를 함께
제공한다. 모바일과 데스크톱에서 가로 스크롤 없이 읽히며 인쇄 시 한
문서로 이어지게 한다.

## 다음 실행 계획

### 1단계: 평가 신뢰성 복구

- event-run 집계에 대한 경계·false alert 회귀 테스트
- threshold, candidate metrics, 사람별 지표 저장
- Logistic/HGB 후보별 동일 validation 비교

승인 기준은 row 지표와 event 지표가 동일 예측으로 재계산되고, 임계값과
분모를 추적할 수 있는 것이다.

### 2단계: 시계열 피처 안정화

- lag 1/5/15/30/60초
- rolling 5/15/30/60/180/300초
- slope, 변화량, baseline deviation
- gap reset, session boundary, causal property 검증

승인 기준은 미래 참조 0건, 사람·session 경계 누출 0건이다.

### 3단계: 모델 비교

- Logistic, HistGradientBoosting, LightGBM
- 동일 train/validation, 동일 threshold 정책
- AUCPR, AUROC, macro F1, event recall, false alerts/hour, ECE, Brier,
  추론시간, bundle 크기 비교

LightGBM은 반복 validation 개선이 명확하고 calibration 또는 false
alert가 악화되지 않을 때만 champion 후보가 된다.

### 4단계: 상태 decoder

- 허용 전이, hysteresis, 최소 지속시간
- OOD·결측에서 abstention
- raw probability와 decoded state를 모두 감사 저장

### 5단계: 실제 데이터 확인

- 실제 데이터는 champion 선택에 사용하지 않는다.
- 장비 동기화·품질·OOD 상태와 함께 확인용 기록만 남긴다.
- 검토 완료 라벨이 여러 날 누적된 뒤 별도 재학습 후보를 만든다.

### 6단계: locked test

validation으로 하나의 composite release를 고른 후 한 번만 실행한다.
locked test 결과로 threshold나 모델을 수정하지 않는다.

## 검증

- HTML 내 수치가 Kaggle champion metrics와 일치해야 한다.
- 결과와 계획에서 `oracle/sanity`, `NOT VERIFIED`, locked test 미사용을
  분명히 표시한다.
- LightGBM이 아직 실행되지 않았음을 명시한다.
- 패턴 event metric 모순을 숨기거나 PASS로 표현하지 않는다.
- 링크, 표, bar chart, 인쇄 레이아웃과 작은 화면 가독성을 확인한다.

