# 1차 합성데이터·시계열 머신러닝 중간보고서 설계

## 목적

합성데이터 생성부터 KNIME 연결과 Kaggle 머신러닝 validation까지의 1차
작업을 시간순으로 설명한다. 성공 결과만 나열하지 않고, 처음 어떤 구조를
선택했는지, 실제로 무엇이 잘되지 않았는지, 그때 어떤 수정을 했는지와 아직
해결하지 못한 문제를 함께 기록한다. 마지막에는 1차에서 배운 내용을 근거로
2차 실행계획을 고정한다.

독자는 기획자이자 개발 책임자인 사용자이며, 결과는 한 페이지 한국어
HTML로 전달한다.

## 서술 원칙

- 보고서 제목과 본문에서 이번 작업을 `1차`로 표시한다.
- 결과보다 문제 정의, 선택, 실패, 수정, 배운 점의 흐름을 우선한다.
- 실패한 Kaggle 버전과 지표 모순을 숨기지 않는다.
- 합성 truth·latent와 실제 센서 데이터를 명확히 분리한다.
- 로컬 계층형 MVP locked test와 Kaggle validation-only 실행을 혼합하지 않는다.
- 2차 계획은 일반적인 권고가 아니라 1차 실패와 직접 연결한다.

## 1차에서 처음 하려던 것

처음 목표는 세 장비의 실제 신호 모델을 만드는 것이 아니라, 기존
`multisensor_synth`가 만든 truth·latent 합성데이터를 이용해 다음 구조가
작동하는지 확인하는 것이었다.

1. train person으로 전역 기준선을 만든다.
2. 개인의 과거 데이터로 기준선을 보정한다.
3. 기준선 편차와 causal 파생변수로 사건을 감지한다.
4. 사건을 LOW, MEDIUM, HIGH, DECREASING, RECOVERY 단계로 구분한다.
5. 관찰 행동 라벨을 이용해 행동별 확률을 예측한다.
6. 결과와 버전을 KNIME에서 확인한다.

## 1차에서 실제로 한 작업

### Oracle 기본환경

- `multisensor_ml`을 별도 저장소로 만들고 생성기 저장소를 읽기 전용
  dependency로 연결했다.
- seed 3개, 36명, 5일 데이터를 사람 기준 24/6/6으로 분리했다.
- causal baseline, rolling feature, Logistic/HGB, stress harness와 안전한
  `.skops` bundle을 만들었다.
- 기존 KNIME workflow에서는 Python이 실행한 결과를 읽어 보여주는
  Oracle Benchmark를 먼저 구성했다.

### 합성데이터 공장과 라벨

- 사건, 5단계 timeline, 행동 10종 다중 라벨을 결정적으로 생성하는
  Synthetic Factory를 만들었다.
- outcome 사건 930건, 행동 양성 라벨 2,416건, 1Hz 단계 timeline
  15,552,000행을 생성했다.
- hard negative와 일반 구간에도 유사 행동이 존재하도록 만들어
  target과 행동 코드가 1:1로 고정되지 않게 했다.

### 계층형 모델과 버전관리

- 기준선, STD 타입, OOD, 사건/단계 모델, 행동 모델과 한국어 결과
  router를 분리했다.
- SQLite 불변 레지스트리와 `candidate → champion → retired` 상태를
  만들었다.
- 로컬 계층형 MVP에서는 선택된 composite release의 locked test를 한 번
  실행했지만, 이후 Kaggle v5에서는 평가 문제를 다시 확인하기 위해
  validation까지만 실행했다.
- KNIME에 Synthetic Factory와 Training Registry workflow를 새로 만들고
  receipt를 실제로 연결했다.

### Kaggle 이전과 실행

- 약 10.23GB, 47개 파일의 비공개 Kaggle Dataset을 게시했다.
- ML/DL 데이터와 benchmark 노트북 네 개를 비공개로 올렸다.
- GPU 중복 사용을 막기 위해 ML benchmark 하나만 CPU로 실행했다.
- 최종 Kaggle ML v5는 약 41분 35초 후 완료됐고 HGB가 champion으로
  선택됐다.

## 1차 시행착오

| 시도 | 잘되지 않은 점 | 실제 대응 | 남은 교훈 |
|---|---|---|---|
| 최초 KNIME Oracle 화면 | 화면은 보였지만 Python 결과를 라우팅할 뿐 학습 단계와 모델 버전이 KNIME에서 연결되어 보이지 않았다 | Synthetic Factory와 Training Registry를 별도 workflow로 만들고 receipt output/input을 실제 연결했다 | KNIME은 학습 코드를 복제하는 곳이 아니라 실행·검토·승격 인터페이스로 사용해야 한다 |
| STD 타입 자동 탐색 | GMM K=2..6이 안정성과 최소 인원 조건을 통과하지 못했다 | 근거 없는 타입을 강제하지 않고 K=1 `STD-A`로 복귀했다 | 합성 인원 36명만으로 다양한 표준 타입이 존재한다고 결론 내릴 수 없다 |
| Oracle 사건·행동 모델 | 사건·단계보다 행동 라벨의 AUCPR이 낮고 사람별 편차가 컸다 | 행동별 one-vs-rest와 모델 1 OOF 확률을 사용하고 support 부족 시 승격하지 않게 했다 | 행동 라벨은 사건 라벨보다 더 많은 사건 수와 관찰 일관성이 필요하다 |
| noise stress | Gaussian 0.20에서 사건 AUCPR이 약 90% 감소했고 axis dropout에서는 event recall이 0까지 하락했다 | stress 결과를 release에 보존하고 실제 장비 성능과 분리해 `oracle/sanity`로 잠갔다 | 현재 모델은 실제 웨어러블 노이즈에 강하다는 근거가 없다 |
| Kaggle ML data v2 | validation Parquet의 `event_id`에 숫자와 문자열이 섞여 Arrow 변환이 실패했다 | 모든 split에서 audit ID를 입력 피처·ML view에서 제거하는 회귀 테스트를 추가했다 | train만 검사하면 validation/test 계약 오류를 놓친다 |
| Kaggle ML data v3 | ML benchmark가 `dataset_id`, `day_key`, `session_id` 부재로 중단됐다 | 물리 파일 단위 dataset/session identity와 UTC day를 모든 role에 결합했다 | 시간 평가는 person뿐 아니라 dataset·day·session 경계가 필요하다 |
| Kaggle ML benchmark v3 | `phase`가 feature contract의 미등록 열로 거부됐다 | phase는 audit용으로 허용하되 feature 선택에서는 제외했다 | 라벨 감사 열과 모델 입력 열을 별도 계약으로 관리해야 한다 |
| Kaggle ML benchmark v4 | objective event onset의 행동 라벨이 pattern/hard-negative mask 밖에 있어 학습이 중단됐다 | `event_binary` onset을 행동 결정 구간에 포함하고 일반 baseline positive는 계속 거부했다 | 사건 onset과 pre-event pattern은 같은 타깃이 아니며 둘을 섞지 않아야 한다 |
| Kaggle ML benchmark v5 | 완료됐지만 champion 지표만 저장되어 Logistic과 HGB의 수치 차이가 사라졌다 | HGB champion 결과와 hash는 보존했지만 후보별 비교는 복구하지 못했다 | 다음 실행은 후보별 metrics·threshold·runtime을 불변 산출물로 남겨야 한다 |
| event 평가 결과 | event recall 1.0, false alerts/hour 0인데 row F1은 0.05로 서로 어울리지 않았다 | PASS로 표현하지 않고 평가 감사 전 사용 금지로 표시했다 | 모델 개선 전에 event-run 집계와 threshold 정의부터 검증해야 한다 |
| 딥러닝 준비 | 데이터와 TCN 노트북은 준비했지만 현재 라벨·평가 문제를 해결하지 않은 채 GPU를 쓰면 비교가 무의미했다 | DL 노트북은 비활성 상태로 두고 ML 한 건만 실행했다 | 2차는 머신러닝 평가와 시계열 피처에 집중한다 |

## 1차의 잠정 결론

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
- Kaggle v5 locked test: 미사용
- 이전 로컬 계층형 MVP locked test: 1회 실행
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

1. 1차에서 해결하려던 문제
2. 합성데이터와 라벨을 어떻게 만들었는가
3. 실제 작업의 시간순 흐름
4. 잘되지 않았던 시도와 수정
5. 현재까지 알게 된 것과 알 수 없는 것
6. 왜 HistGradientBoosting + decoder에 집중하는가
7. 2차 단계별 실행계획과 승인 기준
8. 제한·금지 해석·추가 질문

지표 비교는 접근 가능한 HTML/CSS bar chart와 정확한 수치 표를 함께
제공한다. 모바일과 데스크톱에서 가로 스크롤 없이 읽히며 인쇄 시 한
문서로 이어지게 한다.

## 2차 실행계획

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
- 결과와 계획에서 `oracle/sanity`, `NOT VERIFIED`, Kaggle v5 locked test
  미사용과 이전 로컬 실행을 구분해 표시한다.
- LightGBM이 아직 실행되지 않았음을 명시한다.
- 패턴 event metric 모순을 숨기거나 PASS로 표현하지 않는다.
- 링크, 표, bar chart, 인쇄 레이아웃과 작은 화면 가독성을 확인한다.
