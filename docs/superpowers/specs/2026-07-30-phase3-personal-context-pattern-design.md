# 3차 개인·환경 기준선 패턴 감지 검증 설계

## 목적

3차의 주 목표는 행동 분류가 아니라 개인 기준선 보정 이후 사건 관련 패턴을
얼마나 정확하게 감지하는지 검증하는 것이다. 행동 다중 라벨은 주 모델 선택에
사용하지 않고 보조 감사 지표로만 유지한다.

이번 비교는 합성 latent truth를 허용 목록에 따라 피처로 변환하는
`oracle/sanity` 실험이다. 실제 웨어러블 정확도와 행동·의료 성능은
`NOT VERIFIED`로 유지한다.

## 검증 질문

1. 환경별 전역 기준선보다 개인 기준선 보정이 새로운 사람의 패턴 감지를
   개선하는가?
2. 개인의 모든 환경을 섞은 기준선보다 개인×환경 기준선이 더 안정적인가?
3. 개인 가중치를 높일수록 계속 좋아지는가, 아니면 과적응 구간이 존재하는가?
4. 평균 성능 상승이 특정 seed, 사람 또는 물리 dataset_id의 큰 개선에만
   의존하지 않는가?
5. 개인화로 event recall이 오르더라도 false alert와 calibration이 함께
   악화되지는 않는가?

## 통제 비교

모든 버전은 같은 합성 원천, 사람 분할, 라벨, 학습 표본, 모델 하이퍼파라미터,
OOF fold와 평가 코드를 사용한다. 한 버전에서 바뀌는 변수는 기준선 정책뿐이다.

| 버전 | 기준선 정책 | 개인 가중치 상한 |
|---|---|---:|
| `G0` | 환경별 전역 median/MAD | `0.00` |
| `P1-025` | 현재 개인 통합 기준선 | `0.25` |
| `P1-050` | 현재 개인 통합 기준선 | `0.50` |
| `P1-100` | 현재 개인 통합 기준선 | `1.00` |
| `P2-025` | 개인×환경 과거 기준선 | `0.25` |
| `P2-050` | 개인×환경 과거 기준선 | `0.50` |
| `P2-100` | 개인×환경 과거 기준선 | `1.00` |

전역 기준선은 train person만으로 context별 median/MAD를 계산한다. 개인 기준선은
각 사람의 과거값만 사용한다. `P2`는 같은 context의 과거값만 사용하고 해당
context의 이력이 부족하면 `G0`으로 축소한다.

개인화 계약은 다음과 같다.

- warm-up: 사람×context별 유효 과거 관측 `1,800`초
- lookback: 최근 `21,600`초
- refresh: `60`초
- 가중치:
  `min(weight_cap, n_eff / (n_eff + 1800) × quality_confidence)`
- Oracle 품질값: `quality_confidence=1.0`
- median/MAD 계산에는 현재 행과 미래 행을 포함하지 않는다.
- person, run, day, session 또는 1초 초과 시간 공백에서 시계열 피처 이력을
  초기화한다.

## 데이터 흐름과 누출 방지

3차 피처 준비는 로컬 원천의 `latent_timeline.parquet`을 읽되, 다음 열만
허용한다.

- 7개 latent factor
- `is_awake`
- `context`
- UTC 시간으로부터 만든 주기 피처

라벨은 기존 outcome 테이블에서 독립적으로 결합한다. event truth, phase,
event intensity, hidden archetype, participant truth baseline과 hard-negative
식별자는 모델 입력에서 제외한다.

각 기준선 버전은 별도 namespace의 robust-z, lag, delta, rolling
mean/std/slope를 생성한다. 기준선 간 피처를 한 모델에 동시에 넣지 않는다.
각 산출물은 source hash, split hash, baseline policy hash와 feature schema hash를
기록한다.

## 모델과 선택 절차

주 타깃은 `pattern_binary` 하나다.

- `HistGradientBoosting`: 주 후보
- `Logistic Regression`: 설명 가능성과 회귀 감지용 기준 후보
- 행동 head와 5단계 head는 3차 champion 선택에 사용하지 않는다.
- LightGBM과 TCN은 이번 비교가 끝난 뒤의 별도 실험이다.

임계값과 개인 가중치 상한은 train person-grouped OOF prediction으로만 선택한다.
validation은 선택된 전역 기준선 후보와 개인 기준선 후보를 한 번 비교하는
미사용 평가 집합으로 유지한다. locked test는 열지 않는다.

`P1`과 `P2` 각각에서 OOF 기본 합격 조건을 통과한 가장 작은 weight cap을
선택한다. 상한 증가가 성능을 더 높이더라도 안정성 조건을 만족하지 못하면
더 작은 상한으로 복귀한다.

## 평가 지표

주 지표:

- AUCPR
- event F1
- event recall
- false alerts/hour
- person-macro F1
- person-macro AUCPR
- 하위 25% person recall

보조 지표:

- AUROC
- row F1
- Brier score
- ECE
- context별 AUCPR/event F1
- warm-up 포함 전체와 mature-personal 구간의 차이
- 행동 라벨별 기존 보조 성능

`accuracy`는 불균형 데이터에서 오해를 부를 수 있으므로 보조 표에만 기록하고
모델 선택에는 사용하지 않는다.

## 데이터셋별 상승 분석

전역 기준선 `G0` 대비 선택된 개인 기준선 모델의 변화량을 다음 수준으로
계산한다.

1. split role 전체
2. seed/run 3개
3. 물리 `dataset_id` 36개
4. validation person 6명
5. context

각 그룹은 `G0`, personal, absolute delta와 relative delta를 함께 기록한다.
분모가 0인 relative delta는 `NOT_COMPUTABLE`로 남긴다.

보고서는 평균 상승 외에 다음 항목을 반드시 포함한다.

- 개선/동률/악화 그룹 수
- 중앙값과 사분위 범위
- 가장 크게 개선된 그룹
- 가장 크게 악화된 그룹
- support가 부족해 판단 보류된 그룹

train 지표는 적합도와 OOF 선택 감사용이고, “정확도 상승” 결론은 validation
그룹에서만 낸다. locked-test 그룹은 실행 전까지 `NOT_EVALUATED`다.

## 채택 규칙

개인 기준선 모델은 validation에서 다음을 모두 만족할 때만 `recommended`다.

1. person-macro AUCPR와 event F1이 `G0`보다 높다.
2. 하위 25% person recall이 낮아지지 않는다.
3. false alerts/hour와 ECE가 각각 상대적으로 `5%` 넘게 악화되지 않는다.
4. validation person의 과반에서 AUCPR 또는 event F1이 개선된다.
5. 한 seed/run의 개선만으로 전체 평균 개선이 설명되지 않는다.

조건을 통과하지 못하면 `G0 유지`, `개인 가중치 축소` 또는
`INSUFFICIENT_EVIDENCE`로 결론 내린다. 자동 승격과 자동 재학습은 하지 않는다.

## 산출물

- `phase3_candidate_metrics.parquet`
- `phase3_group_uplift.parquet`
- `phase3_person_predictions.parquet`
- `phase3_baseline_diagnostics.parquet`
- `phase3_manifest.json`
- `reports/goal15_phase3_personal_pattern_ko.html`
- 재현 가능한 Kaggle 노트북과 실행 metadata

한글 HTML 보고서는 1·2·3차 흐름, 기준선 버전 비교, 데이터셋별 상승·악화,
사람별 분포, context별 결과, 채택 여부와 실제 데이터 한계를 한 페이지에
표시한다.

## 승인 경계

- 실제 데이터·의료 성능으로 표현하지 않는다.
- behavior 성능으로 개인 기준선 후보를 선택하지 않는다.
- validation을 본 뒤 weight cap이나 threshold를 다시 조정하지 않는다.
- locked test를 실행하지 않는다.
- 실제 장비 동기화와 물리적 노이즈 보정은 이번 범위에 포함하지 않는다.
