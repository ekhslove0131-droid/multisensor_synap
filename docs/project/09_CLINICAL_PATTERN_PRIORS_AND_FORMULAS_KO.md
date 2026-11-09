# 임상 연구 기반 선행 패턴 prior·파생변수·라벨 계산 계약

문서 버전: `pattern-priors/v1`  
작성 목적: 합성 패턴 시나리오와 이후 관측 데이터 모델의 공통 계산 계약 정의

## 1. 문서의 위치와 한계

이 문서는 멜트다운, 턴트럼, 감각추구, 상동행동과 같은 고수준 행동을
센서만으로 진단하기 위한 문서가 아니다. 연구에서 관찰된 생리·운동·상황의
선행 축을 합성데이터의 **약한 사전정보(synthetic prior)** 로 사용하고, 실제
데이터에서는 관찰자 라벨과 전문가 검토를 통해 검증하기 위한 계약이다.

연구마다 대상·연령·센서·행동 정의·분할 방법이 다르므로 논문의 효과크기나
임계값을 그대로 복사하지 않는다. 이 문서의 부호와 변수군은 시나리오 생성과
초기 가중치 후보에만 사용하며, 최종 가중치는 사람 단위로 분리된 train에서
학습하고 validation에서 선택한다.

## 2. 데이터 계층

```text
truth/
  숨겨진 사건 시작·끝·phase·강도·합성 pattern prior
observed/
  장비별 관측값·노이즈·결측·clock 상태
model_ready/
  관측값에서 causal하게 다시 계산한 파생변수·품질·라벨
```

`truth/`의 pattern 이름, 강도, onset, event id는 일반 모델 입력에 들어가지
않는다. 합성 정답을 이용한 검증용 테이블과 모델 입력 테이블은 별도 hash와
`source_domain`으로 등록한다.

## 3. 표기와 공통 파생변수

사람 `p`, context `c`, 시각 `t`, 신호·피처 `j`를 사용한다.

### 3.1 개인 기준선 대비 편차

과거의 유효한 정상 구간으로만 context별 중심과 변동폭을 계산한다.

```text
center[p,c,j] = median(x[p,c,j] 과거값)
scale[p,c,j]  = 1.4826 × MAD(x[p,c,j] 과거값) + ε
z[p,t,j]      = (x[p,t,j] - center[p,c,j]) / scale[p,c,j]
```

`z`는 `[-20, 20]`으로 제한하고, warm-up 전·품질 실패·장시간 결측에서는
`feature_valid=false`와 `baseline_confidence`를 함께 기록한다. 이벤트 및
회복 구간은 기본 기준선 갱신에서 제외한다.

### 3.2 인과 시간 피처

```text
mean_W(t)  = mean(z[t-W+1:t])
std_W(t)   = std(z[t-W+1:t])
slope_W(t) = (mean(z[t-W+1:t]) - mean(z[t-2W+1:t-W])) / W
delta(t)   = z(t) - z(t-1)
```

기본 `W`는 `5, 15, 30, 60, 180, 300`초다. 미래값을 사용하지 않는다.

### 3.3 표준 예측 잔차

일반 표준 모델이 과거 관측으로 예측한 값과 실제 관측값을 분리한다.

```text
ŷ[m,t+h]        = M0(과거 관측 ≤ t, context, 개인 adapter)
residual[m,t+h] = (x[m,t+h] - ŷ[m,t+h]) / (σ̂[m,t+h] + ε)
```

패턴 모델 입력에는 현재 시각까지 계산 가능한 `forecast_residual`과 예측
불확실성만 넣는다. 미래 실제값과 future label은 타깃으로만 사용한다.

### 3.4 방향이 있는 편차

생리 지표의 방향은 사람·상황·신호에 따라 다를 수 있으므로, 임상 문헌의
부호를 전역 임계값으로 고정하지 않는다.

```text
d[j,t] = direction[k,j] × z[j,t]
direction[k,j] ∈ {-1, +1}
```

`direction`은 pattern prior의 초기값일 뿐이며 train 데이터에서 재학습하거나
0으로 축소할 수 있다. 예를 들어 HRV 감소 후보는 `direction=-1`일 수 있지만,
모든 사람에게 같은 방향이라고 가정하지 않는다.

### 3.5 순간·누적 부하

품질 가중치를 포함한 순간 부하의 후보식은 다음과 같다.

```text
load(t) = Σ_j q[j,t] × max(0, d[j,t]) / (Σ_j q[j,t] + ε)
```

`q`는 0~1의 signal quality·coverage·동기화 신뢰도다. 누적 부하는 망각을
포함한 causal 적분으로 계산한다.

```text
L(t) = exp(-λΔt) × L(t-Δt) + Δt × load(t)
λ    = ln(2) / half_life_sec
```

`L`은 사건 라벨이 아니라 패턴 모델의 입력 후보이며, context·활동·수면·센서
아티팩트와 함께 평가한다.

### 3.6 회복 지표

```text
recovery_rate(t) = - slope_W(load)
recovery_delay   = event_end 이후 load가 기준선 범위로 돌아오는 시간
```

회복 지연은 `DECREASING`·`RECOVERY`를 설명하는 피처로 사용하되, 사건 라벨을
이 값의 임계값으로 생성하지 않는다.

## 4. 연구 기반 패턴 prior

아래 식은 합성 시나리오와 초기 후보 가중치용이다. `pattern_score`만으로
임상 행동을 확정하지 않는다. 모든 패턴은 `label_source`와 `review_status`를
분리해 저장한다.

### 4.1 `meltdown_like` — 과부하·조절상실 양상

연구에서 생리적 각성, EDA, 심박/HRV, 움직임과 관찰행동의 시간적 관계를
살피는 경우가 있다. 합성 prior에서는 단일 신호가 아니라 여러 축의 상승과
회복 지연을 조합한다.

```text
S_meltdown(t) =
    b
  + w_a × autonomic_escalation(t)
  + w_e × eda_escalation(t)
  + w_m × motor_disruption(t)
  + w_l × L(t)
  + w_s × pre_event_slope(t)
  + w_r × recovery_delay_proxy(t)
  + w_q × quality_gate(t)
```

후보 관찰행동은 `ear_covering`, `head_turn_away`, `withdrawal_movement`,
`exit_attempt`, `motion_freeze` 등이다. `S_meltdown`의 높은 값은 “과부하 양상
후보”이지 의학적 진단이나 개입 지시가 아니다.

### 4.2 `tantrum_like` — 요구·상황 반응 양상

턴트럼과 멜트다운의 차이는 생체 부하만으로 결정하지 않는다. 요구가 있었는지,
의사소통 가능했는지, 대안이나 경계에 어떻게 반응했는지는 관찰자·상황 라벨이
필요하다. 따라서 센서 단독 score를 만들지 않고, 다음 조건부 점수로 저장한다.

```text
S_tantrum(t) =
    gate(request_present,
         goal_observed,
         communication_available,
         observer_context_valid)
  × [b + Σ w_j × d[j,t] + w_c × context_features(t)]
```

필수 맥락이 `UNKNOWN`이면 결과는 `NOT_DECISIONABLE`이다. 생체 부하가 낮다는
이유로 행동의 의도나 책임을 추정하지 않는다.

### 4.3 `sensory_seeking_like` — 감각 입력을 반복적으로 찾는 양상

감각 자극·물체 접촉·반복 움직임과 context, 개인별 각성 반응을 함께 본다.

```text
S_seeking(t) =
    b
  + w_p × sensory_context_exposure(t)
  + w_o × repetitive_object_contact(t)
  + w_m × repetitive_movement(t)
  + w_a × autonomic_reactivity(t)
  + w_c × context_match(t)
  + w_q × quality_gate(t)
```

높은 각성만으로 감각추구를 정의하지 않는다. 같은 움직임이 운동·놀이·회복·
감각회피일 수 있으므로 hard negative와 observer context가 필요하다.

### 4.4 `stereotypy_like` — 반복·규칙성 운동 양상

상동행동 후보는 생리 부하보다 움직임의 반복성과 시간 구조가 핵심이 될 수
있다.

```text
periodicity      = max_autocorr(acc_norm, lag ∈ candidate_periods)
spectral_focus   = max_power / total_power
rhythmic_entropy = spectral_entropy(acc_norm)
repeat_duration  = consecutive_duration(movement_state)

S_stereotypy(t) =
    b
  + w_p × periodicity
  + w_f × spectral_focus
  - w_h × rhythmic_entropy
  + w_d × repeat_duration
  + w_c × context_features(t)
```

상동행동 후보는 낮은 부하·높은 부하·즐거움·자기조절 상황에도 나타날 수
있으므로 생리값 하나로 원인을 결정하지 않는다.

### 4.5 `shutdown_like` — 내부화·반응저하 양상

필요할 때 별도 검토 패턴으로 둔다.

```text
S_shutdown(t) =
    b
  + w_f × motion_freeze(t)
  + w_w × withdrawal_movement(t)
  + w_c × communication_drop(t)
  + w_l × L(t)
  + w_r × recovery_delay_proxy(t)
```

말하지 않거나 움직임이 적다는 사실만으로 shutdown을 확정하지 않는다.
수면·정상 휴식·센서 결측·운동 정지 hard negative를 포함한다.

## 5. 가상 라벨과 타깃

### 5.1 사건 구간

패턴 `k`의 합성 사건 경계가 `onset_k`, `end_k`일 때:

```text
event_binary_k(t) = 1[onset_k ≤ t ≤ end_k]
```

### 5.2 선행 예측

```text
will_start_k_H(t) = 1[0 < onset_k - t ≤ H]
```

기본 `H`는 `30, 60, 120, 300`초로 두고, 표준 정상 상태 예측의 30분
`H=1800`초와 혼동하지 않는다.

### 5.3 단계

`NO_EVENT`, `LOW`, `MEDIUM`, `HIGH`, `DECREASING`, `RECOVERY`는 사건 phase
시간축에서 생성한다. 파생변수 threshold가 단계 라벨을 만들지 않는다.

### 5.4 라벨 출처

```text
synthetic_truth       숨은 사건 엔진이 만든 합성 정답
synthetic_prior_v1    연구 기반 가상 패턴 시나리오
model_predicted       모델이 제안한 후보
observer_confirmed    관찰자가 확인한 라벨
expert_reviewed       전문가 검토 완료 라벨
```

`model_predicted`는 재학습 정답으로 바로 사용하지 않는다. 전문가 확인 전에는
`PSEUDO_LABEL_CANDIDATE`로만 보관한다.

## 6. 표준 예측 모델과 패턴 모델 결합

### 6.1 일반 표준 모델 `M0`

각 랜덤 origin `t0`에서 과거 데이터만 사용해 다음 30분의 정상 신호를
예측한다.

```text
ŷ_m(t0+h), h=1..1800
```

평가는 미래 구간 중 `NO_EVENT`, `feature_valid=true`인 행을 주 평가범위로
한다. 정상 구간 MAE·RMSE·정규화 MAE·상관·예측구간 coverage를 보고, 사건이
포함된 경우 정상 구간 성능과 사건 onset 성능을 분리한다.

### 6.2 패턴 모델 `M2`

표준 예측 잔차와 파생변수 가중치를 함께 학습한다.

```text
S_k(t) = b_k
       + Σ_j w_kj × d[j,t]
       + γ_k × forecast_residual(t)
       + η_k × forecast_uncertainty(t)
       + δ_k × L(t)
       + θ_k × context(t)
       + φ_k × quality(t)

p_k(t) = sigmoid(S_k(t))
```

모든 연속 피처는 train에서만 표준화한다. Logistic Regression을 해석 가능한
가중치 기준선으로 두고 HistGradientBoosting/XGBoost를 비선형 후보로 비교한다.
가중치는 수동 임상 임계값이 아니라 train에서 학습하며, validation에서
threshold와 calibration을 고정한다.

## 7. 사람별 차이와 어댑터

전역 prior와 개인 파라미터를 분리한다.

```text
personal_prediction =
  (1 - α_p) × global_model
  + α_p × personal_adapter

α_p = n_eff / (n_eff + shrinkage_strength) × quality_confidence
```

개인 사건·행동 라벨이 부족하면 `α_p`를 제한하고 전역 모델을 사용한다. 개인별
threshold·가중치가 전역 패턴을 과도하게 지우지 않는지 AUCPR, event recall,
false alerts/hour, ECE, lead time을 함께 비교한다.

## 8. 검증 계약

- person/session grouped split: train 24명, validation 6명, locked test 6명
- 랜덤 행 분할 금지
- pattern label과 feature 계산 모듈을 분리
- `truth_*`, event id, hidden pattern, 미래 라벨은 입력 금지
- 패턴별 AUCPR, event recall/F1, false alerts/hour, Brier/ECE
- 단계별 macro-F1와 순서 위반·전이 정확도
- 사람 macro 평균과 confidence interval 또는 multi-seed 변동
- EDA-only, ECG/HRV-only, Watch-only, H10-only, Watch+H10 ablation
- quality/missing/clock shift/noise stress 별 성능 감소율

## 9. 문헌 근거와 사용 방법

다음 연구군은 신호·관찰행동·상황을 연결하는 시나리오 후보를 제공하지만,
보편적 임상 임계값의 출처로 사용하지 않는다.

- [Predicting Imminent Aggression Onset in Minimally-Verbal Youth with Autism Spectrum Disorder Using Preceding Physiological Signals](https://pmc.ncbi.nlm.nih.gov/articles/PMC6230252/)
- [Electrodermal and Behavioral Responses of Children With Autism Spectrum Disorders to Sensory and Repetitive Stimuli](https://pmc.ncbi.nlm.nih.gov/articles/PMC6960008/)
- [Exploring Subtypes of Repetitive Behavior in Children with Autism Through Functional Analysis and Wearable Technology](https://pmc.ncbi.nlm.nih.gov/articles/PMC9881513/)
- [From motion to emotion: exploring challenging behaviors in autism spectrum disorder through analysis of wearable physiology and movement](https://pmc.ncbi.nlm.nih.gov/articles/PMC11775438/)

연구에서 보고된 방향은 `research_prior_strength`로 약하게 반영하고, 사람별
기준선·context·관찰자 라벨·센서 품질을 함께 사용한다. 논문 숫자를 그대로
사용하거나 센서만으로 진단·개입을 자동 결정하지 않는다.

## 10. 현재 구현 상태

- Goal 1 truth generator: 사건·phase·latent truth 구현
- Goal 1.5 Oracle ML: latent 기반 causal 파생변수와 사건/단계 모델 구현
- 실제 EDA·ECG·HRV raw → feature extractor: 후속 observed/model-ready 단계
- 30분 일반 표준 forecast: 설계 계약, 구현·검증 전
- 임상 연구 기반 pattern prior: 이 문서로 계약을 먼저 고정하며, 구현 후
  `pattern-priors/v1`로 별도 dataset/manifest를 만든다.

합성 결과는 `oracle/sanity`, 실제 행동·의료 성능은 `NOT VERIFIED`로 보고한다.
