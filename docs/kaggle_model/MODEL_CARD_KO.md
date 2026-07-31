# Goal 1.5 전체 계층형 합성 모델 카드

## 목적

이 모델은 최초 합성데이터에서 개인 기준선, `STD-A`, 사건, 5단계, 행동 10개가 어떻게 연결되는지 팀이 Kaggle에서 재현하고 아이디어를 논의하기 위한 `candidate`입니다. 상태는 `oracle/sanity`이며 실제 웨어러블 정확도는 `NOT VERIFIED`입니다. 의료 진단·치료·위험 판정에 사용할 수 없습니다.

## 데이터와 분할

- synthetic person 36명, train 24명, validation 6명, locked test 6명
- 사건 930건, 행동 양성 라벨 2,416건
- 사람 기준 분할이며 중복은 0명입니다.
- 이번 Kaggle 재현 샘플은 validation 사람만 사용하며 locked test를 다시 읽지 않습니다.

## 모델 구조와 주요 선택

1. 전역·개인 causal 기준선에서 robust 파생변수를 만듭니다.
2. 표준 타입은 GMM K=1의 `STD-A` soft membership을 사용합니다.
3. 사건 head는 HistGradientBoosting과 validation 임계값 `0.994599`를 사용합니다.
4. 사건 후보에만 HistGradientBoosting 5단계 head와 causal decoder를 적용합니다.
5. 모델 1 확률과 기준선·STD 변수를 행동별 one-vs-rest 모델에 입력합니다.
6. 행동별 validation 결과에 따라 Logistic 또는 HistGradientBoosting을 선택했습니다.

validation에서 사건 AUCPR은 0.5111, event F1은 0.7414, 5단계 macro-F1은 0.6324였습니다. 행동 모델은 보조지표이며 행동별 AUCPR 편차가 큽니다. 수치는 합성 oracle 안의 재현 지표이지 실제 정확도가 아닙니다.

## 취약점

Gaussian perturbation, 시간축 shift, latent-axis dropout에서 성능 저하가 컸습니다. 실제 Muse·Polar·Watch 신호와 물리적 동기화 성능은 검증되지 않았습니다. 원시 센서 신호를 이 모델에 직접 넣을 수 없습니다.

## 안전한 사용

- `.skops` unknown type과 SHA-256을 확인한 후 로드합니다.
- `.pkl`, `.pickle`, `.joblib`은 사용하지 않습니다.
- `oracle/sanity`, `candidate`, `NOT VERIFIED` 표시를 제거하지 않습니다.
- 모델 선택이나 임계값 변경은 별도 validation 실험으로 기록합니다.
