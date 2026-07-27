# Goal 1.5 계층형 MVP 결과

실행일: 2026-07-27  
데이터: `mvp3-oracle-v1`  
릴리스: `goal15-mvp3-oracle-v1-a51f29988fc0` (`candidate`)

이 결과는 합성 truth·latent를 이용한 `oracle/sanity`입니다. 실제 행동 예측
정확도나 의료 성능으로 해석할 수 없습니다. 실제 데이터 성능은
`NOT VERIFIED`, 장비 동기화는 `NOT_AVAILABLE_TRUTH_ONLY`입니다.

## 데이터와 라벨

- seed 3개, 각 12명, 총 36명
- train 24명, validation 6명, locked test 6명
- 사람 중복 0
- 1Hz 5단계 timeline 15,552,000행
- outcome 사건 930건
- 행동 양성 라벨 2,416건
- 10개 행동 코드 모두 train positive 20건·validation positive 5건 조건 통과

## STD 타입과 OOD

GMM `K=1..6` 비교 결과 `K=2..6`은 안정성·최소 유효 인원 조건을 통과하지
못해 `K=1`, 즉 `STD-A` 하나로 복귀했습니다. 36명 중 31명은
`IN_DISTRIBUTION`, 5명은 `OOD_MONITOR`입니다. 합성 데이터에서 근거 없는
복수 표준 타입을 강제로 만들지 않았다는 의미입니다.

## 모델 1: 사건과 5단계

validation event-F1과 5단계 macro-F1 기준으로 두 head 모두
HistGradientBoosting이 선택됐습니다. 임계값은 validation에서
`0.994599`로 고정했고 locked test에서 변경하지 않았습니다.

| 역할 | AUCPR | event recall | event F1 | false alerts/hour | Brier | ECE | 5단계 macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 0.5111 | 0.6232 | 0.7414 | 0.0056 | 0.0643 | 0.1837 | 0.6324 |
| locked test | 0.4871 | 0.5135 | 0.6609 | 0.0042 | 0.0658 | 0.1832 | 0.6282 |

## 모델 2: 행동 다중 라벨

모델 1의 train 입력은 사람 단위 OOF 확률입니다. validation AUCPR로 행동별
Logistic/HGB 후보를 비교했습니다.

| 행동 코드 | 선택 모델 | validation AUCPR | locked-test AUCPR |
|---|---|---:|---:|
| `ear_covering` | Logistic | 0.2041 | 0.2139 |
| `exit_attempt` | Logistic | 0.2361 | 0.1870 |
| `head_turn_away` | HGB | 0.3731 | 0.2492 |
| `motion_freeze` | Logistic | 0.1974 | 0.1279 |
| `movement_reduction` | HGB | 0.2866 | 0.2725 |
| `repetitive_body_movement` | HGB | 0.3648 | 0.3828 |
| `repetitive_hand_movement` | HGB | 0.3608 | 0.3818 |
| `repetitive_object_contact` | HGB | 0.2110 | 0.2167 |
| `sustained_pressure_or_contact` | HGB | 0.1901 | 0.1591 |
| `withdrawal_movement` | HGB | 0.4833 | 0.2565 |

행동 모델 수치는 모델 1보다 낮으며, 합성 라벨 안에서도 행동 구분이 더
어렵다는 것을 보여줍니다. 이 수치를 실제 행동 정확도로 확대 해석하면
안 됩니다.

## Noise stress

- Gaussian robust-z `0.05/0.10/0.20`에서 AUCPR 감소율은
  약 69%/85%/90%였습니다.
- `±1초` shift에서도 감소가 관찰됐고, `-5초`에서는 false alerts/hour가
  1.8까지 증가했습니다.
- latent-axis dropout에서는 event recall이 0으로 떨어졌습니다.
- block missing 결과는 이번 50,000행 결정적 표본에서 거의 변하지 않았습니다.

따라서 이 모델은 합성 oracle 기준에서는 패턴을 학습했지만, 실제 웨어러블
노이즈에 강하다고 볼 근거가 없습니다. 다음 Goal에서 실제 신호·clock
correction·생리적 lag 분리 검증이 필요합니다.

## 레지스트리와 안전성

- release 상태는 `candidate`; 자동 승격하지 않음
- 같은 설정 재실행은 약 5초 안에 기존 receipt와 릴리스를 재사용
- locked test는 model hash×dataset hash당 한 번만 기록
- 모든 모델은 `.skops`; manifest의 unknown type 목록은 비어 있음
- `.pkl`·`.joblib` 산출물 없음
- 한국어 라우터는 `판정`, `단계`, `분포상태`, 행동 확률을 출력하고
  raw probability를 감사용으로 함께 보존
