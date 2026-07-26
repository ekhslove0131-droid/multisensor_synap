# MVP 3-seed oracle/sanity 결과

## 상태

- seed: 20260725, 20260726, 20260727
- 사람: 36 synthetic person
- split: train 24 / validation 6 / locked test 6
- source: Goal 1 truth only
- real accuracy: **NOT VERIFIED**
- device synchronization: **NOT VERIFIED**
- oracle synchronization: `NOT_AVAILABLE_TRUTH_ONLY`

대용량 산출물은 Git에 포함하지 않는다. 로컬 기준 경로는
`artifacts/mvp3-oracle-v1`이다.

## clean 지표

| target | model | role | AUCPR | event recall | false alerts/hour | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|---:|
| event | HistGradientBoosting | validation | 0.4170 | 0.5942 | 0.0083 | 0.0627 | 0.1751 |
| event | HistGradientBoosting | locked test | 0.4179 | 0.4730 | 0.0056 | 0.0632 | 0.1746 |
| event | Logistic Regression | validation | 0.2149 | 1.0000 | 0.0986 | 0.1714 | 0.3605 |
| event | Logistic Regression | locked test | 0.2001 | 1.0000 | 0.1056 | 0.1678 | 0.3555 |
| forecast | HistGradientBoosting | validation | 0.2519 | 1.0000 | 0.0972 | 0.0111 | 0.0493 |
| forecast | HistGradientBoosting | locked test | 0.2387 | 1.0000 | 0.1042 | 0.0112 | 0.0495 |
| forecast | Logistic Regression | validation | 0.0162 | 1.0000 | 0.0972 | 0.2468 | 0.4659 |
| forecast | Logistic Regression | locked test | 0.0083 | 1.0000 | 0.1042 | 0.2497 | 0.4677 |

forecast의 mean/median detected lead time은 네 clean model/role 조합에서 60초였다.
이는 validation threshold가 forecast block 시작부터 alert를 낸 oracle 결과이며,
실제 장비 lead time으로 해석할 수 없다.

## stress

13개 조건과 4개 model-target 조합을 validation 전체 타임라인으로 평가했다.
가장 큰 AUCPR 감소 예시는 다음과 같다.

| target/model | worst scenario | stressed AUCPR | degradation |
|---|---:|---:|---:|
| event / HGB | latent dropout 100% | 0.0242 | 94.2% |
| event / Logistic | latent dropout 100% | 0.0194 | 91.0% |
| forecast / HGB | Gaussian 0.20 | 0.0018 | 99.3% |
| forecast / Logistic | latent dropout 25% | 0.0018 | 89.1% |

이 결과는 특히 forecast 모델이 feature perturbation에 취약하다는 oracle 경고다.
실제 센서 정확도나 물리적 노이즈 제거 성능을 뜻하지 않는다.

## 실행 규모

- raw truth: 약 782MB
- prepared causal features: 약 9.5GB
- model/evaluation bundle: 약 195MB
- generation: 약 1분
- prepare: 약 4분
- train + locked test + 13 full stress: 약 239분

따라서 반복 개발은 quick → clean candidate selection → 필요한 후보만 full stress
순서가 적합하다. full stress checkpoint/cache 분리는 다음 성능 개선 우선순위다.

## KNIME 5.12 readback

- official Big Data File Formats와 External Tool Support 확장 설치: 완료
- External Tool quick materialization과 PASS 출력: 완료
- 6개 Parquet 화면 실행: 완료
- `.knwf` export 후 별도 `Goal1_5_Readback_2` 폴더 import: 완료
- imported workflow Execute all 및 Composite View readback: 완료
- real observed accuracy/synchronization: **NOT VERIFIED**

화면 증거는 `docs/evidence/`에 저장했다. 이 검증은 KNIME 연결과
oracle/sanity 결과 표시가 재현된다는 뜻이며 실제 센서 성능 검증은 아니다.
