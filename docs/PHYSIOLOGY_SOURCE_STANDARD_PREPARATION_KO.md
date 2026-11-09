# 장비별 생체 기준선 준비 · 2026-09-05

상태: LOCAL AUDIT PREPARATION / BLOCKED_NO_REAL_COHORT / NOT EVALUABLE.
Phase06f descriptive core 3/3/24의 후속 구현·해시 차이는
[consumer 호환 기록](PHYSIOLOGY_V1_COMPATIBILITY_KO.md)을 따른다.
2026-09-06 신규 reader/개인 baseline/export 로컬 연결은
[physiology handoff 제안](PHYSIOLOGY_BASELINE_HANDOFF_V1_KO.md)을 따른다.
이해한 범위: 장비별 독립 생체 기준선의 consumer와 Cloud 인계 조건을 준비한다.
모델 소유 checkout은 multisensor_ml이며 multisensor_synth의 합성 truth와 구별한다.
권위 방향은 앱 docs/phases/PHASE-06e-live-features-standard-and-watch-output.md다.

## 확정 경계

- Watch/H10/Muse 세 모델군은 각각 자기 source로 준비·학습한다. 장비 교집합 불필요.
- 사람별 baseline/adapter는 cohort-scoped subject로 묶고 source/schema별 분리한다.
- motion/ACC/Gyro/배터리/연결/접촉/품질은 predictor가 아닌 gate sidecar다.
- EEG 기준 이탈은 stress 증가 방향을 뜻하지 않는다. 밴드 비율의 방향을 고정하지 않는다.
- Watch motion 포함 16→15와 기존 load target은 LEGACY CURRENT다. 기존 해시는 보존한다.
- H10 13-feature에도 quality/missingness와 기존 load가 있어 그대로 최종 표준으로 쓰지 않는다.
- 기존 standard 24/behavior 26 필드 및 receipt는 legacy consumer 계약으로 유지한다.
- 3시간·3일, 5/15/30분, median 방향합산/3은 승인된 계약이 아니며 gate로 사용하지 않는다.

## 최소 source별 제안

| source | 우선 파생 후보 | 추가 확인 |
|---|---|---|
| Watch | HR median, EDA median/변화, 유효 IBI의 변동 | IBI 순서/시간, EDA 상태, PPG/온도 지원을 별도 schema로 고정 |
| H10 | corrected HR median, RR median, window 내 RMSSD | RR beat timestamp 미확정이면 window 경계를 넘는 RR 차분 금지 |
| Muse | 검증된 채널별 EEG power/상대 power와 개인 기준 잔차 | sample rate·채널·단위·필터/PSD·접촉·시계 계약부터 필요 |

RMSSD = sqrt(sum((RR[i+1]-RR[i])**2)/(n-1)), n>=2는 계산 가능 조건일 뿐
통계적 안정성·임상 해석·충분한 수집시간의 근거가 아니다. ECG R-peak/HRV 재생성은
Cloud 버전 알고리즘과 parity가 확보된 후 별도 feature로 추가한다.
PPG/피부온도 등 추가 채널은 실제 값과 단위·지원시간이 검증된 schema에서만 요구한다.
없는 채널은 zero/null padding하지 않는다. 하나의 source 실패로 다른 source를 막지 않는다.

첫 target 제안은 **생리 feature별 개인 안정 기준과 관측치의 잔차**다.
TRAIN의 품질 유효 NO_PATTERN 구간으로 feature별 center/scale을 고정하고,
VALIDATION에서는 재보정 없이 기준 대비 잔차를 기록한다. scale=0은 별도 불충분 상태다.
이는 미래 예측이나 독립 stress 정답의 정확도가 아니다. [-1,+1] 공통 score,
feature 합산·가중치·방향은 아직 미확정이며 배포 scale을 생성하지 않는다.

## 최소 시간과 분할

일률적인 900/1000초 또는 장기 horizon을 신규 gate로 전용하지 않는다.
Cloud는 source별 native coverage, 중복·역행·gap, feature support interval,
유효 window 수/시간, 최장 연속구간, 세션·날짜별 분포, NO_PATTERN 제외 이유를 제공한다.
feature별 minimum samples/history는 계산 알고리즘에 귀속한다. baseline의 충분성은
TRAIN 내 center/scale 안정성 및 유효 지원량으로 사전등록하고 VALIDATION에서 튜닝하지 않는다.
아직 새 source별 수치 기준은 미확정이다. raw 10초 bin 수를 유효 baseline 시간으로 승격하지 않는다.

개인 모델은 chronological-per-subject, population 검증은 person-group으로 나눈다.
capture는 한 subject에만 귀속하고, 실제 feature의 원천 support interval이 split 경계를
넘으면 제외한다. 현 최소 purge 1800초는 보존하되 더 긴 support를 쓰면 확대가 필요하다.
LOCKED는 sealed evaluator만 접근한다. 미래 target은 별도 계약이며 현 baseline gate가 아니다.
NO_PATTERN 미확인/사람 observation/패턴 접촉 hour는 기존 제외 정책을 보존한다.

## 로컬 consumer

`physiology_source_readiness.audit_source_windows()`는 Cloud adapter가 제공할 source별
normalized window를 로컬 집계한다. 새로운 BigQuery view나 production wire schema가 아니다.
FeatureSpec의 origin은 승인 schema에서 가져와야 하며 self-declared origin을 신뢰하는
운영 보안 validator로 사용하지 않는다. 이 함수는 schema 승인/해시/receipt 검증을 대체하지 않는다.
유효 시간은 subject/capture별 시간 합이다. 동시에 열린 capture의 unique person time과 다르다.
중복/overlap window·capture의 subject 혼합은 거부하고, quality/clock/lineage/유한값/NO_PATTERN
실패는 별도 겹침 predicate로 계수한다. 생체값·subject/capture 식별자는 반환하지 않는다.
유효 window가 있어도 training_ready=false, fit_call_count=0이다.

## TimesFM 2.5 확인 및 향후 pin

공식 코드 검토 revision: `0df95ae62085a6ac0d0afd1ad40dee2e6c1356ab`.
실행 시 code revision과 `google/timesfm-2.5-200m-pytorch` weight revision/checksum을 함께
동결해야 한다. weight revision은 아직 확보하지 않았으며 모델을 다운로드/실행하지 않았다.

- src/timesfm/timesfm_2p5/timesfm_2p5_base.py: input patch 32, context limit 16384,
  quantile head length 1024. 모두 step 단위이며 32개가 정확도 보장/수집 gate는 아니다.
- 같은 파일 forecast는 leading NaN 제거 후 linear_interpolation을 호출한다.
  `validate_timesfm25_context()`가 유한값과 연속 고정 grid를 사전검증한다.
  내부 masked patch padding은 실제 생체 결측 대체와 구별한다.
- XReg 및 HF+PEFT LoRA 예제 지원은 공식 README에 있다. 실행·적합 검증은 미수행.
- 3.0 예제/weight는 섞지 않는다. 공식 README는 2.5까지 Apache-2.0,
  3.0 weight는 별도 non-commercial/non-production 조건으로 설명한다.
- 향후 같은 실제 관측 target·horizon·split에서 persistence와 비교한다.
  baseline deviation과 forecasting은 서로 다른 task다. EEG stress truth를 생성하지 않는다.

공식 근거: https://github.com/google-research/timesfm
코드: https://github.com/google-research/timesfm/blob/0df95ae62085a6ac0d0afd1ad40dee2e6c1356ab/src/timesfm/timesfm_2p5/timesfm_2p5_base.py

## Cloud 다음 입력

1. 장비별 독립 canonical feature schema JSON: 순서/단위/수식/support/품질 sidecar/UUID/hash.
2. 같은 raw의 corrected timestamp·source/capture 및 generation/checksum을 검증한 derived
   parity receipt와 위 source별 집계. 직접 identity/원시 경로는 모델 공개면에서 제외한다.
3. source별 baseline target·NO_PATTERN admission/제외 정책·split/support/purge의 동결 계약.
4. 실제 양수 TRAIN+VALIDATION의 authorized cohort와 public cohort/split digest,
   schema identity를 묶는 create-only model-sync receipt 및 destination readback.
5. 신규 source schema를 consumer/exporter/Cloud verifier가 승인했다는 compatibility 증거.

2026-09-05 Cloud 전달 기준 standard/behavior 6테이블 0행이다. 이번 작업에서는 live 재조회,
fit, backfill, bundle, 배포, commit/push를 수행하지 않는다. fixture는 계약 검증 전용이다.
