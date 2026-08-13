# Goal 1.5 학습 아키텍처

## 범위

이 단계는 synthetic truth를 이용해 학습 코드와 평가 규칙이 작동하는지
검증하는 oracle/sanity 단계다. 실제 wearable 신호의 물리적 노이즈 제거,
clock correction, native signal model, TCN, 실제 데이터 fine-tuning은 범위
밖이다.

## 데이터 경계

입력 whitelist는 7개 latent factor, `is_awake`, context, 시간 주기뿐이다.
`active_target_*`, hard-negative ID/type, artifact schedule ID, participant
truth baseline, intensity truth는 feature frame으로 들어가면 즉시 거부된다.
라벨은 별도 `events.parquet`에서만 생성한다.

한 사람의 식별자는 `(run_id, person_id)`이며 SHA-256 순서로 각 12-person
run을 8 train / 2 validation / 2 locked_test로 나눈다. 세 seed의 총합은
24/6/6이며 person 교차 중복은 허용하지 않는다.

## 기준선 모델

global baseline은 train 사람의 context별 median/MAD다. personal baseline은
현재 시점 이전의 값만 사용하며 warm-up 1,800초, lookback 21,600초,
refresh 60초다.

```text
personal_weight =
  0                                           (warm-up 전)
  n_eff / (n_eff + 1800) × quality_confidence (warm-up 후)
```

oracle의 quality confidence는 1이다. 실제 데이터에서는 이 항에 장비 품질과
동기화 신뢰도를 반영할 수 있지만 Goal 1.5에서는 꾸며 넣지 않는다.

## 패턴 모델

각 target에 두 후보를 독립 학습한다.

- `event_binary`: onset부터 event end까지 1
- `forecast_60s`: onset 1–60초 전만 1
- Logistic Regression, class-weight balanced
- HistGradientBoosting, class-weight balanced

학습 행은 모든 positive, 모든 hard-negative, positive 수의 최대 3배인
결정적 matched baseline이다. validation과 locked test는 1Hz 전체
타임라인을 평가한다.

## 평가

validation event-level F1로 threshold를 고정한다. test에서 threshold를
바꾸지 않는다. AUCPR, row/person macro recall/F1, event recall/F1,
false alerts/hour, Brier, ECE, confusion matrix, forecast lead time을 기록한다.

stress harness는 Gaussian 0.05/0.10/0.20, block missing 5/30/120초,
time shift ±1/±5초, latent dropout 25/50/100%를 결정적으로 적용한다.
이는 실제 센서 노이즈 제거 성능이 아니라 feature-level 민감도 검사다.

## 실제 데이터 연결 지점

공통 시간축은 corrected UTC, cardiac reference는 Polar H10,
Watch ECG는 calibration-only다. ECG–PPG 생리적 lag는 clock error와 별도
필드로 보존한다. 공개 계약은
`schemas/synchronization.schema.json`이며, oracle에서는 모든 측정값을
비우고 `NOT_AVAILABLE_TRUTH_ONLY`로 기록한다.

## Kidsignal 운영 모델 플랫폼 경계

Goal 1.5 Oracle 아키텍처는 운영 실데이터 모델과 분리한다. 운영 모델 계약의
canonical 저장소는 이 저장소(`multisensor_ml`)이며, `multisensor_synth`는 계속
합성 truth 생성기로만 사용한다.

- Android/Watch/H10: raw, vendor observation, capture metadata 수집
- GCS: raw 및 committed artifact canonical lake
- BigQuery: 장기 projection과 UUID/digest로 동결한 training cohort
- 모델/Kaggle: 동결 cohort로 reference/candidate 학습 및 immutable bundle 생성
- Cloud Run: corrected time, quality, baseline, feature, ACTIVE/CANDIDATE 추론
- DuckDB `:memory:`: 즉시 흐름·품질·지연 관찰만 수행
- Neon: migration-only legacy

Watch-only, H10-only, Watch+H10은 서로 다른 feature schema와 모델이다. 필수 source
누락을 생체값 0/null로 대체하지 않는다. stage 1~5는 시간 순서이며 severity ordinal이
아니다. 독립 review가 완료되지 않은 변화나 novel pattern은 false positive로 계산하지
않는다. 상세 계약은 `docs/KIDSIGNAL_MODEL_PLATFORM_CONTRACT_KO.md`를 따른다.

현재 Watch+H10 fused feature는 `PROPOSED/NOT VERIFIED`다. exact composition은 scheduling
교집합과 corrected-time provenance만 수행하며 waveform alignment, offset/lag 추정,
보간, 리샘플링은 별도 계약 전까지 금지한다. 모델 notebook에는 train+validation만
노출하고 locked holdout은 sealed evaluator가 최종 immutable bundle로만 평가한다.
