# 센서 가용성별 계층형 모델 사용법

## 목적과 범위

이 패키지는 최초 합성데이터로 학습한 `oracle/sanity` 모델이다. Galaxy Watch8·Polar H10·Muse S가
항상 모두 착용된다는 가정을 깨고, 1종·2종·3종 조합별로 사건 gate와 5단계 상태 decoder를
비교하기 위한 재현 패키지다.

실제 Neon Watch 데이터로 학습한 모델이 아니며 실제 행동 정확도는 `NOT VERIFIED`다. 현재 Neon은
payload 디코드와 시간 교정이 먼저 필요한 상태다. 원시값을 다른 장비의 truth로 바꾸거나 의료적
판단에 사용하지 않는다.

## 포함된 프로파일

| 프로파일 | 의미 |
|---|---|
| `watch_only` | Galaxy Watch8 관측에 해당하는 feature subset |
| `polar_only` | Polar H10 관측에 해당하는 feature subset |
| `muse_only` | Muse S 관측에 해당하는 feature subset |
| `watch_polar` | Watch + Polar |
| `watch_muse` | Watch + Muse |
| `polar_muse` | Polar + Muse |
| `watch_polar_muse` | 세 장비 전체 |

각 프로파일은 동일한 causal time/context/history 피처와 허용된 latent factor subset만 사용한다.
프로파일 이름은 feature-ablation 실험명이지 장비가 실제로 EEG/ECG를 측정한다는 뜻이 아니다.

## 모델 구조

1. Logistic Regression 사건 head가 `event_probability`를 출력한다.
2. validation에서 threshold를 고정한다.
3. 사건 후보에 대해 Logistic Regression 5단계 head를 적용한다.
4. 상태 decoder가 `NO_EVENT → LOW → MEDIUM → HIGH → DECREASING → RECOVERY` 전이를 적용한다.
5. 결측·품질 실패·OOD는 억지로 상태를 유지하지 않고 이후 observed adapter에서
   `NOT_DECISIONABLE`로 라우팅한다.

CPU에서 재현성을 우선해 이 패키지의 프로파일 후보는 Logistic Regression으로 고정했다. 전체
Oracle benchmark의 HGB 후보와 성능을 혼합하지 않는다.

## 로컬 생성

ML 저장소에서 다음처럼 합성 train/validation만 사용해 패키지를 만든다.

```text
multisensor-ml availability-model package \
  --project-root /path/to/multisensor_ml \
  --series mvp3-oracle-v1 \
  --output artifacts/availability-model/oracle-sanity-v1 \
  --wheel dist/multisensor_ml-0.1.0-py3-none-any.whl \
  --max-rows-per-person 20000
```

`--wheel`은 Kaggle/Colab에서 동일한 Python runtime을 재현하기 위한 wheel 경로다. wheel 옆에
있는 오프라인 의존성 wheel도 함께 payload에 기록한다. 생성 명령은 `locked_test`를 읽지 않고,
일곱 프로파일의 모델·feature schema·validation sample·
manifest·SHA-256 archive를 만든 뒤 즉시 안전한 `.skops` 재로딩과 동일 prediction을 검증한다.

```text
multisensor-ml availability-model verify \
  --package artifacts/availability-model/oracle-sanity-v1
```

패키지에는 `.pkl`, `.pickle`, `.joblib`를 넣지 않는다. `model_manifest.json`의 archive hash,
`payload_manifest.json`의 파일 hash, `locked_test_read=false`, `data_status=oracle/sanity`를
확인한다.

## Kaggle에서 재현

Kaggle Model variation은 `bjcoding/multisensor-goal15-availability`의
`scikitLearn/oracle-sanity-v1`이다. `08_sensor_availability_reproduction.ipynb`는 학습을 하지 않고
다음만 수행한다.

1. Model package의 `model_manifest.json`과 archive hash 확인
2. 일곱 프로파일의 `.skops` 로딩 전 trusted type 확인
3. 프로파일별 validation sample prediction을 기대 parquet와 비교
4. 한국어 상태 receipt 출력 (`REPRODUCED` 또는 구체적 실패)

Notebook 설정은 `RUN_TRAINING=False`, `RUN_LOCKED_TEST=False`, CPU다. synthetic 결과는
`oracle/sanity`, 실제 데이터·실제 accuracy·device sync는 각각 `NOT VERIFIED`,
`NOT_AVAILABLE_TRUTH_ONLY`로 표시한다.

## Google Colab 적용 순서

Colab에서는 Kaggle Model variation을 다운로드해 archive를 풀고, 패키지에 포함된 wheel을 설치한
뒤 다음 계약의 `model_ready` DataFrame을 만든다.

```python
from multisensor_ml.sensor_availability import (
    load_availability_variant,
    predict_availability_variant,
)

variant = load_availability_variant("extracted/profiles/watch_only")
prediction = predict_availability_variant(variant, model_ready_frame)
```

`model_ready_frame`에는 `person_key`, `session_id`, `timestamp_utc`와 선택한 프로파일의 feature
schema가 있어야 한다. 실제 Neon에서는 먼저 `neon_mapping.py`로 payload를 디코드하고,
`corrected_utc`·품질·결측·clock metadata를 계산한 뒤 같은 causal feature 이름을 만드는 adapter가
필요하다. `received_at`를 센서 시각으로 사용하면 안 된다.

## 버전 저장과 승격

모델을 사용할 때 다음을 함께 보관한다.

- Kaggle variation/version과 `model_manifest.json` archive SHA-256
- source dataset manifest hash와 feature schema hash
- `profile_id`, adapter version, dependency/uv lock 정보
- 실행 시각, input dataset/session, quality/OOD 상태, prediction receipt

새 버전은 validation 결과와 변경 사유를 남긴 후보로만 올리고, 실제 라벨이 충분히 쌓인 뒤 수동
승격한다. 합성 benchmark 결과만으로 실제 장비 성능을 주장하지 않는다.
