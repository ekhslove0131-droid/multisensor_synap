# Kaggle 전체 계층형 모델 재현 사용법

## 목표

비공개 Kaggle Dataset과 비공개 Kaggle Model을 연결해 기존 합성 validation 샘플의 결과가 `REPRODUCED`인지 확인합니다. 이 실행은 추론만 수행하며 새 학습, GPU, locked test를 사용하지 않습니다. 결과 범위는 `oracle/sanity`, 실제 성능은 `NOT VERIFIED`이고 의료 용도가 아닙니다.

## Kaggle 사용 순서

1. `multisensor-goal15-hierarchical-reproduction` 노트북을 엽니다.
2. Input에서 `bjcoding/multisensor-goal15-oracle-mvp3` Dataset과 `multisensor-goal15-hierarchical/scikitLearn/oracle-sanity-v1` Model이 연결됐는지 봅니다.
3. Accelerator가 `None`이고 Internet이 꺼졌는지 확인합니다.
4. `Run All`을 한 번 실행합니다.
5. 마지막 receipt에서 `status=REPRODUCED`, `run_training=false`, `run_locked_test=false`, `real_data_status=NOT VERIFIED`를 확인합니다.

노트북은 archive·내부 파일 SHA-256, `.skops` unknown type, feature 순서, 샘플 출력의 확률 허용오차와 범주형 exact match를 검사합니다. 하나라도 다르면 `FAILED`로 중단합니다.

## 결과 읽기

- `event_probability`: 사건 확률
- `predicted_stage`: `NO_EVENT`, `LOW`, `MEDIUM`, `HIGH`, `DECREASING`, `RECOVERY`
- 행동 코드 10개 열: 각 행동의 raw probability
- `판정`, `단계`, `분포상태`: 한국어 라우터 결과
- OOD 또는 품질 실패: raw probability는 남기고 `판단 보류`

## 다른 합성 샘플 보기

train 또는 validation의 prepared Parquet만 선택할 수 있습니다. 동일 feature schema로 추론하되 immutable expected output 비교는 고정 샘플에만 적용합니다. locked test 사람을 샘플로 바꾸지 않습니다.

## 오류 해결

- `hash mismatch`: Dataset/Model version을 확인하고 다시 연결합니다.
- `unknown type mismatch`: 임의 신뢰 목록을 추가하지 말고 wheel과 모델 version을 맞춥니다.
- `missing feature`: 입력을 1차 prepared schema로 다시 만듭니다.
- `FAILED`: 첫 불일치 열과 key를 기록하고 모델을 재학습하거나 임계값을 바꾸지 않습니다.

로컬에서는 `multisensor-ml kaggle-model verify` 후 `kaggle-model reproduce` 명령으로 같은 검사를 실행합니다.
