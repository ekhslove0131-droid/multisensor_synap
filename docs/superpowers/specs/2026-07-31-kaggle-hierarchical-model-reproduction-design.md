# Kaggle 전체 계층형 합성 모델 재현 설계

작성일: 2026-07-31  
상태: 승인된 설계  
대상 저장소: `/Users/baital/dev/multisensor_ml`  
대상 브랜치: `codex/kaggle-ml-dl-benchmark`

## 1. 목적

최초 합성데이터로 이미 학습된 Goal 1.5 전체 계층형 모델을 비공개 Kaggle Model로 등록하고, 팀원이 Kaggle 노트북에서 같은 합성데이터에 모델을 적용해 동일 결과를 재현하도록 한다.

이 작업의 목적은 실제 행동 또는 의료 성능을 배포하는 것이 아니다. 합성데이터에서 개인 기준선, 표준 타입, 사건, 5단계, 행동 예측이 어떻게 연결되는지 팀이 직접 확인하고 다음 아이디어를 논의할 수 있는 재현 환경을 만드는 것이다.

새 학습, GPU 사용, 임계값 변경, locked-test 재실행은 범위에 포함하지 않는다.

## 2. 고정 상태와 해석 경계

- 데이터 상태: `oracle/sanity`
- 실제 웨어러블 정확도: `NOT VERIFIED`
- 실제 장비 동기화: `NOT_AVAILABLE_TRUTH_ONLY`
- 모델 상태: `candidate`
- 기본 표준 타입: `STD-A`, GMM `K=1`
- 저장 형식: `.skops`
- 금지 형식: `.pkl`, `.pickle`, `.joblib`
- 모델 입력에는 hidden archetype, event intensity truth, event truth ID 등 금지된 truth 열을 넣지 않는다.
- Kaggle Model과 재현 노트북은 모두 비공개로 등록한다.

## 3. 배포 단위

### 3.1 Kaggle Model 식별자

권장 handle은 다음과 같다.

```text
bjcoding/multisensor-goal15-hierarchical/scikitLearn/oracle-sanity-v1
```

Kaggle의 상위 Model 아래에 Kaggle CLI 2.2.4 enum과 일치하는 `scikitLearn`
framework와 `oracle-sanity-v1` variation을 둔다. Kaggle의 공식 Model/Variation
구조와 CLI 계약을 따른다.

### 3.2 포함 모델

기본 추론 경로에는 다음 선택 모델을 사용한다.

- 사건 head: HistGradientBoosting
- 5단계 head: HistGradientBoosting
- 행동 head 10개: validation으로 선택된 행동별 Logistic 또는 HistGradientBoosting
- 표준 타입: `STD-A` 모델과 전처리 규칙

팀의 비교·감사를 위해 선택되지 않은 Logistic/HGB 후보도 `candidates/` 아래에 보존한다. 기본 실행은 manifest의 `selected` 매핑만 사용하며 후보를 바꾸어도 예상 재현 결과를 덮어쓰지 않는다.

### 3.3 모델 패키지 구조

```text
multisensor-goal15-hierarchical/
├── model_manifest.json
├── SHA256SUMS
├── MODEL_CARD_KO.md
├── KAGGLE_REPRODUCTION_KO.md
├── FEATURE_LABEL_GUIDE_KO.md
├── EXPERIMENT_LESSONS_KO.md
├── uv.lock
├── wheel/
│   └── multisensor_ml-*.whl
├── baseline/
│   ├── global_baseline.json
│   └── personal_baseline.parquet
├── standard_type/
│   ├── standard_type_model.skops
│   ├── standard_type_preprocessing.json
│   └── manifest.json
├── stage_model/
│   ├── selected/
│   ├── candidates/
│   ├── feature_schema.json
│   └── manifest.json
├── behavior_model/
│   ├── selected/
│   ├── candidates/
│   ├── feature_schema.json
│   └── manifest.json
├── inference/
│   ├── predict.py
│   └── state_decoder.json
└── sample/
    ├── sample_input.parquet
    ├── expected_output.parquet
    └── sample_manifest.json
```

대용량 10GB 합성데이터는 모델 패키지에 중복하지 않는다. 기존 비공개 Kaggle Dataset을 별도 Input으로 연결한다.

## 4. 재현 노트북

### 4.1 노트북 역할

새 노트북은 학습 노트북이 아니라 CPU 기반 추론 튜토리얼이다. 노트북 이름은 `multisensor-goal15-hierarchical-reproduction`으로 한다.

### 4.2 실행 순서

1. 연결된 Kaggle Dataset과 Kaggle Model 경로를 탐색한다.
2. `model_manifest.json`과 `SHA256SUMS`를 검증한다.
3. 모든 `.skops` 파일의 unknown type 목록이 manifest와 일치하는지 확인한다.
4. wheel을 설치하고 `uv.lock` 및 dependency version을 표시한다.
5. 합성 참여자 한 명의 제한된 샘플을 불러온다.
6. 개인 기준선과 `STD-A` 입력을 준비한다.
7. 사건 확률과 사건 여부를 계산한다.
8. 사건 후보에 대해 5단계 확률과 decoder 결과를 계산한다.
9. 모델 1 확률을 포함해 행동 10개 확률을 계산한다.
10. 한국어 결과 열을 생성한다.
11. `expected_output.parquet`과 key·확률·단계·행동 결과를 비교한다.
12. 완전히 일치하면 `REPRODUCED`, 아니면 `FAILED`와 첫 불일치 원인을 출력한다.

노트북 상단에는 `RUN_TRAINING = False`, `RUN_LOCKED_TEST = False`, `USE_GPU = False`를 고정한다.

### 4.3 샘플 범위

재현 샘플은 train 또는 validation 사람 중 한 명의 짧은 구간을 사용한다. locked-test 사람은 샘플에 포함하지 않는다. 샘플에는 `person_key`, 시간, context, causal 파생변수와 필요한 기준선·STD 입력만 포함하고 truth 라벨은 결과 비교용 감사 열로 분리한다.

## 5. 추론 인터페이스

팀이 노트북 밖에서도 같은 경로를 사용할 수 있도록 다음 공개 인터페이스를 제공한다.

```text
python -m multisensor_ml.kaggle_reproduce \
  --model-root <downloaded-model-dir> \
  --input <sample-or-prepared-parquet> \
  --output <prediction-parquet>
```

출력에는 다음을 포함한다.

- 사건 raw probability와 threshold 결과
- `NO_EVENT`, `LOW`, `MEDIUM`, `HIGH`, `DECREASING`, `RECOVERY`
- OOD 또는 품질 실패 시 `NOT_DECISIONABLE`
- 행동 10개 raw probability
- `STD-A` membership과 개인 기준선 상태
- 한국어 판정·단계·분포상태
- 모델·데이터·feature schema hash

입력 schema, 모델 hash 또는 dependency 계약이 맞지 않으면 추론을 중단한다. 조용한 fallback은 허용하지 않는다.

## 6. 한글 문서

### 6.1 `MODEL_CARD_KO.md`

- 목적과 사용 범위
- 합성데이터와 24/6/6 person split
- 전체 계층형 구조
- 선택 모델, 임계값, validation 결과
- stress 결과와 취약점
- `oracle/sanity`, `candidate`, `NOT VERIFIED` 경계

### 6.2 `KAGGLE_REPRODUCTION_KO.md`

- Dataset과 Model을 notebook Input으로 연결하는 법
- `Run All` 순서와 예상 출력
- 다른 합성 참여자·구간을 선택하는 법
- `REPRODUCED` 확인 기준
- hash, schema, dependency, `.skops` 오류 해결법

### 6.3 `FEATURE_LABEL_GUIDE_KO.md`

- 전역·개인 기준선과 `STD-A`
- 사건 head와 5단계 head
- 5단계 phase 매핑
- 행동 코드 10개
- context와 causal 파생변수
- 입력 금지 truth 열과 누출 방지 원칙

### 6.4 `EXPERIMENT_LESSONS_KO.md`

- 사건·5단계에서 HGB가 선택된 이유
- 행동별 Logistic/HGB 선택이 달랐던 이유
- 행동 예측이 보조지표인 이유
- 노이즈·axis shift·dropout에서 드러난 취약성
- 개인 기준선과 누적 부하를 다음 실험에서 검증할 질문

## 7. 버전과 재현 계약

- 모델 package version은 content hash 기반으로 고정한다.
- 모델 manifest에는 원본 model, baseline, type, feature schema, label set, wheel, `uv.lock` hash를 기록한다.
- Kaggle version note에는 Git commit, source dataset version, model package hash를 기록한다.
- 동일 package hash가 이미 원격에 있으면 새 version을 만들지 않는다.
- 다른 hash로 같은 variation을 덮어쓰지 않고 새 Kaggle version으로 등록한다.
- 예상 출력은 확률 허용오차와 categorical exact match를 함께 검사한다.

## 8. 실패 처리

- 모델 또는 데이터 Input이 하나라도 없으면 즉시 중단한다.
- SHA-256 불일치 시 로드하지 않는다.
- `.skops` unknown type 불일치 시 로드하지 않는다.
- feature 누락·추가·순서 변경 시 중단한다.
- 예상 출력과 다르면 결과를 저장하되 `FAILED`로 표시하고 Kaggle Model version을 승격하지 않는다.
- 네트워크나 Kaggle upload 실패 시 임의 재학습하지 않는다.

## 9. 테스트와 승인 기준

### 로컬

- package manifest와 모든 파일 hash 일치
- `.pkl`, `.pickle`, `.joblib` 없음
- `.skops` unknown type 선언과 실제 목록 일치
- 선택 모델 mapping 완전성
- 행동 코드 10개 모두 존재
- 금지 truth 열 입력 차단
- sample inference 재실행 결과 일치
- 한글 문서 네 개 존재 및 필수 경계 문구 포함
- notebook JSON/metadata 검증
- notebook에 학습·GPU·locked-test 비활성화가 고정됨
- Ruff, mypy, pytest, `uv lock --check`, `git diff --check` 통과

### Kaggle

- 비공개 Model과 `scikitLearn/oracle-sanity-v1` variation 생성
- 원격 파일명·크기·hash readback 일치
- 비공개 재현 notebook `Run All` 성공
- 최종 셀에 `REPRODUCED` 표시
- 모델 페이지와 notebook URL 기록
- 새 학습·GPU·locked-test 실행 없음

## 10. 범위 밖

- 실제 Muse, Polar, Galaxy Watch 원시 신호 입력
- 실제 clock correction과 생리적 lag 보정
- 실제 행동 또는 의료 성능 주장
- Phase 2·3 재학습
- 모델 자동 승격
- 팀 의견 수집 UI 또는 별도 웹 애플리케이션

## 11. 구현 순서

1. package/reproduction 계약 테스트
2. immutable model package builder
3. inference CLI와 sample expected output
4. 한글 문서 네 개
5. Kaggle 재현 notebook
6. 전체 로컬 검증과 package hash 고정
7. 비공개 Kaggle Model 업로드와 원격 readback
8. 비공개 notebook 업로드·CPU 실행·`REPRODUCED` 확인
9. URL과 최종 검증 기록 커밋

## 12. 참고

- [Kaggle CLI Model/Variation 튜토리얼](https://github.com/Kaggle/kaggle-cli/blob/main/docs/tutorials.md)
- [KaggleHub Model 업로드](https://github.com/Kaggle/kagglehub)
