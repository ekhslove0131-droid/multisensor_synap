# Kaggle 전체 계층형 모델 재현 결과

## 최종 상태

- Kaggle Model: [multisensor-goal15-hierarchical v5](https://www.kaggle.com/models/bjcoding/multisensor-goal15-hierarchical/scikitLearn/oracle-sanity-v1/5) — 비공개
- Kaggle Notebook: [multisensor-goal15-hierarchical-reproduction v9](https://www.kaggle.com/code/bjcoding/multisensor-goal15-hierarchical-reproduction) — `COMPLETE`
- 재현 결과: `REPRODUCED`, validation 표본 600행, 불일치 열·키 없음
- 실행 장치: CPU, 새 학습 `false`, locked test 읽기 `false`
- Model payload SHA-256: `9d5627214c25154b23a678a4e5e74b36553de507bbd4e79f96540a3ea8a8d758`
- Dataset manifest SHA-256: `698b3a0f8368da8e7a32e0bf4757dd49dd910c3515d1367419ecbd926283d4ac`
- 해석 범위: 합성 `oracle/sanity`; 실제 데이터 성능은 `NOT VERIFIED`

## 무엇을 재현했나

하나의 모델 파일이 아니라 다음 계층 전체를 안전한 `.skops` 묶음으로 재현했다.

1. 전역·개인 기준선과 개인 보정
2. 의미를 붙이지 않은 표준 타입 `STD-A`와 OOD 상태
3. 사건 확률과 사건 여부
4. `LOW → MEDIUM → HIGH → DECREASING → RECOVERY` 5단계
5. 관찰 가능한 행동 코드 10개의 개별 확률
6. `판단 보류`를 포함한 한국어 결과 라우터

행동 확률은 사건·단계 판단을 보조하는 지표다. 멜트다운·감각추구 같은 해석 태그를 직접 예측 타깃으로 사용하지 않는다.

## 팀 사용 방법

1. Kaggle Model v5와 기존 비공개 synthetic Dataset을 Notebook에 연결한다.
2. Notebook v9을 `Copy & Edit`한 뒤 CPU, Internet Off 상태를 유지한다.
3. `Run All`을 실행한다. 노트북은 Model 안의 고정 wheelhouse를 격리 경로에 설치한다.
4. 마지막 출력에서 `status=REPRODUCED`, `compared_rows=600`, `run_training=false`, `run_locked_test=false`를 확인한다.
5. 새 데이터 적용 시에는 모델의 feature schema와 동일한 causal 파생변수를 제공해야 한다. truth·hidden archetype·event intensity는 입력할 수 없다.

세부 사용법은 Model에 함께 올라간 `MODEL_CARD_KO.md`, `KAGGLE_REPRODUCTION_KO.md`, `FEATURE_LABEL_GUIDE_KO.md`, `EXPERIMENT_LESSONS_KO.md`를 따른다.

## 주요 시행착오와 해결

- Kaggle Dataset이 `prepared/manifest.json` 구조를 평탄화해 `prepared__manifest.json`으로 제공했다. 탐색기를 두 구조 모두 인식하도록 수정했다.
- Internet Off 환경에 `skops`가 없어 실패했다. Linux Python 3.12 의존 wheel 전체를 Model에 포함했다.
- Kaggle 커널에 직접 설치하자 이미 로드된 NumPy와 충돌했다. `/kaggle/working/multisensor_runtime` 격리 설치 후 새 Python 프로세스에서만 추론하도록 바꿨다.
- Kaggle Model은 `model_payload.tar.gz`를 자동으로 `model_payload/`로 확장했다. 두 형식을 모두 검증하고 결정적 archive를 재구성하도록 했다.
- v6·v7에서 macOS와 Linux 행동 확률이 달랐다. 진단 v8 결과 최대 차이는 `2.682209e-7`이었고 사건 확률 차이는 `5.55e-17`, 범주·단계·OOD·한국어 판정은 600행 모두 같았다. 이에 확률만 절대오차 `1e-6`으로 비교하고 범주 출력은 완전 일치하도록 고정했다. `0.01` 차이는 계속 실패한다.
- Model 버전 생성 직후 목록과 다운로드 API의 전파 시점이 달라 404가 발생했다. 새 버전 번호와 실제 다운로드 가능 상태를 각각 재시도한 뒤 해시를 검사하도록 업로더를 보강했다.
- Notebook 출력에 격리 런타임 전체가 포함되어 다운로드가 불필요하게 커졌다. 최종 판정은 Kaggle 로그의 영수증과 먼저 내려온 `comparison.json`을 교차 확인했다. 다음 개정에서는 런타임 폴더를 Notebook output에서 제외한다.

## 운영 DB 방향

현재 Kaggle 재현 패키지는 파일 기반으로 고정한다. 실제 데이터 운영 단계에서는 역할을 다음처럼 나누는 것이 합리적이다.

- 원시 고주파 센서: 변경 불가 Parquet와 object storage
- PostgreSQL: 참여자·세션·관찰 라벨·label set·모델 버전·승격·감사 기록
- TimescaleDB hypertable: corrected UTC 기준의 동기화된 파생 시계열, 품질·누적 부하·예측 확률·단계 결과

원시 ECG·EEG 전체를 처음부터 DB에 복제하지 않는다. PostgreSQL 기본 파티셔닝과 Timescale hypertable/chunk 정책은 실제 보존 기간·쿼리 패턴을 측정한 뒤 정한다. 참고: [Timescale hypertables](https://docs.timescale.com/use-timescale/latest/hypertables/), [PostgreSQL partitioning](https://www.postgresql.org/docs/current/ddl-partitioning.html).

## 다음 단계

1. 동일 인터페이스로 실제 장비 데이터의 corrected UTC·품질·동기화 보고서를 입력한다.
2. 실제 관찰 라벨을 train/validation 계약으로 등록하되 기존 synthetic locked test와 혼합하지 않는다.
3. 개인 기준선 통과율, 사건·5단계 성능, 행동 보조지표 순서로 평가한다.
4. 여러 날·여러 session에서 반복된 OOD 또는 성능 저하만 재학습 후보로 기록하고 자동 승격하지 않는다.
