# KNIME 합성데이터 공장·학습 레지스트리 사용법

이 문서의 두 워크플로는 합성 truth를 이용한 `oracle/sanity` 검증용입니다.
실제 행동 예측 성능이나 의료 성능을 의미하지 않으며, 실제 데이터 성능은
`NOT VERIFIED`, 장비 동기화는 `NOT_AVAILABLE_TRUTH_ONLY`로 표시합니다.

## 가져올 워크플로

KNIME 5.12의 Local Space에서 다음 파일을 각각 import합니다.

- `knime/Multisensor_Synthetic_Factory_Goal1_5.knwf`
- `knime/Multisensor_ML_Training_Registry.knwf`

기존 `Multisensor_ML_Goal1_5.knwf`는 Oracle Benchmark로 보존합니다.

## 1. 합성데이터 공장

`Multisensor_Synthetic_Factory_Goal1_5`를 열고 **Run all**을 실행합니다.
입력 설정은 quick 기본값으로 시작하며, 실행 결과는 다음 순서로 실제
receipt가 연결됩니다.

```text
입력 설정
→ 합성 생성과 outcome 라벨
→ 실행 상태
→ 참여자와 기간
→ 기준선 생성 QA
→ 사건 5단계 timeline
→ 행동 라벨 조합
→ target·hard negative 비교
→ 분할·manifest
→ 장비·동기화 상태
```

Table View를 선택하면 아래쪽 표에서 결과를 볼 수 있습니다. 모든 결과
화면은 열 패턴 `*`로 저장되어 있어 해당 표의 전체 열을 표시합니다.

## 2. 학습 레지스트리

먼저 공장 실행이 만든 receipt가 있어야 합니다.
`Multisensor_ML_Training_Registry`를 열고 **Run all**을 실행하면 다음
receipt가 앞 단계 output에서 다음 단계 input으로 전달됩니다.

```text
라벨 QA
→ 전역·개인 기준선
→ STD 타입·OOD
→ 모델 1 사건·5단계
→ 모델 2 행동 다중 라벨
→ validation·stress·locked test
→ 예측 감사
→ 한국어 결과 라우터
```

중간 receipt의 hash 또는 선행 단계가 맞지 않으면 다음 단계가 중단됩니다.
이미 같은 manifest hash로 완료된 단계는 `REUSED`로 재사용하며 결과를
덮어쓰지 않습니다.

## 결과 해석

- 모델 1은 `NO_EVENT` 여부를 먼저 판단하고, 사건 후보에 대해
  `LOW/MEDIUM/HIGH/DECREASING/RECOVERY` 확률을 냅니다.
- 모델 2는 사건별 관찰 행동 코드를 다중 라벨로 예측합니다.
- train의 모델 1 확률은 사람 단위 OOF 예측이므로 모델 2에 정답 누출을
  일으키지 않습니다.
- 라벨 수가 부족한 행동은 정확도를 만들지 않고
  `INSUFFICIENT_LABEL_SUPPORT`로 남습니다.
- OOD·품질 실패·장시간 결측은 한국어 결과에서 `판단 보류`입니다.
- 릴리스는 기본적으로 `candidate`이며, validation 결과와 감사 사유를
  확인한 뒤 CLI의 수동 `promote`로만 `champion`이 됩니다.

## CLI에서 같은 실행 재현

```bash
uv run multisensor-ml factory run \
  --config configs/factory_quick.yaml \
  --output-receipt artifacts/receipts/factory_quick.json

uv run multisensor-ml registry run-all \
  --config configs/training_registry_quick.yaml
```

36명×5일 전체 검증은 `factory_mvp3.yaml`과
`training_registry_mvp3.yaml`을 사용합니다.

## GUI 검증 증거

- `docs/evidence/knime_goal15_factory_connected_executed.jpg`
- `docs/evidence/knime_goal15_registry_connected_executed.jpg`
- `docs/evidence/knime_goal15_korean_router_columns.jpg`

2026-07-27 KNIME 5.12에서 두 워크플로를 Local Space에 import하고
**Run all**한 뒤 저장했습니다. 파일 readback 기준 합성 공장 19개 노드,
학습 레지스트리 22개 노드가 모두 `EXECUTED`였고, 기존
`KNIME_project/.knimeLock`은 삭제하거나 변경하지 않았습니다.
