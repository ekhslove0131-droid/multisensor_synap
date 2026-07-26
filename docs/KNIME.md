# KNIME 5.12 사용법

## 역할 분리

Python은 생성 참조, prepare, 학습, 평가, 안전한 bundle을 담당한다.
KNIME은 `scripts/knime_run_goal15.sh`를 External Tool로 실행하고 결과
Parquet/JSON을 읽어 시각화한다. 학습 코드를 KNIME 노드 안에 복제하지 않는다.

## 워크플로 입력

- ML project path: `/Users/baital/dev/multisensor_ml`
- experiment config:
  - quick: `configs/goal15_quick.yaml`
  - full: `configs/goal15_mvp3.yaml`
- run mode: `run-all`, `materialize-synthetic`, `train`

스크립트는 프로젝트 `.venv`와 동일한 `uv.lock`을 `--frozen`으로 사용한다.

## 실행 순서

1. `Multisensor_ML_Goal1_5` 워크플로를 연다.
2. 세 workflow input을 확인한다.
3. full 결과가 이미 있다면 재학습하지 말고 Reader 이후 노드만 실행한다.
4. 새 실험을 실행할 때만 External Tool 노드를 포함해 Run All 한다.
5. `knime/exports/<experiment_id>/dashboard_manifest.json`에 기록된 표를 읽는다.

## 다섯 화면

1. Dataset & Split: run/person/split과 24/6/6 확인
2. Personal Baseline: `personal_weight`, `n_eff`, factor center/MAD 안정화
3. Pattern Performance: probability/phase timeline, PR curve, confusion matrix,
   AUCPR, event recall, false alerts/hour, lead time
4. Noise Stress: scenario별 AUCPR·Brier·ECE·event recall·오경보·degradation
5. Synchronization & Real Audit: oracle와 real 상태를 별도 표시

`real_observed` 행의 accuracy/synchronization이 `NOT VERIFIED`가 아닌 경우는
Goal 1.5 결과로 받아들이면 안 된다.

## 확장

Parquet Reader가 보이지 않으면 KNIME 공식
**Extension for Big Data File Formats**를 설치하고 KNIME을 재시작한다.
기존 `/Users/baital/knime-workspace/KNIME_project`는 변경하지 않는다.
