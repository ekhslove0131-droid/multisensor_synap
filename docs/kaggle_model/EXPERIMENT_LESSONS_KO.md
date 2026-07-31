# 1차 합성 계층형 모델 시행착오와 주요 인사이트

## 목적과 경계

이 기록은 최초 합성데이터 `oracle/sanity` 실험을 팀이 이해하기 위한 문서입니다. 실제 행동·웨어러블 정확도는 `NOT VERIFIED`이며 의료 성능을 뜻하지 않습니다.

## 무엇이 주요했는가

사건과 5단계에서는 비선형 상호작용을 다루는 HistGradientBoosting이 validation에서 선택됐습니다. 다만 긴 시간 기억을 직접 학습한 것이 아니라 causal rolling feature와 상태 decoder가 시간 구조를 보완했습니다. 따라서 HGB 단독 성과로 해석하면 안 됩니다.

행동 10개는 한 모델이 모두 우세하지 않았습니다. 라벨 빈도와 feature 관계가 행동마다 달라 Logistic과 HGB 선택이 섞였습니다. 행동은 개인·환경 편차가 크므로 사건과 단계 뒤의 보조지표로 유지합니다.

## 시행착오

- KNIME 최초 화면은 결과 라우터였고 학습·버전 연결이 부족해 Factory와 Registry receipt 구조를 추가했습니다.
- STD 타입을 여러 개 만들려 했지만 안정성 조건을 통과하지 않아 근거 없이 유형을 강제하지 않고 `STD-A` 하나로 복귀했습니다.
- Kaggle 데이터에서 audit ID dtype, dataset/day/session identity, phase 허용 열 문제가 순차적으로 드러나 입력 feature와 감사 열 계약을 분리했습니다.
- event recall과 false alert가 좋아 보이면서 row F1이 매우 낮은 모순이 생겨 1차 지표를 과장하지 않고 2차에서 segment-aware 평가를 먼저 고쳤습니다.
- Gaussian noise, 축 이동, latent dropout에서 성능이 크게 떨어져 실제 노이즈 내성을 주장하지 않았습니다.

## 다음 아이디어

개인 기준선과 누적 부하는 유력하지만 3차 최종 학습이 완료되지 않았으므로 1차 모델에 성능 향상으로 합쳐 말하지 않습니다. 향후 운영 저장소는 PostgreSQL에 라벨·세션·모델 버전·감사를, TimescaleDB에 동기화된 시간축 파생변수·예측을 두고, 고주파 원본은 immutable Parquet/object storage로 분리하는 방향을 검토합니다.

채택 기준은 validation에서 AUCPR·event F1·person-macro 성능이 개선되고 false alerts/hour와 ECE가 과도하게 악화되지 않는 것입니다. 실제 라벨 검토 전 자동 재학습·자동 승격은 하지 않습니다.
