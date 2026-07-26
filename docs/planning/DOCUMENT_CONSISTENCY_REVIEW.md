# 프로젝트 문서 일관성 검토

- 검토일: 2026-07-25
- 검토 범위: `00_PROJECT_INDEX.md`부터 `08_SESSION_START_PROMPTS.md`
- 현재 단계: Goal 1 구현 전
- 결론: **Goal 1을 막는 문서 모순은 없다.** 아래 항목은 우선순위로 해석하거나 후속 Goal 전에 확정해야 한다.

## 모순·미확정 항목

| ID | 항목 | 관련 문서 | 판정 | Goal 1 적용 |
|---|---|---|---|---|
| DOC-001 | 개인 truth 파일명이 `truth/participants.parquet`와 `truth/person_profiles.parquet`로 다름 | 02, 03 | 문서 간 명칭 불일치 | 문서 우선순위에 따라 `truth/participants.parquet`를 정식 파일명으로 사용한다. |
| DOC-002 | 00의 “1초·5초·30초…”와 03/04의 `[5,15,30,60,180,300]` 피처 윈도 표현이 다름 | 00, 03, 04 | 해상도와 rolling window가 섞여 표현됨 | 1초는 `timeline_1hz` 관측 해상도다. rolling feature window는 `[5,15,30,60,180,300]`으로 해석한다. Goal 1에는 feature 계산이 없다. |
| DOC-003 | 02 출력 예시에 `features_15s.parquet`가 빠져 있으나 03/04에는 15초 윈도가 있음 | 02, 03, 04 | 후속 산출물 목록 누락 | Goal 4 전에 출력 목록을 동기화한다. Goal 1 비차단. |
| DOC-004 | Watch 온도는 공식적으로 event-driven이며 실제 callback 빈도와 별도 ambient 값 노출 여부가 확정되지 않음 | 01, 03, 04 | 실제 SDK capability 미확정 | MVP의 1 Hz 온도 관측은 `simulation_choice`로 유지한다. 실제 온도 signal 구현은 Goal 2 전 capability 결과로 확정한다. |
| DOC-005 | Muse S 세대·SDK가 PPG/AUX/기타 스트림을 실제 노출하는지 미확정 | 01, 04 | 실제 SDK capability 미확정 | Goal 1은 기본 Muse 계약을 config validation에만 사용한다. native signal 구현 전 실제 SDK 문서로 갱신한다. |
| DOC-006 | `truth_time_ns`를 observed native 파일에 둘지, truth 전용 추적 정보로만 둘지 표현이 다소 모호함 | 02, 04 | 후속 schema 경계 미확정 | Goal 1은 truth-only라 영향 없다. Goal 2 전에 observed 기본 schema와 검증용 trace schema를 분리해 확정한다. |
| DOC-007 | `sensor_artifact_episode` schedule은 Goal 1 hard negative에 포함되지만 실제 artifact 주입은 Goal 3 범위임 | 01, 04, 06 | 단계 간 책임 경계 | Goal 1에서는 사건 일정과 truth 분류만 만든다. 파형 왜곡·artifact manifest·quality 계산은 구현하지 않는다. |
| DOC-008 | 재현성 요구와 manifest의 `created_at_utc` 같은 실행 시각 필드가 함께 존재함 | 00, 04, 06 | 비교 대상 정의 필요 | 동일 환경의 deterministic truth table·seed tree·config hash를 비교한다. 실행 시각 같은 volatile provenance는 재현성 비교에서 제외한다. |
| DOC-009 | 사건 단계는 일부 생략 가능하지만 생략된 phase의 저장 표현이 상세히 고정되지 않음 | 02, 03, 04 | truth schema 세부 미확정 | Goal 1은 실현된 phase만 시간 순서로 저장하며, 생략 phase는 만들지 않는다. 필수 `onset`은 모든 target event에 존재해야 한다. |
| DOC-010 | `skin_temperature` 설정에서 `units`, 다른 스트림에서 `unit`을 사용함 | 04 | strict schema 필드명 불일치 가능성 | 제공된 YAML 호환을 위해 Goal 1에서는 temperature에 `units`를 허용하되 내부 정규화는 `unit`으로 한다. 문서 수정은 별도 승인 후 수행한다. |
| DOC-011 | 저장소 자체의 배포 라이선스는 정해지지 않았고 dependency 허용 정책만 정해짐 | 05, 06 | 프로젝트 라이선스 미결정 | Goal 1 구현에는 비차단이다. Codex가 임의로 `LICENSE`를 선택하거나 생성하지 않게 한다. |

## Goal 1 고정 결정 10개

1. 이 저장소는 합성 시계열 데이터 생성기와 검증 도구만 만든다.
2. 고정 장비는 Galaxy Watch8, Muse S, Polar H10이며 각각 Watch EDA·손목 신호, Muse EEG, H10 ECG/RR/HRV가 핵심 역할이다.
3. 공통 정렬 시간축은 UTC 1 Hz이고 MVP 기본 기간은 참여자당 연속 5일이다.
4. MVP 기본 참여자는 12명이며 quick 프로필은 3명·6시간·target 2개/명이다.
5. 전체 기간 1 Hz Tier A와 사건·네거티브 주변 native-rate Tier B를 분리한다.
6. 데이터는 `truth / observed / model_ready` 3계층으로 분리하고 truth를 모델 입력에 섞지 않는다.
7. 생성 원인은 `population → person → day/session drift → latent factor → event phase` 순서를 따른다.
8. 라벨은 hidden event timeline에서 만들고 피처 임계값으로 만들지 않는다.
9. 모든 난수는 namespace child seed를 사용하고 대용량 처리는 participant × hour chunk와 Parquet partition을 사용한다.
10. Python 3.12와 permissive 핵심 의존성을 사용하며 앱·실기기 SDK·LLM/GPT/EEGPT·의료 판정은 구현하지 않는다.

## Goal 1 차단 여부

- 차단 항목: 없음
- Goal 1에서 사용할 정식 개인 truth 파일명: `truth/participants.parquet`
- 후속 문서 정리가 필요한 항목: DOC-003, DOC-004, DOC-005, DOC-006, DOC-010, DOC-011
- 새 장비·샘플링률·의료 라벨 결정: 없음
