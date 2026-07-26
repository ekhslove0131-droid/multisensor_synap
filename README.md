# Multisensor Synthetic Timeline Generator

Galaxy Watch8, Muse S, Polar H10 데이터 계약을 따르는 재현 가능한 다중센서 합성 시계열 데이터 생성기 프로젝트다.

현재 저장소 상태는 **Goal 1 구현 전 문서·개발 프롬프트 준비 단계**다. 구현 코드는 아직 작성하지 않았다.

## 고정 범위

- Python 기반 합성데이터 생성기와 검증 도구만 개발
- 공통 1 Hz, 참여자당 기본 5일
- `truth / observed / model_ready` 계층 분리
- 사건 주변 native-rate clip 병행
- 중립 사건명과 `synthetic`, `non_diagnostic`, `not_for_clinical_use` 경계 유지

앱, 실기기 연결, 서버, LLM/GPT/EEGPT, ASD 진단·의료 판정은 범위 밖이다.

## 문서

- 기준 문서: `docs/project/00_PROJECT_INDEX.md`부터 `08_SESSION_START_PROMPTS.md`
- 문서 검토: `docs/planning/DOCUMENT_CONSISTENCY_REVIEW.md`
- Goal 1 Codex 프롬프트: `docs/planning/GOAL1_CODEX_PROMPT.md`

## 다음 단계

`docs/planning/GOAL1_CODEX_PROMPT.md` 전체를 Codex에 전달해 Goal 1만 구현한다.
