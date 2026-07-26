# 세션 시작 문구 모음

필요한 문구 하나를 그대로 복사해 사용한다. 대괄호 부분만 이번 세션 내용으로 교체한다.

---

# A. 최초 기동 문구

```text
프로젝트에 업로드된 00_PROJECT_INDEX.md부터 08_SESSION_START_PROMPTS.md까지 모두 읽어라.

이 프로젝트는 Galaxy Watch8, Muse S, Polar H10 기반의 합성 시계열 데이터 생성기만 만든다. 앱, 실기기 연결, LLM, GPT, EEGPT, 의료 판정은 범위 밖이다.

다음 순서로 시작해라.
1. 문서 간 모순 또는 아직 확정되지 않은 항목을 표로 정리한다.
2. 모순이 없으면 고정 결정 10개를 한 줄씩 확인한다.
3. 06_CODEX_IMPLEMENTATION_PRD.md의 Goal 1을 구현하기 위한 Codex용 단일 개발 프롬프트를 작성한다.
4. 프롬프트에는 목표, 범위, 파일 구조, 데이터 계약, 테스트, 승인 기준, 범위 밖, 완료 보고 형식을 포함한다.
5. 구현 코드를 직접 작성하지 말고 우선 Codex 전달 프롬프트를 완성한다.

기존 문서에 있는 결정을 다시 질문하지 말고, 정말 막히는 항목만 미결정으로 표시해라.
```

---

# B. 일반 연속 세션 시작

```text
프로젝트 파일과 이전 결정 기록을 다시 읽고 이번 세션을 이어가라.

이번 세션 목표: [목표]
완료 기준: [완료 기준]
현재 구현 단계: [Goal 번호 또는 현재 상태]

먼저 5줄 이내로 다음만 확인해라.
- 현재 단계
- 이미 고정된 관련 계약
- 이번에 수정할 문서·코드 범위
- 차단요인

그 뒤 질문으로 멈추지 말고 문서에 근거한 합리적 기본값으로 산출물을 작성해라. 장비 샘플링률, truth/observed/model_ready 분리, causal feature, 중립 라벨을 변경하지 마라.
```

---

# C. 기술 설계 검토 세션

```text
02_TECHNICAL_ARCHITECTURE.md, 03_FEATURE_AND_LABEL_SPECIFICATION.md, 04_CONFIGURATION_SCHEMA_DRAFTS.md를 기준으로 다음 설계를 검토해라.

검토 대상: [붙여넣기 또는 파일명]

다음 관점만 검사해라.
1. 원인→신호→파생변수→라벨 순서가 유지되는가
2. truth가 model input으로 누출되는가
3. 미래 윈도 누출이 있는가
4. 세 장비의 샘플링률·역할과 충돌하는가
5. 개인 baseline과 day drift가 분리되는가
6. target과 hard negative가 지나치게 쉽게 분리되는가
7. 아티팩트·결측·clock drift가 검증 가능한가
8. chunk 처리와 재현성이 유지되는가

결과는 PASS, 수정 필요, 차단으로 분류하고, 수정이 필요한 경우 Codex가 바로 적용할 수 있는 변경 명세를 작성해라.
```

---

# D. Codex 구현 프롬프트 생성

```text
06_CODEX_IMPLEMENTATION_PRD.md의 [Goal 번호와 이름]만 구현하도록 Codex 프롬프트를 작성해라.

반드시 포함할 항목:
- 목표와 사용자 가치
- 이번 작업의 포함·제외 범위
- 먼저 읽을 문서와 우선순위
- 생성·수정할 예상 파일
- 모듈별 책임과 공개 인터페이스
- 데이터 입력·출력 계약
- 고정 장비 sample rate와 단위
- deterministic seed와 chunk 처리 규칙
- 단위·통합·property·leakage 테스트
- 실제 실행 명령
- 승인 기준
- 완료 보고 형식

금지:
- 앱·실기기 SDK 연결
- GPT/EEGPT
- 의료적 라벨
- 설정 없이 하드코딩
- 테스트를 통과했다고 추정
- 다음 Goal까지 임의 확장

작업을 지나치게 작은 번들로 나누지 말고, 이번 Goal을 독립적으로 검증 가능한 하나의 구현 의뢰로 작성해라.
```

---

# E. Codex 결과 검수 세션

```text
Codex가 제출한 변경 내용과 테스트 결과를 06_CODEX_IMPLEMENTATION_PRD.md의 [Goal 번호] 승인 기준으로 검수해라.

제출 내용:
[Codex 보고 또는 diff/파일]

다음 표를 작성해라.
- 요구사항 ID
- 구현 증거
- 테스트 증거
- 상태: PASS / PARTIAL / FAIL / NOT VERIFIED
- 필요한 수정

특히 다음을 우선 확인해라.
1. 실제 실행 출력이 있는가
2. 동일 seed 재현성이 검증됐는가
3. 샘플링률과 sample count가 맞는가
4. truth/future/person split 누출이 없는가
5. label이 feature threshold에서 생성되지 않는가
6. 범위 밖 앱·모델 기능이 섞이지 않았는가

검증되지 않은 항목을 통과로 처리하지 말고, 수정이 필요하면 Codex용 보완 프롬프트를 하나 작성해라.
```

---

# F. 장비 SDK 계약 업데이트

실제 SDK 문서나 capability 결과를 얻었을 때 사용한다.

```text
다음 실제 SDK capability 또는 문서를 프로젝트의 장비 계약에 반영해라.

장비: [Galaxy Watch8 / Muse S / Polar H10]
자료: [붙여넣기 또는 업로드 파일]

우선 01_EVIDENCE_AND_DEVICE_CONTRACTS.md와 04_CONFIGURATION_SCHEMA_DRAFTS.md의 현재 계약과 비교해라.
결과를 다음으로 구분해라.
- 일치
- 기존 가정 수정 필요
- 장비·펌웨어별 조건부
- 아직 불명확

그 뒤 다음 산출물을 작성해라.
1. 수정할 정확한 문서 항목
2. `device_contract.yaml` 변경안
3. backward compatibility 영향
4. 생성 데이터 schema migration 필요 여부
5. Codex 적용 프롬프트

공식 자료에 없는 샘플링률이나 단위를 추정으로 확정하지 마라.
```

---

# G. ASD 연구 근거 업데이트

```text
다음 ASD 관련 논문을 합성데이터 설계에 사용할 수 있는지 검토해라.

논문 또는 링크: [자료]

다음 순서로 답해라.
1. 연구 설계, 대상자 수, 센서, 상황, 핵심 결과
2. 집단 평균인지 사건 전조인지 구분
3. 이질성·일반화 제한
4. 합성기에 반영 가능한 soft prior
5. 반영하면 안 되는 단정적 규칙
6. 영향을 받는 config parameter와 문서
7. 반대 반응·무반응·hard negative 설계

논문 결과를 ASD 진단 규칙이나 모든 개인의 고정 방향으로 변환하지 마라. 반영 가치가 낮으면 이유와 함께 반영하지 않는 결론을 내려라.
```

---

# H. 피처 추가 세션

```text
새 파생변수 후보를 검토하고 feature registry 변경안을 작성해라.

피처 후보: [이름·설명]

03_FEATURE_AND_LABEL_SPECIFICATION.md 기준으로 다음을 결정해라.
- 원천 stream
- 계산식 또는 알고리즘
- causal window
- 최소 윈도와 coverage
- 단위와 허용범위
- 품질 조건
- 개인 baseline 필요 여부
- 아티팩트 민감성
- label leakage 위험
- 세 장비에서 실제 계산 가능한지

추가가 타당하면 registry YAML, 테스트 케이스, raw-to-feature 재계산 승인 기준, Codex 프롬프트를 작성해라. 의미가 중복되거나 실제 센서로 계산할 수 없으면 제외해라.
```

---

# I. 사건·라벨 추가 세션

```text
다음 패턴을 합성 사건 또는 라벨로 추가할지 설계해라.

패턴 설명: [설명]

먼저 임상명 대신 관찰 가능한 중립 event_type으로 변환해라. 그 뒤 다음을 정의해라.
- context와 target 여부
- phase 구성
- latent factor 변화
- modality별 반응 확률·지연·강도
- 개인 archetype 차이
- 무반응과 반대 반응
- hard negative
- artifact와 구분 방법
- forecast label
- overlap policy
- 검증 기준

label을 특정 피처 임계값으로 만들지 말고 event truth에서 생성하도록 작성해라.
```

---

# J. MVP 생성·검증 세션

```text
현재 구현으로 12명 × 5일 MVP 생성 준비도를 검토해라.

다음 순서로 진행해라.
1. configs/mvp.yaml의 예상 canonical rows, event 수, raw clip 시간, 저장량을 계산한다.
2. memory와 chunk 경계를 검토한다.
3. 실행 전 필수 테스트와 빠른 dry-run을 정한다.
4. 실제 실행 명령을 작성한다.
5. validation report에서 확인할 PASS 기준을 정한다.
6. 실패 시 재개 가능한 checkpoint 정책을 확인한다.

아직 구현되지 않은 기능이 있으면 완료로 가정하지 말고 차단 목록과 Codex 보완 프롬프트를 작성해라.
```

---

# K. 결정 기록 문구

```text
이번 대화에서 승인된 결정만 DECISIONS.md 항목으로 정리해라.

형식:
- Decision ID
- 날짜
- 결정
- 근거
- 영향받는 문서·모듈
- 대안과 기각 이유
- migration 필요 여부
- 미해결 항목

아이디어나 제안 단계의 내용은 승인된 결정으로 기록하지 마라.
```

---

# L. 가장 먼저 Codex에 보낼 권장 문구

아직 저장소가 없다면 다음 문구로 Goal 1 프롬프트를 만든다.

```text
프로젝트 파일 전체를 읽고 06_CODEX_IMPLEMENTATION_PRD.md의 Goal 1 — 실행 가능한 기반과 truth engine을 구현할 Codex 단일 프롬프트를 작성해라. repository scaffold, strict config, deterministic seed tree, population/person/day context, latent factors, event phases, hard negatives, truth tables, manifest, quick profile, unit/property/integration tests까지 포함한다. 신호 파형과 feature 계산은 인터페이스만 정의하고 Goal 2 이후 구현으로 남긴다. 앱·SDK 연결·GPT/EEGPT·의료 라벨은 제외한다.
```
