# Multisensor Synthetic Timeline Generator

> Galaxy Watch8 + Muse S + Polar H10 기반 다중센서 합성 시계열 데이터 생성 프로젝트

- 문서 기준일: **2026-07-25**
- 프로젝트 유형: **합성데이터 생성기**
- 앱·실시간 알림·의료 판정 개발: **범위 밖**
- 생성형 언어모델·EEGPT·기타 EEG 파운데이션 모델: **사용하지 않음**
- 기본 구현 언어: **Python 3.12**
- 패키지 작업명: `multisensor_synth`

---

## 1. 프로젝트 목표

동일한 5일 타임라인 위에 여러 가상 참여자의 다음 데이터를 생성한다.

1. **Muse S**: EEG 중심 신호와 머리 움직임
2. **Polar H10**: ECG, RR/IBI, 심박, 흉부 움직임
3. **Galaxy Watch8**: EDA, 손목 움직임, PPG, 피부·주변 온도, 보조 심박/IBI
4. 개인마다 다른 표준선과 일중 변동
5. 특정 사건 전후의 `pre → onset → peak → recovery` 패턴
6. 실제 센서처럼 보이는 노이즈, 시간 오차, 결측, 접촉 불량
7. 원시 신호에서 다시 계산되는 파생변수
8. 모델 학습용 시계열 라벨과 미래 예측 라벨

이 프로젝트의 결과물은 향후 패턴 감지 모델과 시연을 위한 **실험용 합성데이터**다. 실제 ASD 진단, 임상 판정, 위험 행동 예측 성능을 증명하지 않는다.

---

## 2. 고정 결정

| 항목 | 결정 |
|---|---|
| 대상 장비 | Galaxy Watch8, Muse S, Polar H10 |
| 핵심 심장 신호 | Polar H10 ECG/RR |
| 핵심 EEG 신호 | Muse S 4채널 EEG |
| 핵심 EDA | Galaxy Watch8 EDA |
| 핵심 움직임 | Watch 손목 ACC + Muse 머리 ACC + H10 흉부 ACC |
| 공통 타임라인 | UTC 기준 1초 간격 `canonical_time` |
| 기본 기간 | 참여자당 연속 5일 |
| 기본 참여자 수 | 12명, 설정으로 변경 가능 |
| 원시 데이터 정책 | 전 기간 1 Hz 정렬 데이터 + 사건/하드 네거티브 주변 native-rate clip |
| 라벨 근거 | 숨겨진 사건 엔진의 시간축에서 생성 |
| 파생변수 | 원시·관측 신호에서 재계산 |
| 개인 기준선 | 인구집단 prior → 개인 baseline → 일별 drift → 순간 상태 |
| 의료적 표현 | 진단명 대신 관찰 가능한 중립 패턴명 사용 |
| 생성 모델 | 규칙·확률·신호처리 기반. GPT/EEGPT 미사용 |
| 오픈소스 정책 | MIT/BSD/Apache 계열을 핵심 경로에 우선 |

---

## 3. 핵심 아키텍처 한 줄

```text
population prior
→ person baseline
→ day/session drift
→ latent factors + event phases
→ ideal physiological signals
→ device observation model
→ artifact + packet loss + clock drift
→ aligned 1 Hz timeline
→ derived features
→ labels + validation report
```

### 데이터 계층

```text
truth/       숨겨진 원인, 정확한 사건 시점, 이상적 생리값
observed/    장비별 원시 관측값, 장비 시간, 노이즈와 결측
model_ready/ 정렬 시계열, 파생변수, 품질 피처, 학습용 라벨
```

`truth/` 열은 검증과 정답 생성에만 사용하며 일반 모델 입력에서 제외한다.

---

## 4. 문서 읽기 순서

| 순서 | 파일 | 역할 |
|---:|---|---|
| 1 | `00_PROJECT_INDEX.md` | 범위, 고정 결정, 문서 지도 |
| 2 | `01_EVIDENCE_AND_DEVICE_CONTRACTS.md` | 장비 제약과 ASD 관련 연구를 어떻게 사용할지 |
| 3 | `02_TECHNICAL_ARCHITECTURE.md` | 전체 생성 파이프라인과 모듈 구조 |
| 4 | `03_FEATURE_AND_LABEL_SPECIFICATION.md` | 피처, 윈도, 라벨, 누출 방지 규칙 |
| 5 | `04_CONFIGURATION_SCHEMA_DRAFTS.md` | YAML 설정 계약과 컬럼 스키마 초안 |
| 6 | `05_OPEN_SOURCE_STACK_AND_LICENSES.md` | 오픈소스 역할, 라이선스, 사용 제한 |
| 7 | `06_CODEX_IMPLEMENTATION_PRD.md` | Codex 구현 요구사항, Epic, 승인 기준 |
| 8 | `07_GPT_PROJECT_INSTRUCTIONS.md` | GPT 프로젝트 설정에 붙여 넣을 지침 |
| 9 | `08_SESSION_START_PROMPTS.md` | 첫 세션·연속 세션·Codex 전달 문구 |

---

## 5. 정보 우선순위

충돌이 생기면 아래 순서로 판단한다.

1. 사용자가 보유한 **실제 SDK 패키지·문서·capability 결과**
2. 제조사 공식 최신 문서
3. 이 프로젝트의 승인된 결정과 설정 스키마
4. 원 논문·메타분석
5. 오픈소스 문서
6. 일반적인 추정

연구 논문은 신호 방향과 효과 크기를 그대로 복사하는 명세가 아니다. 합성 파라미터의 후보 범위와 실험 시나리오를 정하는 **약한 사전정보**로만 사용한다.

---

## 6. 반드시 분리할 세 가지

### 6.1 합성 정답

- 사건 시작·끝·단계
- 숨겨진 강도
- 숨겨진 상태 변수
- 실제로 주입한 아티팩트 종류

### 6.2 관측 신호

- 센서가 측정했다고 가정한 값
- 측정 오차와 시간 지연 포함
- 장비별 결측·접촉 불량 포함

### 6.3 모델 입력과 라벨

- 과거 데이터만 사용해 계산한 파생변수
- 개인 기준선 이탈값
- 단계·사건·예측 시간 라벨
- 품질과 유효성 표시

정답에서 직접 계산한 수치를 모델 입력에 섞지 않는다.

---

## 7. 기본 생성 프로필

### `quick` 프로필

CI와 개발 확인용이다.

- 3명
- 6시간
- 사건 2개/명
- 1 Hz 정렬 테이블
- 사건 주변 native-rate clip

### `mvp` 프로필

시연·모델 파이프라인용이다.

- 12명
- 5일
- 하루 1~4개 target event
- 하루 2개 matched baseline clip
- target event 수와 비슷한 hard negative
- 1초, 5초, 30초, 60초, 180초, 300초 관점의 피처

### `full_raw` 프로필

전 구간 고주파 원시 신호 생성이다. 저장량과 실행량이 크므로 기본값에서 제외한다.

---

## 8. 프로젝트 비목표

다음 항목은 이 저장소에 구현하지 않는다.

- Android/Wear OS/iOS 앱
- Galaxy Watch, Muse, Polar 실기기 연결 코드
- 실시간 경고 UI
- 사용자 계정·클라우드·대시보드
- 의료 진단 또는 치료 추천
- ASD 여부 분류
- 실제 아동 데이터 수집
- LLM 기반 신호 생성 또는 해석
- EEGPT 임베딩 또는 파운데이션 모델

향후 실기기 수집기가 만들어져도 이 프로젝트는 **동일한 데이터 계약을 출력하는 독립 합성기**로 유지한다.

---

## 9. GPT 프로젝트에 넣는 방법

1. 새 ChatGPT 프로젝트를 만든다.
2. 가능하면 프로젝트 전용 메모리를 선택한다.
3. 이 폴더의 `00`~`08` 파일을 모두 업로드한다.
4. `07_GPT_PROJECT_INSTRUCTIONS.md`의 복사용 블록을 프로젝트 지침에 붙여 넣는다.
5. 첫 채팅에서 `08_SESSION_START_PROMPTS.md`의 **A. 최초 기동 문구**를 사용한다.
6. Codex 개발을 시작할 때는 `06_CODEX_IMPLEMENTATION_PRD.md`와 **D. Codex 구현 프롬프트 생성 문구**를 사용한다.

---

## 10. 완료 정의

프로젝트가 완료됐다고 말하려면 최소한 다음이 충족되어야 한다.

- 한 명령으로 `quick`과 `mvp` 데이터셋 생성
- 동일 seed에서 바이트 또는 허용 오차 내 재현
- 장비별 native rate와 단위 계약 준수
- 개인별 기준선과 일별 drift 확인
- target event, hard negative, artifact-only event 생성
- 파생변수를 원시 신호에서 재계산해 일치 검증
- 라벨이 숨겨진 사건 타임라인과 정확히 일치
- 사람 단위 train/validation/test split manifest 생성
- 데이터 누출 검사 통과
- JSON 및 HTML 검증 보고서 생성
- 오픈소스 라이선스 목록과 lock 파일 생성
- 테스트가 네트워크 없이 통과
