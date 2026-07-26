# Codex 구현 PRD

## 문서 상태

- 버전: `1.0-draft`
- 기준일: 2026-07-25
- 구현 대상: `multisensor_synth` Python 패키지와 CLI
- 범위: 합성 시계열 데이터 생성·검증·내보내기
- 앱·실기기 연결·의료 판단: 제외

---

# 1. 제품 정의

## 1.1 문제

실제 다중센서 데이터가 충분하지 않은 초기 단계에서 다음을 검증할 데이터가 필요하다.

- 동일 시간축에서 EEG·심장·EDA·움직임을 함께 보는 구조
- 사람마다 다른 표준선과 반응 순서
- 사건 전·중·후 단계 라벨
- 실제 장비에 가까운 노이즈·결측·시간 오차
- 파생변수 계산과 패턴 감지 모델의 데이터 계약

단순 난수 표나 독립 피처 생성으로는 다중센서 시간 관계와 라벨 정합성을 검증할 수 없다.

## 1.2 해결책

규칙·확률·신호처리 기반의 계층적 생성기를 만든다.

```text
population → person → day → latent factor → event → ideal signal
→ device observation → artifact/clock/packet → features → labels → validation
```

## 1.3 핵심 사용자

- 연구·기획 담당자
- Codex를 이용해 구현을 진행하는 개발자
- 향후 패턴 감지 모델 개발자
- 실제 장비 수집기 데이터 계약을 검증하는 개발자

---

# 2. 범위

## 포함

- 설정 기반 가상 참여자 생성
- 5일 공통 타임라인
- Muse S, Polar H10, Galaxy Watch8 계약
- EEG/ECG/RR/EDA/PPG/ACC/temperature 합성
- 개인 baseline, circadian, day drift
- 사건 단계와 hard negative
- 장비별 아티팩트·결측·clock drift
- 1 Hz 정렬 타임라인
- 다중 윈도 파생변수
- 사건·예측·품질 라벨
- Parquet/FIF/JSON/HTML 산출물
- 스키마·통계·누출 검증
- CLI와 테스트

## 제외

- Android, Wear OS, iOS 앱
- 제조사 SDK 실행 또는 기기 연결
- 서버·DB·대시보드
- LLM, GPT, EEGPT
- 의료 진단·치료·임상 성능 주장
- 실제 개인정보 또는 환자 데이터
- 자동 트레이닝·배포 MLOps

---

# 3. 제품 요구사항

## FR-001 설정 로딩

- 엄격한 Pydantic 모델로 YAML을 검증한다.
- 알 수 없는 키, 잘못된 단위, 허용되지 않는 샘플링률은 실패한다.
- `quick`과 `mvp` 기본 설정을 제공한다.
- 실행 전 예상 행 수, native clip 시간, 예상 저장량을 출력한다.

## FR-002 재현 가능한 실행

- 모든 난수는 namespace child seed를 사용한다.
- 동일 code version, config, seed에서 동일 결과를 생성한다.
- manifest에 root/child seed와 dependency version을 기록한다.

## FR-003 Population과 개인 기준선

- 인구집단 prior에서 여러 가상 참여자를 생성한다.
- 사람별 baseline과 response archetype을 5일 동안 유지한다.
- 날짜별 drift와 수면·활동 문맥을 추가한다.
- 연구 prior strength와 반대·무반응 확률을 설정할 수 있다.

## FR-004 사건 엔진

- target event, hard negative, artifact-only event를 생성한다.
- `pre_early`, `pre_late`, `onset`, `peak`, `recovery_early`, `recovery_late`, `post`를 지원한다.
- 단계 길이, 강도, 센서별 지연, modality response를 설정한다.
- label은 사건 truth에서 생성한다.

## FR-005 이상적 생리 신호

- EEG: 1/f aperiodic + band oscillation + burst
- ECG: beat timing + morphology
- RR/HR: beat timing에서 계산
- PPG: ECG와 연결된 pulse waveform
- EDA: tonic + phasic SCR
- ACC: 위치별 orientation/gravity/dynamic motion
- temperature: 느린 body/ambient/contact mixture

신호 간 상관은 latent factor와 사건 엔진으로 만든다.

## FR-006 장비 관측 모델

- Muse S: EEG 256 Hz, head ACC 52 Hz, 선택 PPG 64 Hz
- Polar H10: ECG 130 Hz, HR/RR 1 Hz, ACC 25/50/100/200 Hz
- Galaxy Watch8: ACC 25 Hz, EDA 1 Hz, HR/IBI 1 Hz, PPG 25 Hz, temperature event stream
- Watch on-demand ECG/PPG는 core continuous stream에서 제외한다.
- 실제 capability 차이는 config로 반영한다.

## FR-007 아티팩트와 결측

- modality별 아티팩트 manifest를 먼저 생성한다.
- 실제 관측 신호에 아티팩트를 적용한다.
- random missing, block dropout, channel dropout, packet loss를 구분한다.
- 사건 중 움직임 증가에 따라 artifact 확률이 조건부 증가할 수 있다.

## FR-008 시계·전송

- `truth_time_ns`, `device_time_ns`, `receive_time_ns`, `aligned_time_ns`를 지원한다.
- initial offset, drift ppm, jitter, packet delay, reordering을 생성한다.
- 정렬 후 잔여 오차를 유지한다.

## FR-009 Tier A 전체 타임라인

- 1 Hz canonical table을 참여자 × 날짜 partition으로 작성한다.
- 5일 전체를 메모리에 올리지 않고 chunk 처리한다.
- 장비별 coverage와 quality를 포함한다.

## FR-010 Tier B native clips

- 모든 target event 주변 기본 `-300/+600초` native clip을 생성한다.
- matched baseline, hard negative, artifact-only clip을 생성한다.
- EEG clip은 Parquet와 선택적 MNE FIF를 지원한다.

## FR-011 파생변수

- feature registry에 기반해 계산한다.
- 5/15/30/60/180/300초 causal window를 지원한다.
- 피처별 최소 coverage와 최소 샘플 수를 검증한다.
- 개인 baseline 피처는 과거 데이터만 사용한다.
- invalid feature는 NaN + validity flag로 표현한다.

## FR-012 라벨

- event, phase, time-to-event, forecast horizon, intensity, artifact, validity 라벨을 생성한다.
- target과 context와 artifact를 서로 다른 축으로 유지한다.
- 겹침 규칙을 검증한다.

## FR-013 검증

- schema, range, monotonicity, phase order, rate, sample count
- feature recomputation
- causal-window leakage
- truth column leakage
- split leakage
- target/hard-negative 난이도 sanity check
- statistical summary와 plot

## FR-014 산출물

- `manifest.json`
- truth/observed/model_ready Parquet
- native clips
- split manifest
- JSON/HTML validation report
- dependency lock, license inventory, SBOM

## FR-015 CLI

```bash
multisensor-synth generate --config <path>
multisensor-synth validate --run <run_dir>
multisensor-synth features --run <run_dir>
multisensor-synth inspect-config --config <path>
multisensor-synth estimate --config <path>
```

모든 명령은 명확한 exit code를 반환한다.

---

# 4. 비기능 요구사항

## NFR-001 성능

- `quick` 프로필은 일반 개발 PC에서 CI에 적합해야 한다.
- `mvp`는 GPU 없이 실행 가능해야 한다.
- 시간·메모리 측정은 validation report에 남긴다.
- PyArrow chunk writer를 사용해 메모리 상한을 통제한다.

## NFR-002 재현성

- 멀티프로세싱 여부와 chunk 크기가 결과의 의미를 바꾸지 않아야 한다.
- floating-point 차이가 예상되면 허용 오차를 명시한다.

## NFR-003 유지보수성

- 신호, 장비, 아티팩트, 피처, 라벨 모듈을 분리한다.
- 설정 dict를 코드 전역에서 직접 참조하지 않고 typed model을 주입한다.
- 한 장비 추가가 latent/event core를 수정하지 않도록 한다.

## NFR-004 오프라인 실행

- 데이터 생성과 테스트에 네트워크가 필요하지 않아야 한다.
- 외부 모델·체크포인트 다운로드를 요구하지 않는다.

## NFR-005 라이선스

- core dependency는 승인된 permissive 라이선스만 사용한다.
- 제조사 SDK 파일은 포함하지 않는다.

## NFR-006 안전·표현

- 출력 메타데이터에 `synthetic`, `non_diagnostic`, `not_for_clinical_use`를 기록한다.
- event type은 중립적 명칭을 사용한다.

---

# 5. 데이터 규모 프로필

## Quick

| 항목 | 값 |
|---|---:|
| 참여자 | 3 |
| 기간 | 6시간 |
| target event | 2/명 |
| canonical rows | 64,800 |
| 목적 | CI·개발 |

## MVP

| 항목 | 값 |
|---|---:|
| 참여자 | 12 |
| 기간 | 5일 |
| canonical rows | 5,184,000 |
| target event | 하루 1~4/명 |
| baseline clips | 하루 2/명 |
| hard negative | target과 비슷한 수 |
| 목적 | 시연·모델 파이프라인 |

실제 저장량은 feature 수와 compression에 따라 달라지므로 `estimate` 명령이 실행 전에 계산한다.

---

# 6. 구현 로드맵 — 목표 단위

사용자의 개발 방식에 맞춰 지나치게 작은 번들보다 독립적으로 검증 가능한 목표 단위로 나눈다.

## Goal 1 — 실행 가능한 기반과 truth engine

### 포함

- repository scaffold, `uv`, CLI
- strict config와 schema
- deterministic seed tree
- population/person/day context
- latent factors
- event scheduler, phases, hard negatives
- truth tables와 manifest
- quick profile

### 승인

- `generate --config quick.yaml --truth-only` 성공
- 동일 seed 재현
- event 단계·시간 불변조건 통과
- unit/property tests 통과

## Goal 2 — 장비별 native signal과 관측 모델

### 포함

- EEG/ECG/PPG/EDA/ACC/temperature ideal generators
- Muse/Polar/Watch observation model
- native sample rates와 units
- event effect와 modality lag
- native clip exporter

### 승인

- event와 baseline clip 생성
- sample count·단위·주파수 검증
- 신호 간 관계가 설정된 방향·지연을 통계적으로 반영
- MNE/NeuroKit2 기반 재처리 smoke test 통과

## Goal 3 — 현실성 계층

### 포함

- modality artifacts
- packet loss, block dropout
- device clocks, transport, alignment
- coverage와 quality
- artifact truth manifest

### 승인

- 각 아티팩트에 deterministic test
- artifact-only hard negative 확인
- 정렬 전후 오차 보고
- missingness 구조 검증

## Goal 4 — model-ready features와 labels

### 포함

- 1 Hz canonical table
- feature registry
- 5~300초 causal windows
- online personal baseline
- cross-modal features
- labels와 person split

### 승인

- raw clip 재계산 일치
- 미래 누출·truth 누출 0건
- forecast labels 경계 테스트
- 같은 사람이 split 간 중복되지 않음

## Goal 5 — MVP 실행, 검증, 문서화

### 포함

- 12명 × 5일 chunk generation
- JSON/HTML report
- sanity classifier와 난이도 검사
- license inventory/SBOM
- README/runbook
- benchmark와 저장량 보고

### 승인

- `mvp.yaml` 완주
- 모든 필수 파일 존재
- validation PASS
- 테스트와 lint/type check PASS
- no-network 재실행 가능

## Goal 6 — 실제 데이터 이후 선택 확장

- 실제 장비 통계 calibration importer
- real-vs-synthetic evaluator
- optional SynthCity experiments
- TSTR/TRTS

Goal 6은 현재 구현 완료 조건에 포함하지 않는다.

---

# 7. 테스트 전략

## Unit

- 분포 범위와 상관
- phase curve
- signal primitive
- artifact injection
- clock equation
- feature function
- label boundary

## Property-based

- 모든 설정 허용범위에서 timestamp 단조성
- phase order 불변
- sample rate와 count 관계
- 같은 seed 동일 출력
- 다른 seed에서 충분한 다양성
- invalid config는 항상 거부

## Integration

- quick end-to-end
- target + hard negative + artifact overlap
- native clip → alignment → feature → label
- Parquet round-trip
- FIF export/read

## Golden

작은 1명·10분 fixture를 고정한다.

- manifest hash
- 주요 통계
- event table
- 일부 파형 구간

dependency 업데이트 시 golden 변경을 명시적으로 검토한다.

## Leakage

- feature schema의 금지 접두어 검사
- centered rolling 금지
- 미래 timestamp 접근 검사
- split person overlap 검사
- label module의 feature dependency 검사

## Statistical

- 개인별 baseline 차이
- 일중 변화
- event 전후 효과 분포
- response archetype 분리
- hard negative overlap
- artifact와 quality 관계
- 결측률

---

# 8. Sanity model 정책

목적은 높은 성능 달성이 아니라 데이터 오류 탐지다.

허용 모델:

- Logistic Regression
- Random Forest 또는 HistGradientBoosting
- Isolation Forest

검사 예:

- target AUC가 1.0에 지나치게 가까우면 누출 또는 지나치게 쉬운 규칙 경고
- AUC가 0.5 근처이고 event effect가 설정돼 있으면 생성 관계 실패 경고
- person ID 하나만으로 높은 성능이면 사건 빈도 편향 경고
- artifact-only subset에서 false positive 증가 여부 보고

이 결과를 실제 ASD 예측 성능으로 해석하지 않는다.

---

# 9. 수용·승인 기준 체크리스트

## 기능

- [ ] `quick` 한 명령 생성
- [ ] `mvp` 한 명령 생성
- [ ] truth/observed/model_ready 분리
- [ ] 세 장비 계약 준수
- [ ] target/hard negative/artifact-only 생성
- [ ] 1 Hz 전체 타임라인
- [ ] native-rate clips
- [ ] 다중 윈도 피처
- [ ] 단계·forecast·품질 라벨
- [ ] person split

## 정확성

- [ ] deterministic seed
- [ ] phase order
- [ ] sample rate/count
- [ ] raw→feature recomputation
- [ ] clock/align report
- [ ] no truth leakage
- [ ] no future leakage

## 품질

- [ ] pytest PASS
- [ ] property tests PASS
- [ ] ruff PASS
- [ ] mypy PASS 또는 승인된 예외
- [ ] validation report PASS
- [ ] README와 CLI help
- [ ] licenses/SBOM

---

# 10. Codex 작업 보고 형식

각 목표 완료 시 Codex는 다음 순서로 보고한다.

```markdown
## 완료 목표

## 변경 파일

## 구현한 데이터 계약

## 실행 명령과 실제 결과

## 테스트 결과

## 생성 산출물 예시

## 남은 위험·가정

## 다음 목표의 명확한 시작점
```

“완료”라고 말하기 전에 실제 명령과 테스트 출력을 확인한다.

---

# 11. Definition of Done

다음 명령이 새 환경에서 성공해야 한다.

```bash
uv sync --frozen
uv run ruff check .
uv run mypy src
uv run pytest
uv run multisensor-synth generate --config configs/quick.yaml
uv run multisensor-synth validate --run output/<quick_run_id>
uv run multisensor-synth estimate --config configs/mvp.yaml
```

최종 완료 단계에서는 `mvp.yaml`도 실제 생성·검증한다. 실행 환경 제약으로 완주하지 못했다면 완료로 표현하지 않고, 생성된 범위와 미검증 범위를 명시한다.
