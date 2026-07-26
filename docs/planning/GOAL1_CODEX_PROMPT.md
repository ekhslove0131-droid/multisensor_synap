# Codex 단일 개발 프롬프트 — Goal 1

아래 블록 전체를 한 번에 Codex에 전달한다.

---

## 역할과 작업 원칙

당신은 `multisensor_synth` 저장소의 Goal 1 구현 담당자다. 이번 작업은 **실행 가능한 Python 기반과 truth engine을 한 번에 구현하고 실제 실행으로 검증하는 것**이다.

작업을 작은 번들로 다시 쪼개거나 다음 Goal로 확장하지 마라. 기존 저장소의 사용자 변경을 보존하고, 현재 파일 상태와 Git 상태를 먼저 읽은 뒤 구현하라. 구현을 완료했다고 말하기 전에 아래 명령과 테스트를 실제로 실행하라.

기술 설명과 완료 보고는 한국어로 작성하고, 파일명·코드·식별자는 영어를 사용하라.

## 1. 이번 목표와 사용자 가치

`06_CODEX_IMPLEMENTATION_PRD.md`의 **Goal 1 — 실행 가능한 기반과 truth engine**만 구현한다.

사용자 가치는 다음과 같다.

- Galaxy Watch8, Muse S, Polar H10의 후속 신호 생성이 올라갈 재현 가능한 기반을 확보한다.
- 동일 절대 시간축에서 여러 가상 참여자의 개인 baseline, day context, latent factor, target/hard-negative 사건 일정을 생성한다.
- label의 원인이 되는 hidden truth를 피처나 관측값과 분리한다.
- 같은 code version·config·seed에서 같은 truth 결과를 재생성하고 불변조건으로 검증한다.
- quick 프로필을 CI와 다음 Goal의 독립적인 기준 fixture로 사용한다.

## 2. 먼저 읽을 문서와 우선순위

작업 전에 다음 파일을 **모두** 읽어라.

1. `docs/project/00_PROJECT_INDEX.md`
2. `docs/project/01_EVIDENCE_AND_DEVICE_CONTRACTS.md`
3. `docs/project/02_TECHNICAL_ARCHITECTURE.md`
4. `docs/project/03_FEATURE_AND_LABEL_SPECIFICATION.md`
5. `docs/project/04_CONFIGURATION_SCHEMA_DRAFTS.md`
6. `docs/project/05_OPEN_SOURCE_STACK_AND_LICENSES.md`
7. `docs/project/06_CODEX_IMPLEMENTATION_PRD.md`
8. `docs/project/07_GPT_PROJECT_INSTRUCTIONS.md`
9. `docs/project/08_SESSION_START_PROMPTS.md`
10. `docs/planning/DOCUMENT_CONSISTENCY_REVIEW.md`

충돌 시 우선순위는 다음과 같다.

1. 사용자가 제공한 실제 SDK 문서와 capability 결과
2. 제조사 공식 최신 문서
3. `00_PROJECT_INDEX.md`의 고정 결정
4. `02_TECHNICAL_ARCHITECTURE.md`와 `04_CONFIGURATION_SCHEMA_DRAFTS.md`
5. `03_FEATURE_AND_LABEL_SPECIFICATION.md`
6. `06_CODEX_IMPLEMENTATION_PRD.md`
7. 논문과 오픈소스 문서

`DOCUMENT_CONSISTENCY_REVIEW.md`의 해석을 이번 Goal의 적용 계약으로 사용하되, 원문 장비 샘플링률·채널·단위·라벨 의미를 조용히 변경하지 마라. 구현을 막는 새 충돌을 발견하면 임의로 확정하지 말고 완료 보고의 `남은 위험·가정`에 파일·절·영향을 명시하라.

## 3. 포함 범위

다음을 모두 구현한다.

- Python 3.12 패키지 scaffold와 `uv`
- `pyproject.toml`, `uv.lock`, CLI entry point
- strict Pydantic v2 config와 JSON Schema export
- `quick.yaml`, `mvp.yaml` 기본 설정
- namespace 기반 deterministic child seed tree
- population prior와 participant profile
- 개인 baseline과 response archetype
- day/session context, circadian·sleep·activity 문맥
- 연속 latent factors
- target event scheduler
- phase scheduler와 phase curve
- hard negative와 artifact-only event **일정 truth**
- truth Parquet tables
- config snapshot과 deterministic manifest
- truth-level validation과 JSON report
- quick profile truth-only end-to-end 실행
- unit, property, integration, golden tests
- README의 Goal 1 실행·검증 절차

Goal 1의 `sensor_artifact_episode`는 **일정과 truth 분류만** 만든다. 실제 파형 아티팩트, packet loss, clock model, quality score는 구현하지 않는다.

## 4. 범위 밖

다음을 구현하거나 의존성으로 우회 추가하지 마라.

- EEG, ECG, RR, PPG, EDA, ACC, temperature waveform 생성
- Muse/Polar/Watch device observation model
- native-rate clip 생성 또는 FIF export
- 아티팩트 파형 주입, packet loss, clock drift, transport, alignment
- observed 데이터
- model-ready 1 Hz table
- feature registry 계산, 개인 online baseline, cross-modal feature
- label/forecast table과 person split
- sanity classifier
- Android, Wear OS, iOS 앱과 UI
- 실기기 SDK 연결, SDK binary, 비밀키
- 서버, DB, 클라우드, 계정
- GPT, LLM, EEGPT, EEG foundation model
- ASD 진단·질환 판정·치료 추천·임상 성능 주장
- Goal 2 이후 기능
- 승인되지 않은 BSL/GPL/AGPL dependency

신호·장비·feature 모듈의 빈 placeholder 파일을 대량 생성하지 마라. Goal 2가 사용할 최소 Protocol이나 domain type은 공개 경계를 위해 필요한 경우에만 정의하고, 미구현 기능이 작동하는 것처럼 보이게 하지 마라.

프로젝트 저장소 자체의 배포 라이선스는 아직 승인되지 않았다. dependency 라이선스 정책과 혼동해 `LICENSE`를 임의로 선택하거나 생성하지 마라.

## 5. 생성·수정할 파일 구조

기존 문서 파일은 수정하지 않는다. 최소한 다음 구조를 만든다. 더 작은 helper 파일은 책임 분리가 명확할 때만 추가할 수 있다.

```text
multisensor-synth/
├── pyproject.toml
├── uv.lock
├── README.md
├── configs/
│   ├── quick.yaml
│   ├── mvp.yaml
│   └── schemas/
│       └── run.schema.json
├── src/multisensor_synth/
│   ├── __init__.py
│   ├── cli.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── loader.py
│   │   └── validators.py
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── types.py
│   │   ├── contracts.py
│   │   └── seeds.py
│   ├── population/
│   │   ├── __init__.py
│   │   ├── generator.py
│   │   ├── baselines.py
│   │   └── circadian.py
│   ├── latent/
│   │   ├── __init__.py
│   │   ├── factors.py
│   │   ├── daily_context.py
│   │   └── state_process.py
│   ├── events/
│   │   ├── __init__.py
│   │   ├── scheduler.py
│   │   ├── phases.py
│   │   ├── archetypes.py
│   │   └── hard_negatives.py
│   ├── export/
│   │   ├── __init__.py
│   │   ├── parquet.py
│   │   └── manifest.py
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── schema.py
│   │   ├── invariants.py
│   │   └── report.py
│   └── orchestration/
│       ├── __init__.py
│       └── pipeline.py
└── tests/
    ├── unit/
    ├── property/
    ├── integration/
    └── golden/
```

### 모듈 책임

- `config/models.py`: `extra="forbid"`, frozen strict typed models. `dict[str, Any]`로 schema를 우회하지 않는다.
- `config/loader.py`: YAML load, 명시적 `extends`, path resolution, final config snapshot 생성.
- `config/validators.py`: sample rate, unit, probability, range, phase duration, fraction 합계와 cross-field validation.
- `domain/seeds.py`: root seed와 stable namespace를 결합한 child seed 생성. Python의 process-randomized `hash()`를 사용하지 않는다.
- `population/*`: population prior → participant baseline → day drift 순서와 상관된 participant parameter 생성.
- `latent/*`: 1 Hz truth timeline의 context와 bounded latent factor 생성.
- `events/*`: target, hard negative, artifact-only schedule과 실현된 phase boundary 생성.
- `export/parquet.py`: stable schema·column order·row order·partition을 가진 truth table writer.
- `export/manifest.py`: config hash, version, root/child seed, provenance, 안전 경계 기록.
- `validation/*`: schema, timestamp, row count, phase order, event overlap, range, deterministic truth 검증.
- `orchestration/pipeline.py`: config → seed tree → population → context → latent → event → export → validation 순서를 조정.
- `cli.py`: 명령, exit code, 오류 메시지, 예상 규모 출력.

## 6. 변경하면 안 되는 계약

- 패키지명: `multisensor_synth`
- CLI command: `multisensor-synth`
- Python: `>=3.12,<3.13`
- 기본 공통 timeline: UTC 1 Hz
- MVP: 12명 × 연속 5일
- quick: 3명 × 6시간 × target event 2개/명
- target event: `multimodal_arousal_episode`
- phase 순서: `pre_early → pre_late → onset → peak → recovery_early → recovery_late → post`
- 중립 사건명만 사용
- `truth / observed / model_ready` 경계
- label 원인은 hidden event truth이며 feature threshold가 아님
- 연구 근거는 soft prior이고 개인별 반대·무반응을 허용
- 모든 RNG는 namespace child seed 사용
- 출력 메타데이터: `synthetic=true`, `non_diagnostic=true`, `not_for_clinical_use=true`
- core dependency는 MIT/BSD/Apache/ISC/PSF 계열 우선
- 제조사 SDK 파일·비밀키를 저장소에 포함하지 않음

장비 config 검증값은 다음을 유지한다. Goal 1에서는 파형을 만들지 않지만 잘못된 계약은 거부해야 한다.

| 장비·stream | 계약 |
|---|---|
| Muse S EEG | 256 Hz, `[TP9, AF7, AF8, TP10]`, µV |
| Muse S head ACC | 52 Hz, XYZ, ±4 g |
| Muse S PPG | 64 Hz, 기본 disabled |
| Polar H10 ECG | 130 Hz, µV |
| Polar H10 HR/RR | 1 Hz, RR ms |
| Polar H10 chest ACC | `[25,50,100,200]` Hz 중 하나, `[2,4,8]` g 중 하나 |
| Watch8 wrist ACC | 25 Hz, XYZ |
| Watch8 EDA | 1 Hz, µS |
| Watch8 HR/IBI | 1 Hz |
| Watch8 PPG | 25 Hz, green/IR/red |
| Watch8 temperature | event-driven, synthetic MVP observation 1 Hz |
| Watch on-demand ECG/PPG | core continuous 제외, `calibration_only`만 허용 |

## 7. 입력 설정 계약

`configs/quick.yaml`과 `configs/mvp.yaml`은 `04_CONFIGURATION_SCHEMA_DRAFTS.md`의 통합 YAML 구조를 따른다.

필수 top-level key:

```text
schema_version
project
profile
run
storage
devices
population
context
events
artifacts
features
labels
splits
validation
```

Goal 1 실행이 사용하지 않는 후속 섹션도 typed schema로 검증한다. 다만 그 값을 소비하거나 후속 산출물을 생성하지 않는다.

필수 validation:

- 알 수 없는 key 거부
- 확률은 `[0,1]`
- 모든 `{min,max}`는 `min <= max`
- sample rate·channel·unit은 장비 hard contract와 일치
- participant count·duration·chunk는 양수
- `canonical_rate_hz == 1`
- target phase duration은 0 이상이며 target의 필수 phase는 유효
- response archetype probability 합은 허용 오차 내 1
- hard-negative type probability 합은 허용 오차 내 1
- split fraction 합은 허용 오차 내 1
- `run.start_time_utc`는 timezone-aware UTC
- `storage.write_full_native_streams=false`가 기본
- Watch on-demand는 `calibration_only` 외 거부하고 두 종류 동시 활성 거부

`skin_temperature.units`는 현재 문서 YAML과 호환되도록 읽되 내부 typed representation은 singular `unit`으로 정규화한다. config snapshot에는 사용자가 제공한 원본과 정규화된 resolved config를 구분해 보존한다.

## 8. deterministic seed 계약

root seed 하나를 순차 소비하는 구조를 금지한다.

최소 namespace:

```text
population
person/{person_id}/baseline
person/{person_id}/day/{day_index}
person/{person_id}/latent/{factor_name}
person/{person_id}/event/{event_id}
person/{person_id}/hard_negative/{event_id}
person/{person_id}/artifact_only/{event_id}
split
```

요구사항:

- `numpy.random.SeedSequence`와 stable cryptographic namespace digest를 결합한다.
- Python `hash()`와 module-global RNG를 사용하지 않는다.
- child seed는 호출 순서와 chunk 크기에 독립적이어야 한다.
- 새 namespace를 추가해도 기존 namespace 결과가 바뀌지 않아야 한다.
- root와 실제 사용 child seed 또는 재구성 가능한 spawn key를 manifest에 기록한다.
- 같은 seed는 동일 결과, 다른 seed는 충분한 다양성을 만들어야 한다.

## 9. truth 생성 순서와 의미

반드시 다음 인과 순서를 유지한다.

```text
population prior
→ participant baseline
→ day/session drift and context
→ continuous latent factors
→ event schedule and realized phases
→ truth tables and manifest
```

사건 label이나 phase를 latent factor 임계값으로 역산하지 않는다. 사건 scheduler가 event truth를 먼저 정하고, latent timeline은 그 사건의 영향을 받을 수 있다.

### Latent factor

모든 factor는 `float32`, 기본 범위 `[0,1]`이다.

- `autonomic_arousal`
- `motor_activation`
- `cognitive_load`
- `sleep_pressure`
- `sensory_context`
- `recovery_capacity`
- `social_context`

정상 구간도 평평하지 않아야 한다. 느린 추세, 일중 변화, 사람별 baseline이 존재해야 한다. factor process는 deterministic이고 chunk 경계에 독립적이어야 한다.

### Context state

허용값:

```text
sleep
wake_rest
sedentary_activity
light_activity
moderate_activity
focused_task
meal_context
transition
```

모든 참여자는 동일한 절대 run start/end를 공유하지만 수면·식사·활동 일정은 사람별로 다를 수 있다.

### Event type

최소 생성:

- target: `multimodal_arousal_episode`
- hard negative: `ordinary_physical_activity`
- hard negative: `quiet_cognitive_load`
- hard negative schedule: `sensor_artifact_episode`
- hard negative: `recovery_without_peak`
- hard negative: `false_alarm_like_episode`

`repetitive_motion_episode`, `sleep_transition`은 config로 활성화할 수 있으나 quick 승인에 필수는 아니다.

target event끼리 peak가 겹치지 않게 하고 `minimum_gap_between_target_sec`를 지킨다. context와 hard negative는 target과 겹칠 수 있다. artifact-only는 생리 waveform을 만들지 않고 일정 truth만 기록한다.

### Phase

허용 순서:

```text
pre_early
pre_late
onset
peak
recovery_early
recovery_late
post
```

일부 phase는 config·archetype에 따라 생략 가능하다. 생략한 phase는 음수 duration이나 역전 timestamp로 표현하지 말고 저장하지 않는다. target에는 `onset`이 반드시 존재한다.

## 10. 출력 디렉터리와 파일 계약

Goal 1 truth-only 실행은 다음만 만든다.

```text
output/<run_id>/
├── manifest.json
├── config_snapshot/
│   ├── source.yaml
│   └── resolved.yaml
├── truth/
│   ├── participants.parquet
│   ├── daily_context.parquet
│   ├── latent_timeline.parquet
│   └── events.parquet
└── reports/
    └── truth_validation_report.json
```

`observed/`, `model_ready/`, `clips/`는 Goal 1 truth-only에서 생성하지 않는다.

### 공통 규칙

- timestamp는 UTC nanosecond `int64`
- 모든 table은 `run_id`, `person_id`를 포함
- stable column order와 deterministic row sort
- `person_id` 형식: `P001`, `P002`, …
- `day_index`: 0-based
- Parquet partition 여부와 무관하게 logical content는 같아야 함
- truth 파일에는 `synthetic`, `non_diagnostic`, `not_for_clinical_use` 경계를 schema metadata 또는 manifest로 명시

### `truth/participants.parquet`

최소 컬럼:

| 컬럼 | 타입 | 규칙 |
|---|---|---|
| `run_id` | string | 모든 파일과 동일 |
| `person_id` | string | run 내 unique |
| `age_years` | float32 | config 범위 |
| `biological_sex` | string | config category |
| `response_archetype` | string | 허용 archetype |
| `resting_hr_bpm_truth` | float32 | 개인 baseline |
| `rmssd_ms_truth` | float32 | 양수 |
| `eeg_aperiodic_exponent_truth` | float32 | config 범위 |
| `eeg_relative_alpha_truth` | float32 | `[0,1]` |
| `eeg_relative_gamma_truth` | float32 | `[0,1]` |
| `eda_tonic_level_us_truth` | float32 | 양수 |
| `recovery_capacity_truth` | float32 | `[0,1]` |
| `opposite_response_tendency_truth` | float32 | `[0,1]` |
| `no_response_tendency_truth` | float32 | `[0,1]` |
| `seed` | uint64 | participant baseline child seed |

truth 값이라는 점을 명확히 하기 위해 개인 생리 parameter 컬럼에 `_truth` suffix를 사용한다.

### `truth/daily_context.parquet`

최소 grain: `person_id × day_index` 한 행.

| 컬럼 | 타입 |
|---|---|
| `run_id` | string |
| `person_id` | string |
| `day_index` | int8 |
| `date_utc` | date32 |
| `sleep_start_time_ns` | nullable int64 |
| `sleep_end_time_ns` | nullable int64 |
| `sleep_duration_sec` | int32 |
| `fatigue_drift_truth` | float32 |
| `activity_drift_truth` | float32 |
| `baseline_drift_truth` | float32 |
| `sensor_tolerance_truth` | float32 |
| `seed` | uint64 |

quick 6시간이 하루 일부만 포함해도 `day_index=0` 행을 만든다. run 범위 밖 수면 boundary는 null을 허용한다.

### `truth/latent_timeline.parquet`

grain: 참여자별 canonical 1 Hz. quick의 총 행 수는 정확히 `3 × 6 × 3600 = 64,800`이다.

| 컬럼 | 타입 |
|---|---|
| `run_id` | string |
| `person_id` | string |
| `truth_time_ns` | int64 |
| `timestamp_utc` | UTC timestamp |
| `date_utc` | date32 |
| `day_index` | int8 |
| `seconds_from_start` | int64 |
| `context_state` | string/category |
| `is_awake` | bool |
| `autonomic_arousal` | float32 |
| `motor_activation` | float32 |
| `cognitive_load` | float32 |
| `sleep_pressure` | float32 |
| `sensory_context` | float32 |
| `recovery_capacity` | float32 |
| `social_context` | float32 |
| `active_target_event_id` | nullable string |
| `active_target_event_phase` | nullable string |
| `active_hard_negative_event_ids` | list<string> |
| `active_hard_negative_types` | list<string> |
| `artifact_only_schedule_ids` | list<string> |

`truth_time_ns`는 각 참여자에서 정확히 1초 간격으로 단조 증가한다. 첫 timestamp는 `run.start_time_utc`, 마지막은 `start + duration - 1 second`다.
target, context, hard negative, artifact-only schedule을 하나의 state 열로 압축하지 않는다. 겹침이 가능한 축은 별도 컬럼으로 유지하고, 같은 축의 복수 사건은 list 값으로 보존한다.

### `truth/events.parquet`

한 행은 하나의 event instance다. 기존 draft의 필수 컬럼을 제거하지 말고 phase 검증을 위해 명시적 boundary를 추가한다.

| 컬럼 | 타입 |
|---|---|
| `run_id` | string |
| `person_id` | string |
| `event_id` | string |
| `event_type` | string |
| `is_target` | bool |
| `hard_negative_kind` | nullable string |
| `start_time_ns` | int64 |
| `pre_late_start_time_ns` | nullable int64 |
| `onset_time_ns` | nullable int64 |
| `peak_start_time_ns` | nullable int64 |
| `peak_end_time_ns` | nullable int64 |
| `recovery_early_end_time_ns` | nullable int64 |
| `recovery_late_end_time_ns` | nullable int64 |
| `end_time_ns` | int64 |
| `intensity_truth` | float32 |
| `archetype` | string |
| `is_artifact_only_schedule` | bool |
| `seed` | uint64 |

`start_time_ns`는 첫 실현 phase의 시작이고 `end_time_ns`는 마지막 실현 phase의 끝이다. 생략된 phase boundary는 null이다. target event에는 `onset_time_ns`, `peak_start_time_ns`, `peak_end_time_ns`가 필수다.

### `manifest.json`

최소 필드:

```json
{
  "schema_version": "1.0",
  "generator_version": "0.1.0",
  "run_id": "...",
  "profile": "quick",
  "seed": 20260725,
  "config_sha256": "...",
  "git_commit": "... or null",
  "created_at_utc": "...",
  "deterministic": true,
  "synthetic": true,
  "non_diagnostic": true,
  "not_for_clinical_use": true,
  "implemented_goal": 1,
  "implemented_layers": ["truth"],
  "deferred_layers": ["observed", "model_ready"],
  "child_seed_namespaces": {},
  "tables": {},
  "dependency_versions": {}
}
```

`run_id`는 config hash·root seed·generator version을 기반으로 안정적으로 생성한다. `created_at_utc`, wall-clock duration, absolute output path처럼 실행마다 변할 수 있는 값은 deterministic content comparator에서 제외한다.

## 11. CLI 계약

Goal 1에서 실제 구현할 명령:

```bash
multisensor-synth inspect-config --config <path>
multisensor-synth estimate --config <path>
multisensor-synth generate --config <path> --truth-only
multisensor-synth validate --run <run_dir> --scope truth
```

요구사항:

- `inspect-config`: resolved config와 validation 결과 출력
- `estimate`: participant, duration, canonical truth row, 예상 event 범위를 출력
- `generate`: 실행 전에 estimate 요약, 출력 후 run path와 validation PASS/FAIL 출력
- `validate`: 기존 run의 truth table을 다시 읽어 검증
- 성공 exit code 0
- config/validation 실패 exit code 2
- 예상하지 못한 실행 실패는 nonzero이며 traceback을 숨기지 않음
- Goal 2 이후 명령을 성공한 것처럼 stub 처리하지 않음

## 12. 테스트 요구사항

### Unit

- strict config load와 unknown key 거부
- invalid rate, unit, probability, range, fraction 거부
- stable namespace child seed
- namespace 호출 순서 독립성
- distribution bounds
- response archetype 선택
- circadian·day drift 범위
- phase duration과 phase curve
- event ID uniqueness
- target peak non-overlap
- hard-negative type 선택
- manifest serialization

### Property-based

Hypothesis를 사용해 다음을 검증한다.

- 허용 config에서 timestamp가 항상 단조 증가
- 1 Hz sample count 공식이 항상 성립
- phase boundary가 역전되지 않음
- target onset/peak 필수 boundary 존재
- 동일 seed·namespace 입력은 동일 출력
- 다른 seed에서 최소 하나 이상의 profile/event 결과가 달라짐
- participant 수와 duration이 바뀌어도 person ID와 시간 범위가 유효
- invalid config는 항상 거부

### Integration

- quick truth-only end-to-end
- 정확히 3명, 6시간, 64,800 latent rows
- 각 참여자 target event 정확히 2개
- target, hard negative, artifact-only schedule 존재
- 모든 참여자가 동일한 절대 timeline start/end 공유
- Parquet round-trip 후 schema·row order 유지
- config snapshot과 manifest 존재
- truth validation report PASS
- `observed/`, `model_ready/`, `clips/` 미생성
- 네트워크 없이 실행 가능

### Golden

1명·10분의 작은 config를 고정한다.

- resolved config hash
- participants table
- events table
- latent factor 요약 통계
- deterministic logical-content hash

dependency 변경으로 golden이 바뀌면 자동 덮어쓰지 말고 검토가 필요하도록 실패시킨다.

### Leakage·경계

Goal 1에는 feature/model-ready가 없지만 다음 경계를 테스트한다.

- truth-only run이 observed/model_ready를 만들지 않음
- event scheduler가 feature 모듈을 import하지 않음
- 의료적 사건명이 event type에 없음
- manifest에 세 안전 경계가 모두 true
- SDK binary·credential·network call이 없음

## 13. 재현성 승인 방식

동일 checkout에서 같은 quick config를 서로 다른 임시 output root에 두 번 생성한다.

비교 대상:

- resolved config hash
- participants logical table
- daily context logical table
- latent timeline logical table
- events logical table
- child seed namespace map
- deterministic manifest subset

비교 제외:

- `created_at_utc`
- wall-clock duration
- absolute path
- 파일시스템 metadata

같은 환경에서는 비교 대상이 정확히 같아야 한다. Parquet의 바이트 동일성만으로 재현성을 판정하지 말고, stable schema·sort 후 logical content도 비교한다. 플랫폼 간 floating point 허용 오차가 필요하면 근거와 수치를 테스트에 명시한다.

## 14. 실제 실행 명령

저장소 root에서 다음을 실제 실행한다.

```bash
uv sync
uv lock --check
uv run ruff check .
uv run mypy src
uv run pytest
uv run multisensor-synth inspect-config --config configs/quick.yaml
uv run multisensor-synth estimate --config configs/quick.yaml
uv run multisensor-synth generate --config configs/quick.yaml --truth-only
uv run multisensor-synth validate --run output/<actual_quick_run_id> --scope truth
uv run multisensor-synth estimate --config configs/mvp.yaml
```

그 뒤 깨끗한 환경 조건을 확인하기 위해 가능하면 다음도 실행한다.

```bash
uv sync --frozen
uv run pytest
```

명령이 환경 제약으로 실행되지 않으면 성공으로 추정하지 말고 `NOT VERIFIED`로 보고한다. `<actual_quick_run_id>`는 실제 생성 결과로 바꾼다.

## 15. 승인 기준

다음이 모두 충족돼야 Goal 1 완료다.

- `generate --config configs/quick.yaml --truth-only` 성공
- quick: 3명·6시간·64,800 latent rows
- 각 참여자 target event 2개
- target, hard negative, artifact-only schedule 존재
- 동일 code/config/seed deterministic logical content 일치
- event phase 순서·timestamp monotonicity·범위 불변조건 PASS
- strict config와 장비 hard-contract validation PASS
- `truth/participants.parquet` 등 필수 파일과 manifest 존재
- `observed/`, `model_ready/`, `clips/` 미생성
- unit/property/integration/golden tests PASS
- `ruff` PASS
- `mypy src` PASS 또는 구체적으로 승인 가능한 예외가 보고됨
- test와 generation이 네트워크를 요구하지 않음
- 출력에 `synthetic`, `non_diagnostic`, `not_for_clinical_use` 포함
- Goal 2 이후 코드가 섞이지 않음
- README에 설치·실행·검증·현재 한계가 기록됨

하나라도 실제 검증하지 못하면 Goal 1을 완료로 표현하지 마라.

## 16. 완료 보고 형식

다음 순서를 그대로 사용한다.

```markdown
## 완료 목표

## 변경 파일

## 구현한 데이터 계약

## 실행 명령과 실제 결과

| 명령 | exit code | 핵심 출력 | 상태 |
|---|---:|---|---|

## 테스트 결과

| 구분 | 통과 | 실패 | 미검증 |
|---|---:|---:|---:|

## 생성 산출물 예시

## 승인 기준 체크

| 승인 항목 | 증거 | 상태: PASS/PARTIAL/FAIL/NOT VERIFIED |
|---|---|---|

## 남은 위험·가정

## 다음 목표의 명확한 시작점
```

보고 원칙:

- 실제 실행한 명령과 exit code만 기록한다.
- 실패·skip·warning을 숨기지 않는다.
- 생성된 run ID, row count, event count, report path를 적는다.
- 문서와 다른 새 가정을 만들었다면 영향과 migration 필요성을 적는다.
- 다음 목표는 Goal 2의 시작점만 설명하고 구현하지 않는다.

---

이상 범위로 Goal 1을 구현하고 실제 검증 결과를 보고하라.
