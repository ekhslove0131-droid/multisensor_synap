# Multisensor Synthetic Timeline Generator

Galaxy Watch8, Muse S, Polar H10 데이터 계약을 따르는 재현 가능한 다중센서
합성 시계열 데이터 생성기다. 현재 구현 범위는 **Goal 1: 실행 가능한 기반과
truth engine**이다.

이 결과는 실험용 합성데이터이며 진단이나 임상 판단에 사용할 수 없다. 모든
산출물에는 `synthetic`, `non_diagnostic`, `not_for_clinical_use` 경계를
기록한다.

## 현재 구현 범위

- Python 3.12 패키지와 `multisensor-synth` CLI
- strict Pydantic v2 설정 및 JSON Schema
- namespace 기반 deterministic child seed
- population, participant baseline, daily context
- 1 Hz 연속 latent factors
- target, hard negative, artifact-only 일정 truth
- `participants`, `daily_context`, `latent_timeline`, `events` Parquet
- deterministic manifest와 truth validation JSON report
- quick, MVP 설정과 unit/property/integration/golden 테스트

`observed`, `model_ready`, native waveform, 아티팩트 파형, clock/transport,
feature, label, split은 아직 구현하지 않는다.

## 설치

Python 3.12와 [uv](https://docs.astral.sh/uv/)가 필요하다.

```bash
uv sync
uv lock --check
```

생성·검증은 네트워크 없이 실행된다. 의존성 설치와 lock 갱신만 패키지
인덱스 접근이 필요할 수 있다.

## 설정 확인과 규모 추정

```bash
uv run multisensor-synth inspect-config --config configs/quick.yaml
uv run multisensor-synth estimate --config configs/quick.yaml
uv run multisensor-synth estimate --config configs/mvp.yaml
```

기본 quick 프로필은 3명 × 6시간이며 latent truth는 정확히 64,800행이다.
MVP 프로필은 12명 × 연속 5일이며 예상 latent truth는 5,184,000행이다.

## Goal 1 생성과 재검증

```bash
uv run multisensor-synth generate --config configs/quick.yaml --truth-only
uv run multisensor-synth validate --run output/<actual_run_id> --scope truth
```

Goal 1 실행은 다음 파일만 생성한다.

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

동일 output root에 같은 run이 이미 있으면 기존 산출물을 덮어쓰지 않고
실패한다. 재현성 비교는 서로 다른 임시 output root에서 logical table hash와
deterministic manifest subset을 비교한다.

## 개발 검증

```bash
uv run ruff check .
uv run mypy src
uv run pytest
```

## 현재 한계

- 장비 waveform과 native-rate clip은 Goal 2 범위다.
- 실제 SDK capability에 따라 Muse PPG/AUX와 Watch 온도 계약을 갱신해야 한다.
- `sensor_artifact_episode`는 Goal 1에서 일정 truth만 만들며 신호를 왜곡하지 않는다.
- 연구 prior는 개인 baseline의 약한 사전정보일 뿐이며 의료적 정답이 아니다.
- 저장소 배포 라이선스는 아직 승인되지 않았으므로 `LICENSE`를 포함하지 않는다.

## 기준 문서

- `docs/project/00_PROJECT_INDEX.md`부터 `08_SESSION_START_PROMPTS.md`
- `docs/planning/DOCUMENT_CONSISTENCY_REVIEW.md`
- `docs/planning/GOAL1_CODEX_PROMPT.md`
