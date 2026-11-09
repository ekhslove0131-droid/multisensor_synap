# Phase06f consumer 호환성 · 2026-09-05

범위: 내부 descriptive feature 계약, raw-free projection consumer, 독립 계산 parity.
학습/production admission/생리 stress target 승인 증거가 아니다. fit_call_count=0.
기존 standard 24 / behavior 26 view 및 legacy feature hashes는 변경하지 않았다.

## 수락한 producer schema

| source | width | canonical SHA256 |
|---|---:|---|
| Watch | 3 | 4e949a5033e36b9c2f0ec3d348c52088717a0904dc774a923941f468061bc6fb |
| H10 | 3 | 055bfcee108dd6e059ddd930eb682ef43760c92bc62ef68cb004953a2ae92ae2 |
| Muse | 24 | 97139686fab24c88ed844ba2d69ea81384b76711a7217a954c994e4eaa334ce9 |

이 값은 domain 필드를 포함한 최종 로컬 draft다. 이전 domain 없는 dcefc797… /
dc8609fc… / 37d481fe… 문서는 더 이상 수락하지 않는다. 배포/임상 승인 상태는 아니다.
POSITIVE는 >0, NONNEGATIVE는 >=0, UNIT_INTERVAL은 0<=x<=1이며 SIGNED_FINITE는
유한 부호값이다. 모든 domain에서 bool/NaN/Inf를 거부하고 clamp로 위반을 숨기지 않는다.
required/optional descriptor 모두 같은 검사에 따른다. schema/domain 제거·변경은 hash로 거부한다.

모델 contracts/physiology/v1은 위 producer 문서의 JSON 의미를 보존한다.
canonical 규칙은 UTF-8, ensure_ascii=false, sort_keys=true, separators=(',', ':'),
allow_nan=false이다. descriptor 순서·unit·policy를 포함한 전체 문서를 해시로 고정한다.

초기 Cloud 파일 검사에서 세 JSON의 wave_min_coverage=1과 producer의 1.0이 달랐다.
Python JSON canonicalization은 이 차이를 보존하므로 다음 파일 해시는 수락하지 않는다:

- Watch: 8e6bc838cfb4dcad4d9714968dbc7987377cd84e788b6dd08006af42f3862188
- H10: a7dcf80931553d2fee478006beaa71a8c676211c0397d040c6296b89517a2e49
- Muse: bd3132c1dbc0c9108a7a4e3a4837ef550df3292fc8c530e837820e1ccaeaacda

모델 snapshot은 전달된 hash와 실제 schema_document()에 맞춰 1.0을 사용한다.
consumer가 입력을 1→1.0으로 몰래 변환하지 않는다.
Cloud 수정 후 2026-09-05 실제 파일을 다시 읽어 세 source 모두 Cloud 파일·모델 snapshot·
schema_document()의 canonical bytes exact equality를 확인했다.
BLOCKED_SCHEMA_SERIALIZATION_MISMATCH는 해소됐다. 이후 domain 추가로 위 세 해시를 갱신했다.
수정 파일을 load_schema()에 직접 전달하고 Cloud fixture 독립 수식 parity도 다시 통과했다.

domain 보정 RED: 위반 7사례가 DID NOT RAISE로 실패, 허용 6사례 통과.
GREEN: 신규/관련 테스트 140 passed, scoped Ruff/mypy 통과. domain 포함 실제 Cloud 파일,
모델 snapshot, producer canonical bytes exact equality와 수식 parity를 다시 확인했다.

## 독립 계산과 실제 Cloud fixture 교차 실행

모델 계산은 Cloud physiology_math를 import/복사하지 않는다. Watch OLS/median,
H10 인접 valid segment+sequence pair RMSSD, NumPy batch FFT Welch를 독립 구현했다.
Cloud produce_physiology() 결과와 모델 값을 같은 Cloud unit fixture에 비교했다.

- Watch: [64.5, 5.5, 1.0], 최대 절대 차이 0.
- H10: [72.0, 805.0, 14.142135623730951], 최대 절대 차이 0.
- Muse: 각 채널 alpha=1, entropy=0.16779149288809658,
  24-vector 최대 절대 차이 4.5110965389355465e-32 (<1e-12).
- RR segment/sequence 단절과 invalid interval 양옆 연결 금지 회귀를 포함한다.
- EEG 상수/NaN/불완전 10초 입력 거부와 offset/amplitude 불변성을 검증한다.

## consumer 범위

### Cloud 담당 작업의 최종 교차검증 receipt

2026-09-05 작업 `01a0709b-6dbd-70b2-8bfa-c200bf7305ce`의 전달 증거다.
이번 기록 갱신에서 테스트나 Cloud 실행을 다시 수행하지 않았다.
Cloud produce_physiology → 메모리 create-only artifact → read_projection → 모델
load_schema/consume_descriptors의 전체 경로에서 세 source 모두 MATCHED로 보고됐다.
모델 snapshot과 producer canonical bytes/hash도 exact equality이며 위 최종 해시를 유지한다.
독립 계산 차이는 Watch/H10 0, Muse 4.5110965389355465e-32다.

Cloud canonical JSON은 ensure_ascii=False UTF-8이고, 읽기 projection은 정확한
code/unit/value/readiness/reason 5필드다. required/domain은 schema에서 읽으며
private artifact 내부의 required는 보존한다.
Cloud 검증 보고: 관련 36, 전체 Inference 352, Analytics 971, Ruff F/diff PASS.
origin=CONTRACT_FIXTURE, training_ready=false, fit_call_count=0이다.
이 메모리 artifact 교차검증은 production raw adapter·실제 cohort·배포 증거가 아니다.

load_schema()는 pinned 전체 hash만 수락한다. consume_descriptors()는 내부 result의
code/unit/value/readiness/reason 5필드 raw-free projection만 받는다. 새 wire envelope가 아니다.
unknown/추가/private field, duplicate, 단위 불일치, AVAILABLE null/nonfinite를 거부한다.
core 누락/INSUFFICIENT는 None이며 optional 부재는 core 폭을 늘리거나 막지 않는다.

math helper는 **검증된 normalized signal**의 수식 parity용이다. raw timestamp·sampling·
contact·clock·manifest lineage 검증을 대체하지 않는다. runtime raw adapter parity,
NO_PATTERN admission, 양수 cohort/receipt, target/scale 승인은 여전히 필요하다.
10초 support는 계산 정책이며 충분한 baseline 학습시간을 뜻하지 않는다.
세 singleton family이며 7개 availability subset을 7개 모델로 해석하지 않는다.
BLOCKED_NO_REAL_COHORT / NOT EVALUABLE 상태를 유지한다.
