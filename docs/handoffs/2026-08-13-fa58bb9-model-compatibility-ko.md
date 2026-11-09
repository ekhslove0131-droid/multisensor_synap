# Cloud fa58bb9 ↔ 모델 validator 호환성 결과

> **SUPERSEDED:** 이 문서는 당시 platform description hash `d8314b0c...`를 runtime
> schema hash로 잘못 사용한 역사적 결과다. 현재 유효 계약은 Watch-only
> `watch_standard_16_v2` / `2857f8a1...`만 `CANONICAL_RUNTIME`이다. H10-only와
> Watch+H10의 당시 “training eligible” 결론은 철회한다.

검토일: 2026-08-13  
Cloud commit: `fa58bb92b0f7d9d1497e05ae3bce77f3b891b643`  
Cloud 작업: 읽기 전용

## 결과

fa58bb9의 `TrainingHandoff.to_dict()`로 생성한 실제 Cloud 형태 fixture를 모델의
`validate_model_training_handoff()`에 전달했다. 사람별 capture-session UUID를 고유하게
준 fixture에서는 세 source variant가 모두 통과했다.

| Variant | Schema UUID/hash | Validator | Serving |
|---|---|---|---|
| Watch-only | 일치 | `VALID_SELECTION_HANDOFF` | canonical |
| H10-only | 일치 | `VALID_SELECTION_HANDOFF` | canonical |
| Watch+H10 | 일치 | `VALID_SELECTION_HANDOFF` | `PROPOSED_NOT_APPROVED` |

공통 fixture는 `TRAIN=1`, `VALIDATION=1`, `LOCKED=0`, `purge_seconds=1800`,
`truth_state=REVIEWED_REAL`이었다. NEGATIVE는 `observation_code=NO_EVENT`로 전달됐고
`temporal_stage=5`는 audit 값으로만 보존됐다.

첫 교차 시도에서는 Cloud test helper가 서로 다른 사람에게 같은 capture-session UUID를
재사용해 `training capture set spans subjects`로 거부됐다. 사람별 고유 capture-session
UUID로 다시 생성하자 통과했다. 이는 모델의 cohort-scoped sequence group 안전장치가
정상 작동한 결과다.

## 남은 충돌·미결정

### 1. BigQuery flat row와 모델 envelope

authorized view SQL은 envelope 필드를 반복한 flat row를 반환하지만 모델 validator는
cohort envelope와 `rows` 배열 하나를 받는다. 동기화 adapter는 반드시
`training_cohort_uuid + cohort_digest` 하나만 조회하고, 모든 행의 envelope 값이 동일한지
확인한 뒤 한 payload로 조립해야 한다. 다른 cohort/digest 혼합은 거부해야 한다.

### 2. digest 독립 재계산 불가

`cohort_digest`와 `split_digest`는 Cloud 내부 person/capture lineage를 포함해 만들어진다.
최소권한 모델 view는 직접 UUID를 제거하므로 모델이 공개 행만으로 두 digest를 독립
재계산할 수 없다. Cloud manifest 신뢰만으로 충분한지, 아니면 공개 handoff rows 기준
별도 `handoff_digest`를 추가할지 확정이 필요하다.

### 3. event time 누락

view의 `window_start_ms/window_end_ms`는 exact window 시간이지 검토된 사건 onset/end가
아니다. 현재 필드만으로 event delay, forecast lead time, onset 기준 false alert를 계산할
수 없다. 해당 지표가 필요하면 reviewed event onset/end를 최소권한 필드로 별도 제공해야
한다.

### 4. review disposition 누락

`evaluation_class=NEGATIVE`는 binary 학습과 전체 FPR 분모에는 충분하지만
`VALID_NON_EVENT`와 `HARD_NEGATIVE`를 나눠 감사할 수 없다. hard-negative 성능을 별도
보고하려면 model view에 비식별 `review_disposition`을 추가해야 한다.

### 5. live 증거 없음

fa58bb9은 저장소 계약과 fixture를 닫았지만 live BigQuery table 3개, authorized view,
실제 reviewed cohort는 아직 배포·생성되지 않았다. 따라서 fixture 통과를 실제 학습 성공,
정확도 또는 배포 준비 완료로 해석하지 않는다.

## 현재 상태

- contract fixture compatibility: `VERIFIED_LOCAL`
- live sync: `BLOCKED_NO_FROZEN_TRAINING_VIEW`
- training: `NOT STARTED`
- threshold/adapter: `NOT STARTED`
- locked evaluation: `NOT STARTED`
- real performance: `NOT VERIFIED / NOT EVALUABLE`
