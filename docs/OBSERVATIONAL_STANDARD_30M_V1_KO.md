# observational_standard_30m_v1 별도 학습 계약

> 상태: `legacy_not_for_serving`. 이 문서는 최초 15-feature 실험 기록으로만
> 보존한다. 실제 streaming 및 신규 학습 계약은
> `OBSERVATIONAL_STANDARD_30M_WATCH_V2_KO.md`를 사용한다.

작성 기준: 2026-08-04

## 목적

이 모델은 기존 `v2.0.0-oracle-sanity` 합성 번들을 대체하지 않는다. 실제 Galaxy
Watch 관측값에서 **현재 시점 이후 30분의 생리 부하 평균**을 미리 예측하는 별도
기준선 모델이다. 행동·사건·임상 판단 라벨이 아니므로 행동 정확도나 임상 정확도로
표시하지 않는다.

현재 Neon 스냅샷에는 `corrected_utc`가 없고 독립 관찰 라벨도 없어 실제 학습을
승격할 수 없다. 이 저장소의 구현은 조건이 갖춰졌을 때 같은 명령으로 재현되도록
준비되어 있으며, 지금 생성되는 synthetic 실행은 `oracle/sanity` 검증으로만 남긴다.

## 입력 계약

입력은 Neon `neon-adapter derive` 결과와 같은 1 Hz Parquet이다.

필수 컬럼은 `person_key` 또는 `session_id`, `corrected_utc`, 그리고 다음 Watch 관측
중 하나 이상이다.

- `observed__eda_us`
- `observed__heart_rate_bpm`
- `observed__motion_magnitude`

가능하면 `quality_confidence`와 `missing_seconds`를 함께 제공한다. `received_at`나
교정되지 않은 센서 시간을 모델 시간축으로 사용하지 않는다. `corrected_utc`가 없으면
즉시 실패한다.

## target과 파생변수

각 시점 `t`의 target은 `t+1..t+1800` 구간에서 품질 게이트를 통과한
`watch_load_raw`의 평균이다. `watch_load_raw`는 개인별 첫 900초의 median/MAD로
EDA·HR·motion을 robust z-score로 만든 뒤 양의 편차의 평균으로 계산한다.

품질은 `quality_confidence >= 0.70`인지를 판단하는 게이트이자 입력 피처다. 품질을
생리 부하에 곱하지 않는다. 미래 구간의 유효 초가 80% 미만이면 target을 결측으로
두고 학습에 넣지 않는다. 모델 피처는 현재까지의 z-score, 30/60초 causal rolling
통계, 300초 rolling median, 품질, 결측 초이며 미래 값은 들어가지 않는다.

## 학습·선택

행 random split은 사용하지 않는다.

- 여러 사람: 사람 단위 train/validation/locked-test 분리
- 한 사람: 세션 시간순 35/40/25 분할
- 모든 경계에 1,800초 purge/embargo 적용
- target window가 경계를 넘는 행과 decision warm-up 전 행은 `purged`
- locked test는 선택·튜닝에서 읽지 않으며 기본 실행은 `locked_test_read=false`

후보는 persistence, 300초 robust rolling median, Ridge, ElasticNet,
HistGradientBoosting이다. validation RMSE가 persistence보다 최소 2% 좋아지는 후보 중
가장 단순한 후보를 선택한다. 통과 후보가 없으면 `NO_MODEL_BEATS_PERSISTENCE`로
남기고 새 모델을 승격하지 않는다.

## 실행

```bash
multisensor-ml observational-standard train \
  --input data/observed/watch_1hz.parquet \
  --output artifacts/observational-standard-30m-v1 \
  --source-domain real_observed
```

산출물은 `manifest.json`, `feature_schema.json`, `baseline.parquet`,
`split_assignments.parquet`, validation metrics/predictions, 그리고 승격된 학습
후보의 `model.skops`·`model.onnx`다. `.pkl`·`.joblib`는 만들지 않는다. `.skops`의
unknown type은 허용 목록을 먼저 검사하며 임의 pickle 로드는 금지한다.

## 현재 상태와 다음 조건

현재 Neon 읽기 전용 스냅샷은 관측 전송 매핑만 확인되었고 `corrected_utc` 0건이다.
따라서 실제 `observational_standard_30m_v1`의 성능은 `NOT VERIFIED`다. 다음 순서로
진행한다.

1. 외부에서 측정한 clock offset/drift를 적용해 corrected UTC를 생성한다.
2. 1 Hz 표와 품질·결측 컬럼을 Neon/Parquet으로 만든다.
3. 위 명령으로 validation forecast를 측정한다.
4. persistence 대비 개선, 결측/OOD, serving parity를 검토한다.
5. 실제 행동 라벨이 축적되기 전에는 행동 모델이나 5단계 임상 결과로 해석하지 않는다.
