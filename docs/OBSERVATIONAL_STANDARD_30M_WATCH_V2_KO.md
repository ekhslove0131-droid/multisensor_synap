# Watch observational standard 30분 예측 v2

작성 기준: 2026-08-04

## 경계

이 계약은 실제 Galaxy Watch 관측값의 미래 1,800초 생리 부하 평균을 예측하기 위한
비임상 observational shadow 계약이다. 합성 `monitoring_v2`, 행동 라벨, 사건 단계와
분리한다. 실제 성능은 독립 검증 전까지 `NOT VERIFIED`이며 `stage=null`이다.

기존 `observational_standard_30m_v1` 15-feature artifact는
`legacy_not_for_serving`이다. 품질 기본값 1.0, 일부 core 신호 허용, 80% 미래 유효률,
미래 target과 serving decisionability 결합 때문에 subset 호환이나 serving 재사용을
금지한다.

## 고정 버전

```text
baseline shadow release: observational_standard_30m_watch_baseline_shadow_v1
candidate trainer: observational_standard_30m_watch_candidate_v2
feature schema: watch_standard_16_v2
feature schema SHA-256: 2857f8a16cd4450a18f8701c1c9c9f397dee4b291fefd2475279883a0f8a29de
baseline: personal_robust_baseline_900s_v2
load: watch_positive_z_bounded_ema_v2
quality: watch_core_quality_min_v1
target: observational_future_mean_1800_valid90_gap5_v2
```

## Canonical 입력 순서

ONNX 경계는 `float32[N,16]`이다. 내부 계산은 float64이며 다음 순서를 바꾸지 않는다.

1. `watch_eda_z`
2. `watch_hr_z`
3. `watch_motion_z`
4. `watch_load_raw`
5. `watch_eda_mean_30`
6. `watch_hr_mean_30`
7. `watch_motion_mean_30`
8. `watch_eda_slope_60`
9. `watch_hr_slope_60`
10. `watch_motion_slope_60`
11. `watch_load_std_60`
12. `watch_load_median_300`
13. `watch_load_ema_1800`
14. `watch_load_ema_21600`
15. `quality_confidence`
16. `watch_ineligible_fraction_60`

정확한 수식·history·단위는 bundle의 `feature_schema.json`이 canonical이다. 시간
sin/cos는 corrected UTC만으로 생활 맥락을 추측해 일정 누출을 만들 수 있어 제외했고,
중복성이 큰 `watch_load_mean_30`도 제외했다.

## 판단 계약

- HR·EDA·ACC와 `quality_confidence`가 모두 필수다. 누락은 즉시 실패한다.
- baseline은 최대 1,000 wall seconds 안의 900 eligible seconds이며 gap은 최대 5초다.
- scale은 `1.4826*MAD`, 실패 시 `IQR/1.349`, 다시 실패하면
  `BASELINE_SCALE_ZERO`다.
- 900초 baseline은 총 1,800 eligible decision warm-up에 포함된다.
- rolling window는 `[t-window,t)` strict past이며 slope는 corrected-second OLS다.
- EMA는 eligible seconds에서만 갱신하고 5초 초과 gap이면 dynamic state를 초기화한다.
- `feature_decisionable`은 과거·현재만, `target_valid`는 미래 유효성만 나타낸다.
- target은 `t+1..t+1800` load 평균이며 90% 이상 유효하고 gap이 5초 이하여야 한다.
- 품질은 gate/feature이며 `watch_load_raw`에 곱하지 않는다.

## baseline-only shadow

실제 데이터가 부족한 현재 배포 가능 artifact는 학습 champion이 아니라
`rolling_median_300` selector다. 신규 16-vector의 12번째 열(0-based index 11)을
그대로 반환한다. 이 출력은 파이프라인 재현·latency·누락 상태를 검증하기 위한 것이며
정확도나 임상 성능을 뜻하지 않는다.

```bash
multisensor-ml observational-standard export-baseline-shadow \
  --output artifacts/observational-standard-30m-watch-baseline-shadow-v1

multisensor-ml observational-standard verify-baseline-shadow \
  --bundle artifacts/observational-standard-30m-watch-baseline-shadow-v1
```

학습 후보를 만들 때는 시간순/person split, 1,800초 purge/embargo, validation 선택만
사용한다. locked test는 champion 한 번 전까지 읽지 않는다.
