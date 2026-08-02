# 센서 가용성 모델 Kaggle 재현 결과

확인일: 2026-08-02

## 최종 상태

- Kaggle Model: [multisensor-goal15-availability](https://www.kaggle.com/models/bjcoding/multisensor-goal15-availability)
- variation: `ScikitLearn/oracle-sanity-v1/3`
- 최종 archive SHA-256: `63526a7d90e80bbb02e9437674361fb939204912dad0f0fbc0173cf80043a313`
- Kaggle 재현 kernel: [goal15-availability-reproduction](https://www.kaggle.com/code/bjcoding/goal15-availability-reproduction)
- kernel version: `4`, 상태 `COMPLETE`
- 결과: `REPRODUCED`, 7개 프로파일 × 600 validation sample
- 학습 실행: `False`, locked test 읽기: `False`, GPU: `False`
- 데이터 범위: `oracle/sanity`, 실제 Neon accuracy: `NOT VERIFIED`

최종 receipt는 Kaggle kernel의 `availability_reproduction_receipt.json`에 저장되며, 모든
프로파일이 `REPRODUCED`다.

## 포함 프로파일

`watch_only`, `polar_only`, `muse_only`, `watch_polar`, `watch_muse`, `polar_muse`,
`watch_polar_muse` 일곱 조합을 동일한 사건 gate·5단계 decoder로 재현했다. 출력은
`event_probability`, `predicted_event`, `predicted_stage`다.

## 시행착오와 해결

1. 최초 v1 archive는 모델 자체와 custom wheel만 포함했다. Kaggle 기본 환경에 `skops`가 없어
   `ModuleNotFoundError: skops`가 발생했다.
2. v2에는 `skops`·`prettytable` 오프라인 wheel을 추가했다. 이때 Kaggle의 scikit-learn 1.6.1이
   로컬 1.9.0으로 저장된 LogisticRegression을 읽기는 했지만 `multi_class` 속성이 없어
   `predict_proba`에서 실패했다.
3. 모델을 다시 학습하지 않고 안전한 `.skops` 로더에 버전 호환 처리를 넣었다. 누락된
   `multi_class`만 역사적 기본값 `auto`로 복원하고, unknown type 검사는 계속 수행한다.
4. Kaggle Model 다운로드가 archive를 `model_payload/`로 자동 확장하는 동작도 확인했다. 노트북은
   archive가 있는 로컬 경로와 확장 payload인 Kaggle 경로를 모두 검증한다.

최종 v3 archive에는 다음 runtime wheel이 들어 있다.

- `multisensor_ml-0.1.0-py3-none-any.whl`
- `skops-0.14.0-py3-none-any.whl`
- `prettytable-3.18.0-py3-none-any.whl`

## 사용 경계

이 결과는 합성 latent feature subset을 이용한 가용성 ablation 재현이다. 실제 Galaxy Watch8
payload는 먼저 `neon_mapping.py`로 디코드하고 `corrected_utc`, 품질, 결측, clock metadata와
관찰자 라벨을 만든 뒤 별도 adapter에서 평가한다. 3축 truth를 한 장비의 신호로 대체하지 않으며,
실제 행동/사건 정확도나 의료 판단으로 해석하지 않는다.
