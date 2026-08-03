# Monitoring v2 합성 검증 결과

실행 시리즈: `standard-pattern-v2-quick`
데이터: 3 seed × 12명 = 36명, train/validation/locked_test = 24/6/6
평가 범위: validation 6명만 읽음, `locked_test_read=false`
상태: `oracle/sanity` · 실제 정확도 `NOT VERIFIED`

## 표준선 30분 예측

| 센서 뷰 | M0 MAE | M0 RMSE | persistence MAE | 판정 |
|---|---:|---:|---:|---|
| Galaxy Watch | 0.12821 | 0.17349 | 0.30909 | M0가 기준선보다 낮은 오차 |
| Galaxy Watch + H10 | 0.11388 | 0.15565 | 0.30909 | 두 센서가 가장 낮은 오차 |

이번 합성 시나리오에서는 H10을 추가한 M0가 표준선 예측 MAE가 더 낮았다.
이 결과는 실제 장비 우월성의 증명이 아니라 현재 합성 관측 노이즈 설정에 대한
검증이다.

## 패턴 head validation

| 패턴 | Watch AUCPR | Watch+H10 AUCPR | Watch F1 | Watch+H10 F1 |
|---|---:|---:|---:|---:|
| meltdown_like | 0.2900 | 0.2676 | 0.3237 | 0.2605 |
| tantrum_like | 0.0491 | 0.0560 | 0.1175 | 0.1231 |
| sensory_seeking_like | 0.1832 | 0.2587 | 0.2969 | 0.3625 |
| stereotypy_like | 0.2907 | 0.1791 | 0.3628 | 0.2799 |
| shutdown_like | 0.0871 | 0.1078 | 0.1514 | 0.1706 |

패턴 결과는 가상 prior의 재현성 확인용이다. 특히 `tantrum_like`는 센서만으로
구분할 수 없도록 설계했기 때문에 낮은 AUCPR이 오히려 계약과 일치한다. 실제
운영에서는 요구·의사소통·관찰자 라벨이 추가되기 전 `NOT_DECISIONABLE`로 둔다.

## 산출물

- 모델 결과 JSON: `artifacts/monitoring-v2/standard-pattern-v2-quick/result.json`
- Watch bundle: `services/onnx_api/models/goal15-monitor-v2/galaxy_watch/`
- Watch+H10 bundle: `services/onnx_api/models/goal15-monitor-v2/galaxy_watch_h10/`
- Dockerfile: `services/onnx_api/Dockerfile.v2`
- API: `services.onnx_api.app_v2:app`

두 bundle 모두 표준선 ONNX 1개와 패턴 ONNX 5개를 포함하고, 각 파일의 SHA-256을
manifest에 기록한다. graph 이름·LightGBM seed를 고정해 동일 설정을 다시 실행해도
ONNX hash가 일치한다. 로컬 Docker 이미지 빌드와 `/healthz`, `/v2/predict` smoke
readback을 통과했다.
