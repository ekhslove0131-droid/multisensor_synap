# Kidsignal 모델 플랫폼 계약 구현 계획

> 범위: 모델 저장소만 수정한다. Kidsignal, GCP, BigQuery, Firebase, Android, 배포,
> Git commit/push는 수행하지 않는다.

1. 저장소 및 기존 artifact lineage를 확인한다.
2. UUID/hash, source, feature, stage, review, split 계약을 실패 테스트로 고정한다.
3. Watch-only, H10-only, Watch+H10 schema를 구현한다.
4. 동결 cohort digest와 locked holdout 누출 차단을 구현한다.
5. train-only strongest simple reference를 CPU ONNX로 내보낸다.
6. 독립 truth가 없을 때 빈 FP/FN과 `NOT EVALUABLE`을 내는 평가 manifest를 구현한다.
7. 로컬 contract fixture handoff를 생성하고 hash/ONNX를 재검증한다.
8. Cloud/BigQuery가 구현할 필드·table/view와 모델 산출물 보증을 문서화한다.
9. 관련 테스트, Ruff, mypy, lock, diff를 검증한다.

실제 BigQuery 인증과 실데이터 cohort가 제공된 뒤에만 다음 단계로 이동한다.

- train reference MAE 계산
- validation candidate/adapter/threshold 선택
- champion 하나의 locked holdout 최종 비교
- 독립 review 기반 FP/FN 사례 생성
- 배포 후보 여부 판정
