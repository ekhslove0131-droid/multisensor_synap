# Multisensor ML — Goal 1.5

`../multisensor_synth`가 만든 Goal 1 truth에서 최초 학습 환경을 검증하는
**oracle/sanity 전용** 프로젝트입니다. 실제 장비, 의료 성능 또는 Goal 2–4
`model_ready` 성능을 주장하지 않습니다.

## 가장 먼저 알아둘 것

- 생성기는 `../multisensor_synth`이고 이 저장소는 생성기를 editable path
  dependency로 읽기만 합니다.
- `materialize-synthetic`를 실행하면 새 synthetic 데이터가 이 저장소의
  `data/raw/`에 실제로 생성됩니다. 생성기 코드만 만드는 명령이 아닙니다.
- 대용량 데이터, prepared feature, 모델 bundle, KNIME runtime export는 Git에
  넣지 않습니다.
- 실제 데이터가 아직 없으므로 real accuracy와 device synchronization은 항상
  `NOT VERIFIED`입니다.

## 빠른 실행

Python 3.12와 `uv` 환경을 고정한 뒤 다음처럼 실행합니다.

```bash
uv sync --dev
uv run multisensor-ml run-all --config configs/goal15_quick.yaml
```

36명×5일 MVP 전체 실행:

```bash
uv run multisensor-ml run-all --config configs/goal15_mvp3.yaml
```

MVP 최초 실측은 generation 약 1분, feature prepare 약 4분,
전체 validation stress 포함 학습 약 239분이었습니다. 반복 실험에서는 quick로
후보를 좁힌 뒤 full stress를 실행하는 편이 합리적입니다.

## 공개 CLI

```text
multisensor-ml materialize-synthetic --config <yaml>
multisensor-ml prepare --series <series_id>
multisensor-ml train --experiment <yaml>
multisensor-ml evaluate --bundle <dir> --dataset <id> --role validation|locked_test
multisensor-ml export-knime --experiment <id>
multisensor-ml run-all --config <yaml>
multisensor-ml factory run --config <yaml> --output-receipt <json>
multisensor-ml factory validate --receipt <json>
multisensor-ml registry init --config <yaml>
multisensor-ml registry import-oracle --experiment <id>
multisensor-ml registry run-stage --stage <stage> --run-id <id> \
  --input-receipt <json> --output-receipt <json>
multisensor-ml registry run-all --config <yaml>
multisensor-ml registry compare --target <target>
multisensor-ml registry promote --release <id> --audit-reason <text>
multisensor-ml registry predict --release <id> --dataset <id>
multisensor-ml registry export-knime --run-id <id>
```

`locked_test` 재평가에는 `--audit-reason`이 필요하며, 동일 model hash와
dataset hash 조합은 두 번 실행할 수 없습니다.

## 학습 구조

1. train 사람만 사용해 context별 global median/MAD를 계산합니다.
2. 개인별 과거 데이터만 사용해 1,800초 warm-up, 21,600초 lookback,
   60초 refresh personal baseline을 만듭니다.
3. `n_eff / (n_eff + 1800) × quality_confidence`로 global/personal 기준선을
   혼합합니다. oracle quality는 1입니다.
4. 7개 latent robust-z에서 5/15/30/60/180/300초 causal
   mean/std/slope를 생성합니다.
5. `event_binary`, `forecast_60s` 각각 Logistic Regression과
   HistGradientBoosting을 학습합니다.
6. 임계값은 validation event-F1로 한 번만 고정하고 locked test에서는
   변경하지 않습니다.

상세 내용은 [아키텍처](docs/ARCHITECTURE.md),
[KNIME 사용법](docs/KNIME.md), [MVP 결과](docs/RESULTS_MVP3.md)를 참고하세요.
합성 공장과 계층형 학습 레지스트리 사용법은
[한국어 KNIME 안내](docs/KNIME_FACTORY_AND_REGISTRY_KO.md)를 참고하세요.
기존 [Multisensor_ML_Goal1_5.knwf](knime/Multisensor_ML_Goal1_5.knwf)는
Oracle Benchmark로 보존하며, 새 워크플로는
[합성데이터 공장](knime/Multisensor_Synthetic_Factory_Goal1_5.knwf)과
[학습 레지스트리](knime/Multisensor_ML_Training_Registry.knwf)입니다.

## 안전한 번들

모델은 `.skops`만 사용합니다. 로드 전에 SHA-256과 unknown type 목록을
검사하며 `.pkl`, `.pickle`, `.joblib`은 금지합니다. 번들에는 baseline,
feature schema, threshold, validation/locked-test/noise metrics, prediction,
PR curve, dependency versions, `uv.lock`, real/synchronization 상태가 함께
들어갑니다.
