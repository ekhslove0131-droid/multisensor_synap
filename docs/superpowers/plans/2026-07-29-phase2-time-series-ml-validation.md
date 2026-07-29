# Phase 2 Time-Series ML Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the time-series machine-learning evaluation trustworthy, reproducible, and comparison-ready before adding LightGBM or opening the locked test.

**Architecture:** Keep the existing causal baseline and tabular time-series feature architecture. Establish one segment-aware event evaluation contract shared semantically by the package and Kaggle notebook, persist every validation candidate, then compare Logistic Regression, HistGradientBoosting, and conditionally LightGBM. Apply a finite-state decoder and calibration only after raw probability evaluation passes.

**Tech Stack:** Python 3.12, pandas, NumPy, PyArrow, scikit-learn, optional LightGBM, pytest, Hypothesis, Ruff, mypy, skops, Kaggle notebooks, KNIME 5.12.

## Global Constraints

- Data status remains `oracle/sanity`; real accuracy remains `NOT VERIFIED`.
- Preserve the person split: train 24, validation 6, locked test 6, overlap 0.
- Do not use locked-test rows for feature selection, threshold selection, calibration, or champion selection.
- Never run two Kaggle kernels concurrently.
- Run data preparation and classical ML on CPU. Run the DL benchmark only
  after its upstream kernels succeed, with `NvidiaTeslaT4`, exactly two CUDA
  devices, and `torchrun --nproc_per_node=2`.
- Do not promote a release automatically.
- Preserve raw probability, threshold, decoded state, dataset hash, split hash, and feature schema hash.
- Keep truth, audit-only columns, and model features strictly separated.
- LightGBM is optional and may be installed only after Tasks 1–4 pass.
- Existing `.pkl` and `.joblib` prohibition remains.

---

### Task 1: Publish the Phase 1 Interim Report

**Files:**
- Create: `reports/goal15_phase1_interim_ko.artifact.json`
- Create: `reports/goal15_phase1_interim_ko.html`
- Create: `tests/test_phase1_report.py`
- Read: `docs/superpowers/specs/2026-07-29-time-series-ml-interim-report-design.md`
- Read: `docs/RESULTS_HIERARCHICAL_MVP3.md`
- Read: `docs/RESULTS_MVP3.md`

**Interfaces:**
- Consumes: committed Phase 1 design and recorded Oracle/Kaggle results.
- Produces: one-page Korean HTML report and a source artifact containing the same narrative and metrics.

- [x] **Step 1: Write the failing artifact-content test**

```python
from pathlib import Path


def test_phase1_report_contains_history_failures_and_phase2_plan() -> None:
    html = Path("reports/goal15_phase1_interim_ko.html").read_text()
    required = {
        "1차 합성데이터",
        "시행착오",
        "Kaggle ML data v2",
        "Kaggle ML benchmark v4",
        "event recall 1.0",
        "row F1 0.05",
        "2차 실행계획",
        "oracle/sanity",
        "NOT VERIFIED",
    }
    assert required.issubset(set(filter(lambda item: item in html, required)))
```

- [x] **Step 2: Run the test to verify it fails**

Run: `./.venv/bin/pytest tests/test_phase1_report.py -v`

Expected: FAIL because `reports/goal15_phase1_interim_ko.html` does not exist.

- [x] **Step 3: Build the canonical report artifact**

Create a technical report artifact with these visible sections:

```text
1차에서 해결하려던 문제
합성데이터와 라벨 생성
실제 작업의 시간순 흐름
시행착오: 실패 → 수정 → 배운 점
현재 알게 된 것과 아직 모르는 것
권장 시계열 ML 구조
2차 테스트와 합격 기준
```

Include a bar chart with the three clean validation summaries:

```json
[
  {"target":"패턴","aucpr":0.494423,"auroc":0.810170,"f1":0.049752},
  {"target":"5단계 평균","aucpr":0.639080,"auroc":0.866254,"f1":0.603361},
  {"target":"행동 평균","aucpr":0.398993,"auroc":0.540588,"f1":0.379214}
]
```

- [x] **Step 4: Render and verify the self-contained HTML**

Run the Data Analytics report validator, then package:

```bash
npm run report:deliver -- \
  --input /absolute/path/reports/goal15_phase1_interim_ko.artifact.json \
  --output /absolute/path/reports/goal15_phase1_interim_ko.html
```

Expected: validation and delivery succeed; HTML contains no external data dependency.

- [x] **Step 5: Run report tests and commit**

Run:

```bash
./.venv/bin/pytest tests/test_phase1_report.py -v
git add reports/goal15_phase1_interim_ko.artifact.json reports/goal15_phase1_interim_ko.html tests/test_phase1_report.py
git commit -m "docs: add Phase 1 ML interim report"
```

Expected: PASS and one focused report commit.

---

### Task 2: Lock Segment-Aware Event Metric Invariants

**Files:**
- Modify: `src/multisensor_ml/metrics.py`
- Modify: `tests/test_models_metrics.py`
- Modify: `tests/test_kaggle_notebooks.py`
- Modify: `kaggle/02_ml_benchmark.ipynb`

**Interfaces:**
- Consumes: ordered rows with `dataset_id`, `person_key`, `day_key`, `session_id`, `canonical_time`, truth, and probability.
- Produces: `evaluate_segmented_probabilities(frame, *, threshold, truth_column, probability_column) -> dict[str, float | int | list[list[int]]]`.

- [x] **Step 1: Write failing metric invariants**

Add tests that prove:

```python
def test_alert_run_overlapping_truth_is_not_split_into_false_alerts() -> None:
    truth = np.array([0, 0, 1, 1, 0, 0], dtype=np.int8)
    probability = np.ones(6, dtype=np.float64)
    metrics = evaluate_probabilities(
        truth, probability, threshold=0.5, duration_hours=6 / 3600
    )
    assert metrics["event_recall"] == 1.0
    assert metrics["false_alerts"] == 0
    assert metrics["row_f1"] < 1.0


def test_segment_boundaries_never_merge_false_alerts() -> None:
    frame = pd.DataFrame(
        {
            "dataset_id": ["D1", "D1"],
            "person_key": ["P1", "P1"],
            "day_key": ["2026-01-01", "2026-01-01"],
            "session_id": ["S1", "S2"],
            "canonical_time": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z"],
                utc=True,
            ),
            "label": [0, 0],
            "probability": [0.9, 0.9],
        }
    )
    result = evaluate_segmented_probabilities(
        frame,
        threshold=0.5,
        truth_column="label",
        probability_column="probability",
    )
    assert result["false_alerts"] == 2
```

This follows the maximal-alert-run contract already used in the Kaggle
notebook: one continuous alert that overlaps a truth event is one valid alert,
not two false fragments. Also assert that a time gap, future row, another
person, or another session cannot merge truth or alert runs.

- [x] **Step 2: Run focused tests to verify RED**

Run:

```bash
./.venv/bin/pytest tests/test_models_metrics.py \
  -k "all_positive or segment_boundaries" -v
```

Expected: at least the new segmented API test fails because the function is absent.

- [x] **Step 3: Implement the segmented evaluator**

In `metrics.py`, group with stable sorting:

```python
EVENT_SEGMENT_KEYS = (
    "dataset_id",
    "person_key",
    "day_key",
    "session_id",
)


def evaluate_segmented_probabilities(
    frame: pd.DataFrame,
    *,
    threshold: float,
    truth_column: str,
    probability_column: str,
) -> dict[str, float | int | list[list[int]]]:
    ordered = frame.sort_values(
        [*EVENT_SEGMENT_KEYS, "canonical_time"], kind="mergesort"
    )
    detected = missed = false_alerts = 0
    duration_hours = 0.0
    for _, segment in frame.sort_values(
        [*EVENT_SEGMENT_KEYS, "canonical_time"], kind="mergesort"
    ).groupby(list(EVENT_SEGMENT_KEYS), sort=False, dropna=False):
        truth = segment[truth_column].to_numpy(dtype=np.int8)
        probability = segment[probability_column].to_numpy(dtype=np.float64)
        segment_duration = max(
            (
                pd.to_datetime(segment["canonical_time"], utc=True).iloc[-1]
                - pd.to_datetime(segment["canonical_time"], utc=True).iloc[0]
            ).total_seconds()
            + 1.0,
            1.0,
        ) / 3600
        metrics = evaluate_probabilities(
            truth,
            probability,
            threshold=threshold,
            duration_hours=segment_duration,
        )
        detected += int(metrics["detected_events"])
        missed += int(metrics["missed_events"])
        false_alerts += int(metrics["false_alerts"])
        duration_hours += segment_duration
    result = evaluate_probabilities(
        ordered[truth_column].to_numpy(dtype=np.int8),
        ordered[probability_column].to_numpy(dtype=np.float64),
        threshold=threshold,
        duration_hours=duration_hours,
    )
    result.update(
        detected_events=detected,
        missed_events=missed,
        false_alerts=false_alerts,
        event_recall=detected / (detected + missed) if detected + missed else 0.0,
        event_precision=detected / (detected + false_alerts)
        if detected + false_alerts
        else 0.0,
        false_alerts_per_hour=false_alerts / duration_hours
        if duration_hours
        else 0.0,
    )
    precision = float(result["event_precision"])
    recall = float(result["event_recall"])
    result["event_f1"] = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return result
```

Reject duplicate time inside one segment, missing identity, non-binary truth, non-finite probability, and probability outside `[0, 1]`.

- [x] **Step 4: Prove Kaggle/package semantic parity**

Use deterministic fixtures to compare the Kaggle notebook namespace with
`evaluate_segmented_probabilities`. The same truth, probability, threshold, and
segment keys must produce identical detected, missed, false-alert, recall, F1,
and duration values.

Run:

```bash
./.venv/bin/pytest tests/test_models_metrics.py tests/test_kaggle_notebooks.py \
  -k "event_metric or segment" -v
```

Expected: PASS with identical direct and common-grid results.

- [x] **Step 5: Run the full suite and commit**

Run:

```bash
./.venv/bin/ruff check .
./.venv/bin/mypy src
./.venv/bin/pytest
git add src/multisensor_ml/metrics.py tests/test_models_metrics.py tests/test_kaggle_notebooks.py kaggle/02_ml_benchmark.ipynb
git commit -m "fix: unify segment-aware event evaluation"
```

Expected: all commands exit 0.

---

### Task 3: Persist Every Validation Candidate

**Files:**
- Modify: `kaggle/02_ml_benchmark.ipynb`
- Modify: `tests/test_kaggle_notebooks.py`
- Modify: `src/multisensor_ml/contracts.py`

**Interfaces:**
- Consumes: Logistic and HGB validation prediction frames.
- Produces: `validation_candidate_metrics.parquet`, `validation_candidate_manifest.json`, and the existing champion artifact.

- [x] **Step 1: Write failing artifact-contract tests**

Require exactly two candidate names, one threshold per candidate/target, unique
`model_name × target × metric × stress_condition`, runtime seconds, row support,
event count, source dataset hash, split hash, and feature schema hash.

- [x] **Step 2: Verify RED**

Run:

```bash
./.venv/bin/pytest tests/test_kaggle_notebooks.py \
  -k "candidate_metric_artifact" -v
```

Expected: FAIL because v5 writes champion-only metrics.

- [x] **Step 3: Write candidate metrics before champion filtering**

The manifest must use:

```json
{
  "schema_version": "goal1.5/ml-validation-candidates/v1",
  "split_role": "validation",
  "locked_test_used": false,
  "candidate_models": ["hist_gradient_boosting", "logistic_regression"]
}
```

Reject overwrite when the manifest hash differs.

- [x] **Step 4: Verify reload and deterministic champion selection**

Reload the Parquet, select the champion from it, and assert that row order changes
do not change the selected model or threshold.

- [x] **Step 5: Run tests and commit**

Run full Ruff, mypy, pytest and commit:

```bash
git commit -m "feat: persist Kaggle validation candidates"
```

---

### Task 4: Prove Causal Time-Series Features

**Files:**
- Modify: `src/multisensor_ml/features.py`
- Modify: `tests/test_baseline_features.py`
- Modify: `kaggle/01_ml_data.ipynb`
- Modify: `kaggle/02_ml_benchmark.ipynb`
- Modify: `kaggle/03_dl_sequence_data.ipynb` (schema parity only; do not run)
- Modify: `kaggle/04_dl_tcn_benchmark.ipynb` (schema parity only; do not run)
- Modify: `tests/test_kaggle_notebooks.py`

**Interfaces:**
- Consumes: one-Hz person/session timelines and causal baseline values.
- Produces: lag `1/5/15/30/60s`, rolling `5/15/30/60/180/300s`, delta, slope, and baseline-deviation features.

- [x] **Step 1: Add future-mutation and boundary tests**

For every feature family:

```python
past_before = build_features(frame).loc[:cutoff]
mutated = frame.copy()
mutated.loc[mutated.index > cutoff, factor_columns] += 999
past_after = build_features(mutated).loc[:cutoff]
pd.testing.assert_frame_equal(past_before, past_after)
```

Also assert lags reset at person, dataset, day, session, and time-gap boundaries.

- [x] **Step 2: Verify RED**

Run the new focused tests and confirm at least one gap-reset test fails.

- [x] **Step 3: Implement explicit segment grouping and gap reset**

Use only rows at or before the current timestamp. Never backward-fill a causal
feature. Add a boolean `history_sufficient` audit column that is not a feature.

- [x] **Step 4: Prove role parity**

Train, validation, and locked-test schemas must match exactly while labels,
audit-only columns, and future values remain excluded.

- [x] **Step 5: Run full verification and commit**

Commit:

```bash
git commit -m "feat: harden causal time-series features"
```

---

### Task 5: Re-run the Two Existing ML Candidates

**Files:**
- Modify: `kaggle/02_ml_benchmark.ipynb`
- Create: `docs/results/phase2_ml_baseline_validation.md`

**Interfaces:**
- Consumes: corrected ML data view and candidate artifact contract.
- Produces: Logistic/HGB validation comparison without locked test.

- [ ] **Step 1: Push the data notebook only if its hash changed**

Keep `enable_gpu=false`, `enable_internet=false`, and `RUN_LOCKED_TEST=False`.

- [ ] **Step 2: Run exactly one ML benchmark kernel**

Do not run DL notebooks. Record Kaggle version, runtime, input hashes, output
hashes, and candidate artifacts.

- [ ] **Step 3: Apply the validation acceptance gate**

Both candidates must report:

```text
AUCPR, AUROC, row F1, person-macro F1,
event recall, event precision, event F1,
false alerts/hour, Brier, ECE, runtime, artifact size
```

Reject the run if any metric is missing, segment parity fails, or event counts do
not reconstruct from predictions.

- [ ] **Step 4: Record the evidence and commit**

Do not call either model a final champion. Commit the validation comparison only.

---

### Task 5B: Run the DL Comparison Sequentially on T4 x2

**Files:**
- Modify staging metadata only for `kaggle/03_dl_sequence_data.ipynb`
- Modify staging metadata only for `kaggle/04_dl_tcn_benchmark.ipynb`
- Create: `docs/results/phase2_dl_t4x2_validation.md`

**Interfaces:**
- Consumes: successful Task 5 ML outputs and the full causal sequence view.
- Produces: one DL validation comparison; no locked-test execution.

- [ ] **Step 1: Wait for every earlier Kaggle kernel to finish**

Confirm no data or ML kernel is running. Do not start another kernel while one
is pending.

- [ ] **Step 2: Run sequence preparation on CPU**

Push `03_dl_sequence_data` with GPU disabled and wait for `complete`.

- [ ] **Step 3: Run one T4 x2 DL kernel**

Push `04_dl_tcn_benchmark` with:

```text
--accelerator NvidiaTeslaT4
RUN_TRAINING=True
RUN_LOCKED_TEST=False
torch.cuda.device_count()==2
torchrun --nproc_per_node=2
```

The notebook must fail before training if Kaggle supplies anything other than
two CUDA devices.

- [ ] **Step 4: Record ML/DL validation evidence**

Record device names/count, runtime, hashes, AUCPR/AUROC, row and event metrics,
calibration, and stress results. Keep the result `oracle/sanity` and do not
promote a release.

---

### Task 6: Add LightGBM Behind a Dependency and Evidence Gate

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `src/multisensor_ml/models.py`
- Modify: `tests/test_models_metrics.py`
- Modify: `kaggle/02_ml_benchmark.ipynb`

**Interfaces:**
- Consumes: the exact Task 5 train/validation feature schema and metric contract.
- Produces: a third `lightgbm` validation candidate.

- [ ] **Step 1: Confirm the entry gate**

Proceed only if Tasks 2–5 pass and the user-approved environment permits the
new dependency. Otherwise record `LIGHTGBM_NOT_RUN_GATE_FAILED`.

- [ ] **Step 2: Add a failing three-candidate contract test**

Assert the candidate set is exactly:

```python
{"logistic_regression", "hist_gradient_boosting", "lightgbm"}
```

- [ ] **Step 3: Add the minimal CPU LightGBM candidate**

Use deterministic seeds, class balancing, bounded leaves and estimators, and no
GPU. Do not tune against locked test.

- [ ] **Step 4: Compare with a predeclared promotion rule**

LightGBM may replace HGB only if validation AUCPR and event F1 improve without
worsening false alerts/hour or ECE beyond 5%, and person-macro F1 does not fall.

- [ ] **Step 5: Verify, document, and commit**

Store all three candidates whether or not LightGBM wins.

---

### Task 7: Add Decoder, Calibration, Stress, and Abstention

**Files:**
- Modify: `src/multisensor_ml/hierarchical.py`
- Modify: `src/multisensor_ml/stress.py`
- Modify: `tests/test_hierarchical.py`
- Modify: `tests/test_bundle_stress.py`

**Interfaces:**
- Consumes: raw event/stage probabilities and quality/OOD state.
- Produces: raw probability, calibrated probability, decoded stage, and `NOT_DECISIONABLE`.

- [ ] **Step 1: Add state-transition property tests**

Reject forbidden transitions, enforce minimum duration and hysteresis, allow any
initial state, and abstain on OOD, long gaps, or failed quality.

- [ ] **Step 2: Add person-grouped calibration tests**

Compare sigmoid and isotonic on validation only. Adopt calibration only when both
Brier and ECE improve without reducing event F1 beyond 0.01.

- [ ] **Step 3: Re-run deterministic stress**

Run Gaussian, block missing, axis shifts, and latent dropout. Require candidate
and clean hashes in every stress row.

- [ ] **Step 4: Verify and commit**

Raw and decoded outputs must both survive bundle reload.

---

### Task 8: Select One Composite Release and Open Locked Test Once

**Files:**
- Modify: `src/multisensor_ml/model_registry.py`
- Modify: `tests/test_model_registry.py`
- Create: `docs/results/phase2_locked_test.md`

**Interfaces:**
- Consumes: one validation-selected baseline/type/stage/behavior/calibration/decoder composite.
- Produces: one immutable locked-test evaluation and no automatic promotion.

- [ ] **Step 1: Verify the release gate**

Require complete candidate metrics, metric parity, validation audit reason,
stress results, translation coverage, and zero previous locked-test record for
the model hash × dataset hash.

- [ ] **Step 2: Run the existing duplicate-rejection tests**

Confirm a repeated locked-test request fails before executing the first run.

- [ ] **Step 3: Execute one locked test**

Do not change models, thresholds, feature schema, calibration, or decoder after
seeing the result.

- [ ] **Step 4: Record outcome without overclaiming**

Keep `oracle/sanity`, actual sensor performance `NOT VERIFIED`, and release
`candidate` until manual promotion.

- [ ] **Step 5: Run final verification and commit**

Run:

```bash
./.venv/bin/ruff check .
./.venv/bin/mypy src
./.venv/bin/pytest
uv lock --check
```

Commit the immutable result and audit record.
