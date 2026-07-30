# Phase 3 Personal-Context Pattern Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quantify whether causal personal-context baselines and cumulative-load state improve pattern detection over a global-context baseline for every split, run, physical dataset, validation person, and context.

**Architecture:** Build reusable causal baseline and cumulative-load modules, then run a pattern-only controlled ablation with identical labels, sampling, estimators, and group metrics. Use train-person grouped OOF predictions to select baseline weight caps and thresholds, keep validation untouched until selection, and never open locked test. Kaggle uses a private source-preparation notebook followed by one CPU benchmark notebook; behavior remains audit-only.

**Tech Stack:** Python 3.12, NumPy, pandas, PyArrow, scikit-learn, Pydantic, PyYAML, pytest, Hypothesis, Ruff, mypy, Kaggle CLI, self-contained HTML.

## Global Constraints

- Data status is `oracle/sanity`; real accuracy is `NOT VERIFIED`.
- Preserve person split train 24, validation 6, locked test 6 with overlap 0.
- Never use locked-test rows in this phase.
- Never use event truth, phase, behavior labels, hidden archetype, event intensity, or participant truth baseline as model features.
- Only baseline policy changes between ablation versions.
- `pattern_binary` is the only champion-selection target.
- Behavior results are audit-only.
- Cumulative load is causal and label-free.
- Cumulative-load baseline influence is capped at 10% of the global MAD.
- Run Kaggle kernels sequentially on CPU.
- Do not automatically promote or retrain a release.

---

### Task 1: Lock Phase 3 Contracts

**Files:**
- Create: `src/multisensor_ml/phase3_contracts.py`
- Create: `configs/phase3_personal_pattern.yaml`
- Create: `tests/test_phase3_contracts.py`
- Modify: `src/multisensor_ml/settings.py`

**Interfaces:**
- Produces: `BaselineVariant`, `CumulativeLoadConfig`, `Phase3Config`, `load_phase3_config(path: Path) -> Phase3Config`.
- Candidate ids: `G0`, `P1-025`, `P1-050`, `P1-100`, `P2-025`, `P2-050`, `P2-100`, `P2+CL`, `P2+CL-B`.

- [x] **Step 1: Write failing schema tests**

Test exact candidate ids, caps `{0.0, 0.25, 0.5, 1.0}`, horizons
`{1800, 21600, 86400, 259200}`, warm-up `1800`, lookback `21600`,
refresh `60`, baseline load influence `0.10`, and rejection of any
locked-test option.

- [x] **Step 2: Run RED**

Run:

```bash
./.venv/bin/pytest tests/test_phase3_contracts.py -v
```

Expected: import failure because `phase3_contracts` does not exist.

- [x] **Step 3: Implement strict Pydantic contracts and YAML loader**

The loader must reject unknown fields and require:

```yaml
schema_version: goal1.5/phase3-personal-pattern/v1
series_id: mvp3-oracle-v1
split_counts: {train: 24, validation: 6, locked_test: 6}
warmup_sec: 1800
lookback_sec: 21600
refresh_sec: 60
weight_caps: [0.0, 0.25, 0.5, 1.0]
cumulative_horizons_sec: [1800, 21600, 86400, 259200]
baseline_load_influence_cap: 0.10
run_locked_test: false
```

- [x] **Step 4: Run GREEN and commit**

Run the focused test, Ruff and mypy for the new module, then commit:

```bash
git commit -m "feat: define Phase 3 validation contracts"
```

---

### Task 2: Add Causal Personal-Context Baselines

**Files:**
- Create: `src/multisensor_ml/phase3_baselines.py`
- Create: `tests/test_phase3_baselines.py`

**Interfaces:**
- Produces:
  - `fit_train_global_context_baseline(frame, factors) -> pd.DataFrame`
  - `build_causal_baseline(frame, global_baseline, *, mode, weight_cap, warmup_sec, lookback_sec, refresh_sec) -> pd.DataFrame`
  - `baseline_policy_hash(...) -> str`
- `mode` is one of `global_context`, `personal_pooled`, `personal_context`.

- [x] **Step 1: Write failing causal tests**

Prove:

- changing a future row does not alter any prior center, MAD, `n_eff`, or weight;
- `personal_context` never consumes rows from another context;
- the first 1,800 valid same-context observations use weight 0;
- every weight is within `[0, weight_cap]`;
- G0 centers and MADs equal the train-only global context table;
- current row is excluded from personal median/MAD;
- unknown context fails closed.

- [x] **Step 2: Run RED**

Run:

```bash
./.venv/bin/pytest tests/test_phase3_baselines.py -v
```

Expected: missing module/API failure.

- [x] **Step 3: Implement the minimal baseline engine**

Use stable ordering by `person_key`, `run_id`, `timestamp_utc`. Personal history
is trailing-only. Refresh values every 60 seconds and forward-fill only within
the same person and policy context. Apply `MAD_FLOOR=1e-4`.

- [x] **Step 4: Run GREEN, property tests, and commit**

Commit:

```bash
git commit -m "feat: add causal personal-context baselines"
```

---

### Task 3: Add Label-Free Cumulative Load

**Files:**
- Create: `src/multisensor_ml/cumulative_load.py`
- Create: `tests/test_cumulative_load.py`

**Interfaces:**
- Produces:
  - `instantaneous_load(robust_z: pd.DataFrame) -> pd.Series`
  - `build_cumulative_load(frame, robust_z, *, horizons_sec) -> pd.DataFrame`
  - `classify_load_state(load_frame, train_quantiles) -> pd.Series`
- Output columns:
  - `cumulative_load_30m`
  - `cumulative_load_6h`
  - `cumulative_load_24h`
  - `cumulative_load_72h`
  - per-factor horizon columns
  - `load_state`
  - `load_slope_30m`

- [ ] **Step 1: Write failing load-state tests**

Prove:

- future mutation cannot change past load;
- constant exposure increases load monotonically toward a bounded state;
- sustained recovery decreases load;
- person/run changes reset state;
- a known time gap applies exponential decay for elapsed time;
- event labels and phase columns do not affect results;
- motor and social factors are excluded from the scalar composite but retained as independent horizon features;
- 72-hour maturity is false before 259,200 valid seconds.

- [ ] **Step 2: Run RED**

Run:

```bash
./.venv/bin/pytest tests/test_cumulative_load.py -v
```

- [ ] **Step 3: Implement causal decay**

For horizon `h`, use:

```text
decay(dt) = exp(-ln(2) * dt / h)
state_t = max(0, decay(dt) * state_(t-1) + (1 - decay(dt)) * instantaneous_load_t)
```

The scalar load is the equal-weight mean of positive autonomic, cognitive,
sensory and sleep-pressure deviations plus negative recovery deviation.
Load-state quantiles are fit on train people only.

- [ ] **Step 4: Run GREEN and commit**

Commit:

```bash
git commit -m "feat: add causal cumulative-load state"
```

---

### Task 4: Build Pattern-Only Ablation and Group Uplift

**Files:**
- Create: `src/multisensor_ml/phase3_evaluation.py`
- Create: `tests/test_phase3_evaluation.py`
- Modify: `src/multisensor_ml/metrics.py`

**Interfaces:**
- Produces:
  - `evaluate_pattern_predictions(frame, *, threshold) -> dict[str, float]`
  - `evaluate_group_uplift(global_predictions, personal_predictions, *, group_columns) -> pd.DataFrame`
  - `select_smallest_stable_cap(oof_metrics, *, relative_guardrail=0.05) -> str`
  - `phase3_adoption_decision(metrics, uplift) -> str`

- [ ] **Step 1: Write failing metric tests**

Test exact absolute and relative deltas for:

- split role;
- run id;
- 36 physical dataset ids;
- validation person;
- context;
- cumulative-load state.

Test `NOT_COMPUTABLE` for a zero denominator, `INSUFFICIENT_SUPPORT` for a
single-class group, and improvement/tie/degradation counts. Prove that accuracy
is exported but never used in selection.

- [ ] **Step 2: Run RED**

Run:

```bash
./.venv/bin/pytest tests/test_phase3_evaluation.py -v
```

- [ ] **Step 3: Implement group-safe evaluation**

Use the existing segment-aware event contract. Group metrics must never merge
events across dataset, person, day, session or a time gap. `G0` is always the
uplift denominator.

- [ ] **Step 4: Run GREEN and commit**

Commit:

```bash
git commit -m "feat: evaluate Phase 3 dataset uplift"
```

---

### Task 5: Implement the Streaming Phase 3 Pipeline

**Files:**
- Create: `src/multisensor_ml/phase3_pipeline.py`
- Create: `tests/test_phase3_pipeline.py`
- Modify: `src/multisensor_ml/cli.py`

**Interfaces:**
- CLI:

```text
multisensor-ml phase3 prepare --config <yaml>
multisensor-ml phase3 train-validate --config <yaml>
multisensor-ml phase3 report-input --config <yaml>
```

- Produces:
  - `phase3_candidate_metrics.parquet`
  - `phase3_group_uplift.parquet`
  - `phase3_person_predictions.parquet`
  - `phase3_baseline_diagnostics.parquet`
  - `phase3_cumulative_load.parquet`
  - `phase3_manifest.json`

- [ ] **Step 1: Write a failing quick integration test**

Use the existing quick raw series. Assert deterministic source/split/policy/
feature hashes, no locked-test reads, identical sampling ids across variants,
train-person grouped OOF cap/threshold selection, and complete output schemas.

- [ ] **Step 2: Run RED**

Run:

```bash
./.venv/bin/pytest tests/test_phase3_pipeline.py -v
```

- [ ] **Step 3: Implement streaming preparation**

Read one person at a time from the three raw latent timelines. Fit global
baselines from train people only. Generate one candidate variant at a time so
multiple full 15.5M-row feature matrices are never resident together.

- [ ] **Step 4: Implement train OOF selection and validation**

Use identical deterministic positive/hard-negative/matched-baseline row ids for
every policy. Select weight cap and threshold only from person-grouped OOF.
Evaluate `G0`, selected `P1`, selected `P2`, `P2+CL`, and `P2+CL-B` on the
untouched validation people. HGB is primary; Logistic is retained for G0 and
the final selected personalized candidate as a regression reference.

- [ ] **Step 5: Verify quick integration and commit**

Commit:

```bash
git commit -m "feat: run Phase 3 personal-pattern ablation"
```

---

### Task 6: Add Two Sequential Kaggle Notebooks

**Files:**
- Create: `kaggle/05_phase3_source_prepare.ipynb`
- Create: `kaggle/06_phase3_personal_pattern.ipynb`
- Create: `kaggle/05_phase3_source_prepare.kernel-metadata.json`
- Create: `kaggle/06_phase3_personal_pattern.kernel-metadata.json`
- Modify: `tests/test_kaggle_notebooks.py`

**Interfaces:**
- 05 consumes the private raw Oracle source and emits a sanitized, hash-bound Phase 3 source.
- 06 consumes only 05 output and emits the Phase 3 metrics, uplift, predictions, diagnostics and manifest.

- [ ] **Step 1: Write failing notebook contract tests**

Require private kernels, CPU, internet off, `RUN_LOCKED_TEST=False`, exact
05→06 kernel-source handoff, Oracle denylist enforcement, and output schemas.

- [ ] **Step 2: Run RED**

Run:

```bash
./.venv/bin/pytest tests/test_kaggle_notebooks.py -k phase3 -v
```

- [ ] **Step 3: Create standalone notebooks**

Keep all default execution flags false in Git. The staging copy alone enables
the requested run. Do not duplicate arbitrary pickle/joblib persistence.

- [ ] **Step 4: Run notebook/static verification and commit**

Commit:

```bash
git commit -m "feat: add Phase 3 Kaggle validation notebooks"
```

---

### Task 7: Execute Validation Sequentially

**Files:**
- Create outside Git: private Kaggle source staging and downloaded run evidence.
- Modify after readback: `docs/superpowers/plans/2026-07-30-phase3-personal-context-pattern.md` checkboxes only.

**Interfaces:**
- Kaggle ids:
  - `bjcoding/multisensor-goal1-5-05-phase3-source`
  - `bjcoding/multisensor-goal1-5-06-phase3-pattern`

- [ ] **Step 1: Run full local verification**

Run:

```bash
./.venv/bin/ruff check --no-cache .
./.venv/bin/mypy src
./.venv/bin/pytest
```

- [ ] **Step 2: Upload the private bounded raw source**

Include only the three latent timelines, existing outcome labels, split
registry and hashes. Exclude hidden archetype, participant truth baseline,
event intensity and all unrelated artifacts.

- [ ] **Step 3: Push 05 on CPU and wait for COMPLETE**

Download and verify every output hash before starting 06.

- [ ] **Step 4: Push 06 on CPU and wait for COMPLETE**

Download all Phase 3 artifacts. Recompute a bounded sample of group metrics
locally and require equality.

- [ ] **Step 5: Record evidence and commit**

Do not claim a 3차 performance result until this readback passes.

---

### Task 8: Publish the Korean Phase 3 Report

**Files:**
- Create: `reports/goal15_phase3_personal_pattern_ko.artifact.json`
- Create: `reports/goal15_phase3_personal_pattern_ko.html`
- Create: `tests/test_phase3_report.py`

**Interfaces:**
- Consumes only verified Phase 3 artifacts.
- Produces one-page Korean HTML with 1·2·3차 narrative and dataset-level uplift.

- [ ] **Step 1: Write failing report-content tests**

Require:

- `개인·환경 기준선`;
- `누적 부하`;
- G0 and selected personal candidate;
- overall, run, dataset, person, context and load-state uplift;
- improvement/tie/degradation counts;
- worst dataset;
- adoption decision;
- `oracle/sanity`;
- `NOT VERIFIED`;
- explicit locked-test non-use.

- [ ] **Step 2: Run RED**

Run:

```bash
./.venv/bin/pytest tests/test_phase3_report.py -v
```

- [ ] **Step 3: Build the source-backed artifact and HTML**

Keep AUCPR/event F1 primary. Label raw accuracy as an auxiliary imbalance-
sensitive metric. Do not infer clinical utility.

- [ ] **Step 4: Verify and commit**

Run report tests, full tests, Ruff, mypy and `git diff --check`, then commit:

```bash
git commit -m "docs: publish Phase 3 personal-pattern report"
```
