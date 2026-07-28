# Kaggle ML/DL Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create four safe, unexecuted Kaggle notebooks for a fair ML-versus-TCN benchmark and publish the shared MVP3 oracle training data as one private Kaggle Dataset without starting any model training.

**Architecture:** One immutable Parquet dataset supplies the same person split, labels, and causal feature whitelist to both model families. The ML path creates row-based views for Logistic Regression and HistGradientBoosting; the DL path creates bounded causal sequences for a compact multi-head TCN. Both model notebooks emit the same prediction and metric contracts so validation results can be compared without using locked-test results for model selection.

**Tech Stack:** Python notebook format v4, Python 3.12-compatible code, pandas, PyArrow, NumPy, scikit-learn, PyTorch DDP, W&B, Kaggle CLI 2.2.4, pytest, Ruff

## Global Constraints

- Repository: `/Users/baital/dev/multisensor_ml`.
- Source series: `mvp3-oracle-v1`, 36 synthetic people, 15,552,000 1 Hz rows.
- Split is immutable: train 24, validation 6, locked test 6, zero person overlap.
- Data status is always `oracle/sanity`; real accuracy is `NOT VERIFIED`.
- Device synchronization is `NOT_AVAILABLE_TRUTH_ONLY`.
- One private Kaggle Dataset is shared by both model families.
- Exactly four benchmark notebooks are created under `kaggle/`.
- Notebook outputs are empty and every `execution_count` is `null`.
- No notebook is pushed or executed during implementation.
- No model training, locked-test evaluation, W&B login, or GPU allocation is performed.
- `WANDB_API_KEY` is read only from Kaggle Secrets and never stored.
- ML is CPU-only; DL fails closed before training unless exactly two CUDA devices exist.
- AUCPR is the primary selection metric; AUROC is secondary.
- Validation chooses candidates and thresholds; locked test never chooses a model.
- Existing unrelated worktree changes must be preserved.

---

## File Map

**Create**

- `kaggle/01_ml_data.ipynb` — validate the shared Dataset and write row-based ML views.
- `kaggle/02_ml_benchmark.ipynb` — train/evaluate Logistic Regression and HistGradientBoosting when explicitly enabled.
- `kaggle/03_dl_sequence_data.ipynb` — build deterministic 300/600-second causal sequence indexes and caches.
- `kaggle/04_dl_tcn_benchmark.ipynb` — define and optionally train the dual-T4 TCN.
- `tests/test_kaggle_notebooks.py` — statically validate notebook safety, contracts, and cross-notebook consistency.

**Modify**

- `.gitignore` — ignore local notebook caches, W&B state, Kaggle staging, checkpoints, and downloaded artifacts.

**Temporary, never committed**

- `/private/tmp/multisensor-goal15-oracle-mvp3-upload/` — hard-link upload staging.
- `datasets-metadata.json` and `dataset-card.md` inside that staging directory.

## Shared Notebook Contracts

Every notebook declares:

```python
SERIES_ID = "mvp3-oracle-v1"
EXPECTED_SPLIT_COUNTS = {"train": 24, "validation": 6, "locked_test": 6}
DATA_STATUS = "oracle/sanity"
REAL_ACCURACY_STATUS = "NOT VERIFIED"
RUN_TRAINING = False
RUN_LOCKED_TEST = False
```

Both model notebooks write prediction rows with:

```python
PREDICTION_COLUMNS = [
    "model_family",
    "model_name",
    "series_id",
    "dataset_id",
    "run_id",
    "person_key",
    "canonical_time",
    "split_role",
    "target",
    "label",
    "probability",
    "threshold",
]
```

Both model notebooks write metric rows with:

```python
METRIC_COLUMNS = [
    "model_family",
    "model_name",
    "series_id",
    "split_role",
    "target",
    "metric",
    "value",
    "support",
    "data_status",
]
```

---

### Task 1: Notebook Safety and Structure Contract

**Files:**
- Create: `tests/test_kaggle_notebooks.py`
- Create: `kaggle/01_ml_data.ipynb`
- Create: `kaggle/02_ml_benchmark.ipynb`
- Create: `kaggle/03_dl_sequence_data.ipynb`
- Create: `kaggle/04_dl_tcn_benchmark.ipynb`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: notebook filenames and global constants from this plan.
- Produces: four syntactically valid notebook-v4 JSON documents with stable section headings and no execution state.

- [ ] **Step 1: Write the failing notebook structure tests**

Add tests that parse each notebook with `json.loads`, assert
`nbformat == 4`, require at least one markdown and one code cell, and enforce:

```python
NOTEBOOKS = [
    "01_ml_data.ipynb",
    "02_ml_benchmark.ipynb",
    "03_dl_sequence_data.ipynb",
    "04_dl_tcn_benchmark.ipynb",
]

def test_notebooks_are_unexecuted() -> None:
    for notebook in load_notebooks():
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert cell["outputs"] == []

def test_notebooks_do_not_embed_secrets() -> None:
    forbidden = ("wandb.ai/authorize", "kaggle.json", "api_key=", "WANDB_API_KEY=")
    for path, notebook in load_notebooks().items():
        source = notebook_source(notebook)
        assert not any(token in source for token in forbidden), path
```

Also require the exact shared status constants and `RUN_TRAINING = False`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
uv run pytest tests/test_kaggle_notebooks.py -v
```

Expected: FAIL because `kaggle/*.ipynb` do not exist.

- [ ] **Step 3: Create the minimum safe notebook shells**

Create four notebook-v4 JSON files. Each contains:

- a Korean title and an English machine-readable purpose;
- an oracle/sanity warning;
- one setup code cell with the exact shared constants;
- `metadata.kernelspec.name = "python3"`;
- empty code outputs and null execution counts.

Append these ignore patterns:

```gitignore
kaggle/.ipynb_checkpoints/
kaggle/cache/
kaggle/checkpoints/
kaggle/wandb/
kaggle/staging/
wandb/
```

- [ ] **Step 4: Run tests and verify GREEN**

Run:

```bash
uv run pytest tests/test_kaggle_notebooks.py -v
uv run ruff check tests/test_kaggle_notebooks.py
```

Expected: PASS with no warnings.

- [ ] **Step 5: Commit**

```bash
git add .gitignore kaggle tests/test_kaggle_notebooks.py
git commit -m "test: define safe Kaggle notebook contracts"
```

---

### Task 2: Machine-Learning Data Notebook

**Files:**
- Modify: `kaggle/01_ml_data.ipynb`
- Modify: `tests/test_kaggle_notebooks.py`

**Interfaces:**
- Consumes: flat Kaggle files prefixed by `prepared__`, `outcomes__`, and `registry__`.
- Produces: `/kaggle/working/goal15_ml_view/{train,validation,locked_test}.parquet` and `view_manifest.json`.

- [ ] **Step 1: Write failing static contract tests**

Extract and compile all code cells, then assert definitions for:

```python
resolve_kaggle_dataset_root
load_split_registry
validate_split_contract
validate_manifest_hashes
assert_no_truth_leakage
build_ml_role_view
write_ml_view_manifest
```

Assert that source contains:

```python
ML_OUTPUT_ROOT = Path("/kaggle/working/goal15_ml_view")
ORACLE_FEATURE_DENYLIST = (
    "active_target_",
    "hard_negative_id",
    "hard_negative_type",
    "artifact_schedule_id",
    "participant_truth_baseline",
    "event_intensity_truth",
)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
uv run pytest tests/test_kaggle_notebooks.py::test_ml_data_contract -v
```

Expected: FAIL because the functions are absent.

- [ ] **Step 3: Implement dataset validation and row-view creation**

Add cells implementing:

```python
def validate_split_contract(split: pd.DataFrame) -> None:
    counts = split.groupby("split_role")["person_key"].nunique().to_dict()
    if counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"split mismatch: {counts}")
    overlaps = [
        set(split.loc[split["split_role"] == role, "person_key"])
        for role in EXPECTED_SPLIT_COUNTS
    ]
    if any(overlaps[i] & overlaps[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("person leakage across split roles")

def assert_no_truth_leakage(columns: Iterable[str]) -> None:
    leaked = [
        column for column in columns
        if any(column == token or column.startswith(token) for token in ORACLE_FEATURE_DENYLIST)
    ]
    if leaked:
        raise ValueError(f"truth leakage columns: {sorted(leaked)}")
```

`build_ml_role_view` must join labels by stable person/time keys, retain all
positive and hard-negative rows, sample deterministic matched baselines up to
three per positive for train, and retain the full validation/locked-test
timeline. It writes ZSTD Parquet and records SHA-256, row count, columns,
source dataset hash, and split hash in `view_manifest.json`.

The final orchestration cell is guarded:

```python
RUN_DATA_PREPARATION = False
if RUN_DATA_PREPARATION:
    build_all_ml_views()
else:
    print("준비 완료: RUN_DATA_PREPARATION=True로 바꿀 때만 데이터를 생성합니다.")
```

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run pytest tests/test_kaggle_notebooks.py -v
uv run ruff check tests/test_kaggle_notebooks.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add kaggle/01_ml_data.ipynb tests/test_kaggle_notebooks.py
git commit -m "feat: add Kaggle ML data notebook"
```

---

### Task 3: Machine-Learning Benchmark Notebook

**Files:**
- Modify: `kaggle/02_ml_benchmark.ipynb`
- Modify: `tests/test_kaggle_notebooks.py`

**Interfaces:**
- Consumes: ML role views and `view_manifest.json` from Task 2.
- Produces: common-schema predictions, validation metrics, optional locked-test metrics, and W&B logs.

- [ ] **Step 1: Write failing model-notebook contract tests**

Require compiled definitions:

```python
load_ml_views
verify_ml_view_manifest
fit_logistic_candidate
fit_hgb_candidate
select_validation_threshold
compute_common_metrics
bootstrap_people_ci
select_validation_champion
login_wandb_from_kaggle_secret
```

Assert `CUDA_VISIBLE_DEVICES` is absent, `device = "cpu"` is present,
`WANDB_API_KEY` appears only as the argument to `UserSecretsClient().get_secret`,
and both `RUN_TRAINING` and `RUN_LOCKED_TEST` default to `False`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
uv run pytest tests/test_kaggle_notebooks.py::test_ml_benchmark_contract -v
```

Expected: FAIL because model functions are absent.

- [ ] **Step 3: Implement ML candidates and common evaluation**

Implement Logistic Regression and HistGradientBoosting candidates for:

- binary event probability;
- conditional five-stage one-vs-rest probabilities;
- ten behavior-code one-vs-rest probabilities.

Use validation AUCPR as the primary comparator. Break ties by event recall,
then lower false alerts/hour, then lower ECE. Report, but never select by,
locked-test metrics.

W&B login must be isolated:

```python
def login_wandb_from_kaggle_secret() -> bool:
    try:
        from kaggle_secrets import UserSecretsClient
        import wandb
        key = UserSecretsClient().get_secret("WANDB_API_KEY")
    except Exception as exc:
        print(f"W&B 비활성화: {type(exc).__name__}")
        return False
    return bool(wandb.login(key=key, verify=True))
```

The training gate must be:

```python
if not RUN_TRAINING:
    print("학습 비활성화: RUN_TRAINING=False")
elif not validated_manifest:
    raise RuntimeError("검증된 ML view manifest가 필요합니다.")
else:
    run_ml_training()
```

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run pytest tests/test_kaggle_notebooks.py -v
uv run ruff check tests/test_kaggle_notebooks.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add kaggle/02_ml_benchmark.ipynb tests/test_kaggle_notebooks.py
git commit -m "feat: add Kaggle ML benchmark notebook"
```

---

### Task 4: Deep-Learning Sequence Data Notebook

**Files:**
- Modify: `kaggle/03_dl_sequence_data.ipynb`
- Modify: `tests/test_kaggle_notebooks.py`

**Interfaces:**
- Consumes: the same flat Kaggle Dataset and immutable splits as Task 2.
- Produces: sequence-index Parquet files, normalized feature statistics fitted on train only, and `sequence_manifest.json`.

- [ ] **Step 1: Write failing sequence-contract tests**

Require definitions:

```python
validate_shared_dataset_identity
fit_train_normalization
make_causal_window_index
assert_window_boundaries
sample_training_windows
write_sequence_manifest
```

Assert:

```python
SEQUENCE_LENGTHS_SECONDS = (300, 600)
SEQUENCE_OUTPUT_ROOT = Path("/kaggle/working/goal15_dl_sequences")
```

and source-level conditions that `window_end >= window_start`,
`person_key` is part of every grouping key, and normalization is fitted only
where `split_role == "train"`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
uv run pytest tests/test_kaggle_notebooks.py::test_dl_sequence_contract -v
```

Expected: FAIL because sequence functions are absent.

- [ ] **Step 3: Implement causal sequence indexing**

Implement `make_causal_window_index` so every sample satisfies:

```python
window_start = prediction_time - pd.Timedelta(seconds=length_seconds - 1)
window_end = prediction_time
```

Reject windows that cross a `person_key`, `run_id`, `dataset_id`, missing block,
or day/session boundary declared by the source data. Fit median/IQR
normalization on train people only and persist the fitted statistics with a
source hash.

Training indexes contain deterministic positive-centered, hard-negative, and
matched-baseline windows. Validation and locked-test indexes are deterministic
sliding windows and are not class-balanced.

The final cell is guarded by `RUN_DATA_PREPARATION = False`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run pytest tests/test_kaggle_notebooks.py -v
uv run ruff check tests/test_kaggle_notebooks.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add kaggle/03_dl_sequence_data.ipynb tests/test_kaggle_notebooks.py
git commit -m "feat: add Kaggle DL sequence notebook"
```

---

### Task 5: Dual-T4 TCN Benchmark Notebook

**Files:**
- Modify: `kaggle/04_dl_tcn_benchmark.ipynb`
- Modify: `tests/test_kaggle_notebooks.py`

**Interfaces:**
- Consumes: sequence indexes, train-only normalization, and `sequence_manifest.json` from Task 4.
- Produces: common-schema predictions/metrics, validation champion data, optional locked-test results, and W&B logs.

- [ ] **Step 1: Write failing TCN safety and architecture tests**

Require definitions:

```python
require_exactly_two_cuda_devices
setup_ddp
cleanup_ddp
CausalConvBlock
Goal15TCN
Goal15SequenceDataset
masked_multitask_loss
train_one_epoch
evaluate_common_schema
login_wandb_from_kaggle_secret
```

AST assertions must verify:

- `torch.cuda.device_count() != 2` raises before `setup_ddp`;
- `DistributedDataParallel` and `DistributedSampler` are used;
- `RUN_TRAINING = False`;
- `RUN_LOCKED_TEST = False`;
- event, stage, and behavior heads are defined;
- no call to the training entrypoint occurs outside a `RUN_TRAINING` guard.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
uv run pytest tests/test_kaggle_notebooks.py::test_dl_tcn_contract -v
```

Expected: FAIL because TCN functions/classes are absent.

- [ ] **Step 3: Implement compact causal TCN and DDP gate**

Define a 0.5-5 million parameter TCN with left-only causal padding, residual
dilated convolution blocks, masked pooling, and three heads:

```python
self.event_head = nn.Linear(hidden_size, 1)
self.stage_head = nn.Linear(hidden_size, 5)
self.behavior_head = nn.Linear(hidden_size, 10)
```

The loss is:

```python
total = (
    event_bce
    + stage_weight * stage_cross_entropy_on_event_rows
    + behavior_weight * behavior_bce_on_event_decision_rows
)
```

Use `torchrun`-compatible environment variables, NCCL, one process per GPU,
`DistributedSampler`, AMP, gradient clipping, deterministic seeds, and rank-0
W&B logging. Fail closed:

```python
def require_exactly_two_cuda_devices() -> None:
    count = torch.cuda.device_count()
    if count != 2:
        raise RuntimeError(f"T4 x2가 필요합니다. 감지된 CUDA 장치: {count}")
```

Champion selection uses validation AUCPR and the same tie-break rules as Task
3. Include a final optional comparison cell that joins ML and DL validation
metric files by `target` and `metric` and writes
`model_comparison_validation.parquet`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
uv run pytest tests/test_kaggle_notebooks.py -v
uv run ruff check tests/test_kaggle_notebooks.py
```

Expected: PASS without executing notebook cells.

- [ ] **Step 5: Commit**

```bash
git add kaggle/04_dl_tcn_benchmark.ipynb tests/test_kaggle_notebooks.py
git commit -m "feat: add dual-T4 TCN benchmark notebook"
```

---

### Task 6: Full Static Verification

**Files:**
- Modify only if verification exposes a defect in the five files from Tasks 1-5.

**Interfaces:**
- Consumes: all four completed notebooks.
- Produces: verified local notebook artifacts with no execution state or secrets.

- [ ] **Step 1: Run notebook-specific verification**

```bash
uv run pytest tests/test_kaggle_notebooks.py -v
uv run ruff check tests/test_kaggle_notebooks.py
```

Expected: all notebook contract tests PASS.

- [ ] **Step 2: Run the complete repository suite**

```bash
uv lock --check
uv run ruff check .
uv run mypy src
uv run pytest
```

Expected: all commands exit 0.

- [ ] **Step 3: Perform secret and execution-state scans**

```bash
rg -n 'sk-[A-Za-z0-9]|WANDB_API_KEY\\s*=|\"execution_count\": [0-9]' kaggle
```

Expected: no credential-like assignment, executed cell, or nonempty output.
The literal secret name used in `get_secret("WANDB_API_KEY")` is allowed.

- [ ] **Step 4: Verify the scoped diff**

```bash
git status --short
git diff --check
git diff --stat HEAD~4..HEAD
```

Expected: only planned notebook, test, ignore, spec, and plan files.

---

### Task 7: Private Kaggle Dataset Staging and Upload

**Files:**
- Read: `data/prepared/mvp3-oracle-v1/**`
- Read: `data/outcomes/mvp3-oracle-v1/**`
- Read: `data/registry/mvp3-oracle-v1/**`
- Create temporarily: `/private/tmp/multisensor-goal15-oracle-mvp3-upload/**`

**Interfaces:**
- Consumes: existing immutable MVP3 Parquet and manifest files.
- Produces: one private Kaggle Dataset version and authenticated readback evidence.

- [ ] **Step 1: Confirm authentication without printing credentials**

Run:

```bash
test -n "${KAGGLE_API_TOKEN:-}" && echo "KAGGLE_API_TOKEN=present"
kaggle quota
```

Expected: token presence and authenticated quota output. Do not run
`kaggle auth print-access-token` or print environment values.

- [ ] **Step 2: Create a same-volume temporary hard-link staging tree**

Create `/private/tmp/multisensor-goal15-oracle-mvp3-upload/` with flat,
collision-free filenames:

```text
prepared__people__${dataset_id}.parquet
prepared__personal_baseline.parquet
prepared__manifest.json
outcomes__outcome_events.parquet
outcomes__outcome_stages.parquet
outcomes__outcome_behaviors.parquet
outcomes__outcome_review_tags.parquet
outcomes__manifest.json
registry__splits.parquet
registry__registry.jsonl
registry__manifest.json
```

Use hard links, not copies, and verify every staged data file has the same
SHA-256 as its source.

- [ ] **Step 3: Resolve the authenticated owner and generate safe metadata**

Resolve the owner without displaying the token:

```bash
kaggle_owner="$(kaggle config view | sed -n 's/^username: //p')"
test -n "${kaggle_owner}"
staging_dir="/private/tmp/multisensor-goal15-oracle-mvp3-upload"
kaggle datasets init -p "${staging_dir}"
```

If `kaggle_owner` is empty, stop before creating metadata or uploading. Set
the generated `datasets-metadata.json` to:

```json
{
  "title": "Multisensor Goal 1.5 Oracle MVP3",
  "id": "${kaggle_owner}/multisensor-goal15-oracle-mvp3",
  "licenses": [{"name": "other"}],
  "keywords": ["synthetic-data", "time-series", "wearable-sensors", "machine-learning"]
}
```

Create `dataset-card.md` with the exact status warnings and dataset statistics
from the approved design. No personal data or credential is included.

- [ ] **Step 4: Pre-upload audit**

Verify:

```bash
du -sh /private/tmp/multisensor-goal15-oracle-mvp3-upload
find /private/tmp/multisensor-goal15-oracle-mvp3-upload -type f | wc -l
find /private/tmp/multisensor-goal15-oracle-mvp3-upload -type l | wc -l
```

Expected: approximately 9.5 GB logical size, the exact planned file count, and
zero symbolic links. Confirm the target Dataset identifier does not already
exist; if it exists with a different manifest hash, stop instead of replacing
or versioning it.

- [ ] **Step 5: Upload privately without CSV conversion**

Run:

```bash
kaggle datasets create \
  --path /private/tmp/multisensor-goal15-oracle-mvp3-upload \
  --keep-tabular \
  --dir-mode skip
```

Do not pass `--public`. Expected: upload accepted and a Dataset URL returned.
This command uploads a Dataset only and does not allocate GPU or run a notebook.

- [ ] **Step 6: Authenticated remote readback**

Run:

```bash
kaggle datasets status "${kaggle_owner}/multisensor-goal15-oracle-mvp3"
kaggle datasets files "${kaggle_owner}/multisensor-goal15-oracle-mvp3"
kaggle datasets metadata "${kaggle_owner}/multisensor-goal15-oracle-mvp3" \
  --path /private/tmp/multisensor-goal15-oracle-mvp3-readback
```

Verify private status, version 1, title, file inventory, total size, and the
absence of model artifacts, credentials, caches, or notebook outputs.

- [ ] **Step 7: Record upload evidence without credentials**

Add a short section to `docs/RESULTS_HIERARCHICAL_MVP3.md` containing the
Dataset identifier, version, private status, file count, logical size, upload
timestamp, and readback result. Do not include token fragments or local secret
paths.

- [ ] **Step 8: Commit upload evidence**

```bash
git add docs/RESULTS_HIERARCHICAL_MVP3.md
git commit -m "docs: record private Kaggle dataset upload"
```

---

### Task 8: Final Scope and Remote-Safety Readback

**Files:**
- Read only, unless an evidence mismatch requires a narrow correction.

**Interfaces:**
- Consumes: local Git state, remote Kaggle Dataset metadata, four notebooks.
- Produces: final evidence-backed handoff with no model run.

- [ ] **Step 1: Re-run local verification**

```bash
uv lock --check
uv run ruff check .
uv run mypy src
uv run pytest
git status --short --branch
```

Expected: all validation passes and only pre-existing unrelated changes, if
any, remain.

- [ ] **Step 2: Confirm no remote notebook or experiment was started**

Do not call `kaggle kernels push`. Confirm no W&B run directory exists under
the repository and no model checkpoint was produced:

```bash
find . -type d -name wandb -o -type f -name '*.ckpt' -o -type f -name '*.pt'
```

Expected: no new W&B state or trained checkpoint.

- [ ] **Step 3: Report exact outcomes**

Report:

- four local notebook paths and commit SHAs;
- private Kaggle Dataset identifier, URL, version, and readback status;
- test commands and outcomes;
- `training: NOT STARTED`;
- `GPU allocation: NOT REQUESTED`;
- `W&B login: NOT EXECUTED`;
- `real accuracy: NOT VERIFIED`.
