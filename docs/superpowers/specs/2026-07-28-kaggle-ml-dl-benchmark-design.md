# Kaggle ML/DL Benchmark Design

Date: 2026-07-28  
Status: approved for implementation  
Scope: Goal 1.5 oracle/sanity only

## Objective

Publish one private Kaggle Dataset from the existing `mvp3-oracle-v1`
training data and create four local Kaggle notebooks that compare a classical
machine-learning baseline with a causal deep-learning model. Dataset upload is
in scope. Notebook upload, GPU allocation, W&B authentication, and training
execution are not.

The benchmark must not be described as real-device or medical performance.
Real accuracy remains `NOT VERIFIED`.

## Deliverables

The repository will contain exactly four benchmark notebooks:

1. `kaggle/01_ml_data.ipynb`
   - locate the attached Kaggle Dataset;
   - validate manifests and required Parquet files;
   - verify the deterministic person split is train 24, validation 6, and
     locked test 6 with no person overlap;
   - enforce the oracle feature whitelist and truth-leakage denylist;
   - create deterministic row-based ML training views and bounded caches.

2. `kaggle/02_ml_benchmark.ipynb`
   - consume only notebook 1 outputs or recreate the same validated view;
   - compare Logistic Regression and HistGradientBoosting;
   - train event, five-stage, and behavior one-vs-rest targets;
   - select thresholds on validation only;
   - contain optional W&B login code using the Kaggle
     `WANDB_API_KEY` secret;
   - remain CPU-only and not consume a Kaggle GPU allocation.

3. `kaggle/03_dl_sequence_data.ipynb`
   - validate the same Dataset identity and split hashes;
   - construct causal 300- and 600-second sequence windows without crossing
     person, run, or dataset boundaries;
   - sample positive, hard-negative, and matched-baseline windows
     deterministically;
   - keep validation and locked-test timelines deterministic and unsampled for
     final comparison;
   - write local Kaggle working caches only.

4. `kaggle/04_dl_tcn_benchmark.ipynb`
   - define a compact dilated causal TCN shared backbone;
   - provide event, five-stage, and ten-code behavior heads;
   - require exactly two CUDA devices before any training cell can continue;
   - configure single-node PyTorch DistributedDataParallel for T4 x2;
   - use mixed precision and deterministic seeds;
   - contain optional W&B login code using the Kaggle
     `WANDB_API_KEY` secret;
   - define training and evaluation cells but not execute them.

Notebook cell outputs and execution counts must be empty in Git so creating the
files cannot be confused with having run an experiment.

## Dataset Contract

One private Kaggle Dataset named `Multisensor Goal 1.5 Oracle MVP3` will contain
the immutable data needed by both model families:

- `prepared/mvp3-oracle-v1/`;
- `outcomes/mvp3-oracle-v1/`;
- `registry/mvp3-oracle-v1/`;
- a Kaggle dataset metadata file and a concise dataset card.

Model artifacts, validation predictions, locked-test predictions, caches,
`.venv`, credentials, W&B files, and Git metadata are excluded. The upload
must preserve Parquet files without CSV conversion. The Dataset description
must state:

- synthetic truth only;
- `oracle/sanity`;
- 36 synthetic people;
- train 24 / validation 6 / locked test 6;
- real accuracy `NOT VERIFIED`;
- device synchronization `NOT_AVAILABLE_TRUTH_ONLY`;
- no medical or diagnostic interpretation.

The existing files are approximately 9.5 GB, 15,552,000 1 Hz rows, 155
prepared columns, 930 events, and 2,416 positive behavior-label rows.

## Common Comparison Contract

Both model families must use:

- the exact same dataset, label-set, and split hashes;
- train people only for fitting;
- validation people only for model, hyperparameter, and threshold selection;
- locked-test people only for the final selected candidate;
- the same output keys at one-second resolution;
- person-grouped evaluation and confidence intervals;
- no participant overlap across roles.

The model inputs may differ only in representation:

- ML: one row of causal derived features per prediction time;
- DL: a causal sequence ending at the same prediction time.

Both outputs will be normalized into a common prediction schema before metric
calculation.

## Model and Metric Design

### Machine Learning

Logistic Regression is the interpretable linear baseline.
HistGradientBoosting is the nonlinear tabular baseline. Existing Goal 1.5
feature sampling rules remain in force: all positives and hard negatives plus
at most three deterministic matched-baseline rows per positive for training.

### Deep Learning

The first deep-learning candidate is a compact TCN rather than a Transformer.
Dilated causal convolutions match the 5-300 second feature horizons, preserve
causality, parallelize efficiently, and reduce overfitting risk for only 36
synthetic people. The initial parameter budget is 0.5-5 million parameters.

The model has a shared TCN backbone and three heads:

- binary event probability;
- five conditional event-stage probabilities:
  `LOW`, `MEDIUM`, `HIGH`, `DECREASING`, `RECOVERY`;
- ten sigmoid behavior probabilities.

`NO_EVENT` is represented by the event head being negative. Stage loss is
masked outside event intervals. Behavior loss is applied at event-level
decision points using the immutable behavior labels.

### Evaluation

AUCPR is the primary selection metric because the targets are imbalanced.
AUROC is reported as a secondary metric. The shared report includes:

- AUCPR and AUROC;
- event recall and event-level F1;
- false alerts per hour;
- Brier score and expected calibration error;
- mean and median forecast lead time where applicable;
- stage macro-F1, balanced accuracy, per-stage recall, and confusion matrix;
- behavior micro/macro AUCPR, AUROC, and F1 with label support;
- person-grouped bootstrap 95% confidence intervals;
- deterministic noise-stress degradation.

W&B runs use project `multisensor-goal15-benchmark`, groups
`machine-learning` and `deep-learning-tcn`, and tags `oracle-sanity`, `mvp3`,
`split-24-6-6`, and `not-real-verified`.

## Secrets and Execution Safety

No credential is stored in a notebook, output, dataset, manifest, Git file, or
W&B config. The notebook retrieves `WANDB_API_KEY` from Kaggle Secrets only
inside an explicitly marked optional login cell. A missing secret disables W&B
without blocking local metric export.

The deep-learning notebook checks `torch.cuda.device_count() == 2` before
initializing W&B or constructing a trainer. Failure is fail-closed and produces
an instruction to select a T4 x2 accelerator. Creating or uploading the private
Dataset does not run either notebook.

Kaggle notebook push is outside this step because pushing a kernel can start a
remote run. Only the Dataset is uploaded after local manifest, file-count, and
hash validation.

## Validation and Acceptance

- Four valid `.ipynb` files with empty outputs and execution counts.
- No embedded Kaggle or W&B credential-like strings.
- Both data notebooks validate the same dataset and split hashes.
- ML input is row-based; DL input is causal sequence-based.
- Sequence windows never cross person, run, or dataset boundaries.
- The DL notebook aborts before training unless exactly two CUDA devices exist.
- The ML notebook is CPU-only.
- Both model notebooks emit the same prediction and metric schemas.
- Locked-test execution is isolated behind an explicit final-evaluation gate.
- Dataset upload is private and uses Parquet-preserving options.
- Authenticated Kaggle readback confirms the created Dataset identifier,
  privacy state, version, and file inventory.
- No model training, W&B login, notebook push, or GPU allocation occurs during
  implementation and upload.
