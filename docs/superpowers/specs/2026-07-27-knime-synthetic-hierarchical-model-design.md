# KNIME Synthetic Factory and Hierarchical Pattern Model Design

## Status

- Approved: 2026-07-27
- Implementation repository: `/Users/baital/dev/multisensor_ml`
- Generator dependency: `/Users/baital/dev/multisensor_synth` (read-only)
- Result domain: `oracle/sanity`
- Real performance: `NOT VERIFIED`
- Device synchronization: `NOT_AVAILABLE_TRUTH_ONLY`

## Purpose

Build two connected KNIME workflows:

1. `Multisensor_Synthetic_Factory_Goal1_5`
2. `Multisensor_ML_Training_Registry`

The factory runs the existing synthetic truth generator and creates a separate,
versioned outcome-label extension. The training workflow consumes the factory
receipt, learns population and personal baselines, discovers neutral standard
types, trains a pattern-stage model and a multi-label observed-behavior model,
and records immutable model releases.

The existing `Multisensor_ML_Goal1_5` workflow remains an Oracle Benchmark.

## Non-negotiable boundaries

- Event truth is scheduled before signals and labels; feature thresholds never
  create event truth.
- Hidden archetype, truth intensity, event identifiers, and future labels are
  forbidden model inputs.
- Synthetic and real labels share a schema but never share performance claims.
- `reported_meltdown_like` and `reported_sensory_seeking_like` are review tags,
  not event types and not direct training targets.
- Raw probabilities are retained for audit. OOD, low-quality, and unsupported
  results are displayed as `NOT_DECISIONABLE` / `판단 보류`.
- No device SDK, medical decision, or real behavior accuracy claim is in scope.

## Data flow

```text
generator config
→ multisensor_synth truth run
→ dataset registry and person split
→ synthetic outcome labels
→ factory receipt
→ population baseline
→ standard-type discovery
→ capped personal adaptation and OOD
→ model 1: event gate and five-stage classifier
→ model 2: observed-behavior multi-label classifiers
→ validation/stress evaluation
→ immutable candidate release
→ manual promotion
→ Korean result router
```

## Outcome label contract

Schema: `goal1.5/behavior-outcomes/v1`

Artifacts:

- `outcome_events.parquet`: one row per event and its truth boundaries.
- `outcome_stages.parquet`: one row per canonical second.
- `outcome_behaviors.parquet`: long-form event/behavior labels.
- `outcome_review_tags.parquet`: observer-style review tags and confidence.
- `event_prediction_audit.parquet`: final joined audit view; never a training
  source.

Stage mapping:

| Output | Source phase |
|---|---|
| `NO_EVENT` | outside a target-related phase |
| `LOW` | `pre_early` |
| `MEDIUM` | `pre_late`, `onset` |
| `HIGH` | `peak` |
| `DECREASING` | `recovery_early` |
| `RECOVERY` | `recovery_late`, `post` |

Initial observable behavior codes:

- `ear_covering`
- `head_turn_away`
- `withdrawal_movement`
- `exit_attempt`
- `repetitive_hand_movement`
- `repetitive_body_movement`
- `movement_reduction`
- `motion_freeze`
- `repetitive_object_contact`
- `sustained_pressure_or_contact`

Behavior labels are deterministic for a fixed seed, but probabilistically
conditioned on hidden event properties and context. Similar behavior labels
also occur on hard negatives so no event type or single feature becomes a
perfect proxy.

## Model architecture

### Baseline and standard types

Train-person baseline summaries feed GMM candidates for `K=1..6`. A candidate
must satisfy stability, minimum effective component size, soft-membership
confidence, and robust KMeans agreement. The lowest-BIC feasible candidate is
selected; otherwise the result is `K=1`. Type IDs are neutral (`STD-A`, etc.).

Personal adaptation uses the existing causal weight formula plus a
validation-selected cap. A larger cap is rejected when it materially degrades
AUCPR, event recall, lead time, false alerts, or calibration.

### Model 1

The event gate predicts `NO_EVENT` versus an event-related interval. A second
classifier predicts `LOW`, `MEDIUM`, `HIGH`, `DECREASING`, or `RECOVERY`.
Logistic Regression and HistGradientBoosting are compared on validation only.
A causal state decoder prevents implausible backward transitions while allowing
any initial state for recordings that begin mid-event.

### Model 2

Each observable behavior has one-vs-rest Logistic and HistGradientBoosting
candidates. Inputs include causal features, standard-type probabilities,
personal deviations, context, and model-1 probabilities. Train-time model-1
inputs are person-grouped out-of-fold predictions.

Unsupported behavior labels remain `INSUFFICIENT_LABEL_SUPPORT`. Personal
calibration starts only after 20 valid events and per-label positive/negative
support. Champion comparison requires at least 50 events spanning three days.

## Registry and release

SQLite stores immutable versions for datasets, labels, baselines, standard
types, stage models, behavior models, evaluations, predictions, releases,
promotion events, and stage receipts.

A release is a composite reference and moves only through:

```text
candidate → champion → retired
```

Promotion is manual and requires an audit reason. Locked test evaluation is
allowed once for the selected candidate model/dataset hash pair.

## KNIME behavior

Every component produces a one-row receipt containing the pipeline run,
stage status, artifact URI, hashes, versions, timestamps, and Korean message.
The receipt output port is connected to the next stage input. A failed or stale
receipt stops downstream execution.

The factory displays generation settings, population/baseline distribution,
event stages, behavior combinations, hard negatives, split lineage, and the
truth-only synchronization status.

The training workflow displays label QA, standard types, personal/OOD status,
stage performance, behavior performance, stress results, releases, promotion
history, and Korean prediction results.

