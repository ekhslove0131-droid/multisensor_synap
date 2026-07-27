# KNIME Synthetic Factory and Hierarchical Pattern Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox syntax for tracking.

**Goal:** Build a KNIME-driven synthetic factory, immutable local model
registry, five-stage event model, and observed-behavior multi-label model.

**Architecture:** Python owns generation, labels, training, persistence, and
evaluation. KNIME owns configuration, stage orchestration, and Korean
visualization. Stage receipts establish real execution dependencies.

**Tech Stack:** Python 3.12, uv, pandas, PyArrow, scikit-learn, skops, SQLite,
pytest, Hypothesis, Ruff, mypy, KNIME 5.12.

## Global constraints

- Do not modify `multisensor_synth`.
- Preserve the existing Oracle Benchmark workflow and artifacts.
- Keep truth, observed, model-ready, predictions, and audit joins separate.
- Do not train directly on review tags or hidden truth fields.
- Do not report synthetic performance as real performance.
- Do not use pickle or joblib persistence.
- Test first, verify the expected failure, implement minimally, then commit.

## Tasks

- [ ] Add factory, receipt, and behavior-ontology configuration contracts.
- [ ] Generate deterministic event, stage, behavior, and review-tag artifacts.
- [ ] Add immutable SQLite registry, release, promotion, and receipt storage.
- [ ] Add standard-type discovery, capped personalization, and OOD states.
- [ ] Add event-gate, five-stage classifier, and causal state decoder.
- [ ] Add OOF-fed multi-label behavior classifiers and personal calibration.
- [ ] Add versioned Korean result routing and audit output.
- [ ] Add connected KNIME factory and training-registry workflows.
- [ ] Run quick, MVP, static checks, bundle reload, and KNIME GUI readback.

## Commit checkpoints

1. `docs: record synthetic factory architecture`
2. `feat: add deterministic synthetic outcome factory`
3. `feat: add immutable local model registry`
4. `feat: add standard types and guarded personalization`
5. `feat: add hierarchical stage model`
6. `feat: add behavior model and Korean routing`
7. `feat: add connected KNIME factory and registry workflows`
8. `test: verify Goal 1.5 factory and hierarchical pipeline`

