from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import GroupKFold

from multisensor_ml.metrics import evaluate_probabilities, select_event_threshold
from multisensor_ml.models import (
    ProbabilityClassifier,
    fit_candidate_models,
    select_training_rows,
)

MODEL_STAGE_CODES: tuple[str, ...] = (
    "LOW",
    "MEDIUM",
    "HIGH",
    "DECREASING",
    "RECOVERY",
)
BehaviorSupport = Literal["SUPPORTED", "INSUFFICIENT_LABEL_SUPPORT"]
PersonalCalibrationStatus = Literal[
    "GLOBAL_STD_ONLY",
    "CALIBRATION_CANDIDATE",
    "CHAMPION_COMPARISON_ALLOWED",
]

_ALLOWED_NEXT: dict[str, frozenset[str]] = {
    "NO_EVENT": frozenset({"LOW"}),
    "LOW": frozenset({"LOW", "MEDIUM", "NO_EVENT"}),
    "MEDIUM": frozenset({"MEDIUM", "HIGH", "DECREASING"}),
    "HIGH": frozenset({"HIGH", "DECREASING"}),
    "DECREASING": frozenset({"DECREASING", "RECOVERY"}),
    "RECOVERY": frozenset({"RECOVERY", "NO_EVENT", "LOW"}),
}


@dataclass(frozen=True, slots=True)
class OOFResult:
    probability: NDArray[np.float64]
    fold: NDArray[np.int16]


@dataclass(frozen=True, slots=True)
class StageCandidateResult:
    event_models: Mapping[str, ProbabilityClassifier]
    stage_models: Mapping[str, ProbabilityClassifier]
    selected_event_model: str
    selected_stage_model: str
    event_threshold: float
    metrics: pd.DataFrame


@dataclass(frozen=True, slots=True)
class BehaviorCandidateResult:
    models: Mapping[tuple[str, str], ProbabilityClassifier]
    selected_model_by_behavior: Mapping[str, str]
    status_by_behavior: Mapping[str, BehaviorSupport]
    metrics: pd.DataFrame


def decode_stage_sequence(
    event_probability: NDArray[np.float64],
    stage_probability: NDArray[np.float64],
    *,
    event_threshold: float,
    valid_mask: NDArray[np.bool_] | None = None,
) -> list[str]:
    """Decode event and five-stage probabilities with a causal transition guard."""

    if stage_probability.shape != (len(event_probability), len(MODEL_STAGE_CODES)):
        raise ValueError("stage_probability must have shape (rows, 5)")
    if valid_mask is None:
        valid_mask = np.ones(len(event_probability), dtype=bool)
    if len(valid_mask) != len(event_probability):
        raise ValueError("valid_mask length must match probabilities")

    output: list[str] = []
    previous: str | None = None
    for index, event_score in enumerate(event_probability):
        if not bool(valid_mask[index]):
            output.append("NOT_DECISIONABLE")
            previous = None
            continue
        if float(event_score) < event_threshold:
            output.append("NO_EVENT")
            previous = "NO_EVENT"
            continue
        desired = MODEL_STAGE_CODES[int(np.argmax(stage_probability[index]))]
        if previous is None:
            selected = desired
        else:
            allowed = _ALLOWED_NEXT[previous]
            if desired in allowed:
                selected = desired
            else:
                candidates = [
                    (float(stage_probability[index, stage_index]), stage)
                    for stage_index, stage in enumerate(MODEL_STAGE_CODES)
                    if stage in allowed
                ]
                selected = max(candidates)[1] if candidates else desired
        output.append(selected)
        previous = selected
    return output


def grouped_oof_probabilities(
    features: NDArray[np.float32],
    target: NDArray[np.int8],
    groups: NDArray[np.str_] | list[str],
    *,
    random_state: int,
    folds: int = 4,
) -> OOFResult:
    """Produce person-grouped OOF probabilities for downstream behavior models."""

    group_array = np.asarray(groups, dtype=str)
    unique_groups = np.unique(group_array)
    split_count = min(folds, len(unique_groups))
    if split_count < 2:
        raise ValueError("OOF prediction requires at least two person groups")
    probability = np.full(len(target), np.nan, dtype=np.float64)
    fold_assignment = np.full(len(target), -1, dtype=np.int16)
    splitter = GroupKFold(n_splits=split_count)
    for fold, (train_index, holdout_index) in enumerate(
        splitter.split(features, target, group_array)
    ):
        train_target = target[train_index]
        if len(np.unique(train_target)) < 2:
            probability[holdout_index] = float(train_target.mean())
        else:
            model = LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=random_state + fold,
                solver="lbfgs",
            )
            model.fit(features[train_index], train_target)
            probability[holdout_index] = model.predict_proba(
                features[holdout_index]
            )[:, 1]
        fold_assignment[holdout_index] = fold
    if np.isnan(probability).any() or (fold_assignment < 0).any():
        raise RuntimeError("OOF prediction did not cover every row")
    return OOFResult(probability=probability, fold=fold_assignment)


def grouped_oof_stage_probabilities(
    features: NDArray[np.float32],
    target: NDArray[np.int8],
    groups: NDArray[np.str_] | list[str],
    *,
    random_state: int,
    folds: int = 4,
) -> OOFResult:
    """Produce five-class person-grouped OOF probabilities for model 2."""

    group_array = np.asarray(groups, dtype=str)
    unique_groups = np.unique(group_array)
    split_count = min(folds, len(unique_groups))
    if split_count < 2:
        raise ValueError("OOF prediction requires at least two person groups")
    probability = np.zeros((len(target), len(MODEL_STAGE_CODES)), dtype=np.float64)
    fold_assignment = np.full(len(target), -1, dtype=np.int16)
    splitter = GroupKFold(n_splits=split_count)
    for fold, (train_index, holdout_index) in enumerate(
        splitter.split(features, target, group_array)
    ):
        train_target = target[train_index]
        classes = np.unique(train_target)
        if len(classes) < 2:
            probability[holdout_index, int(classes[0])] = 1.0
        else:
            model = LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=random_state + fold,
                solver="lbfgs",
            )
            model.fit(features[train_index], train_target)
            fold_probability = model.predict_proba(features[holdout_index])
            for source_index, class_code in enumerate(model.classes_):
                probability[holdout_index, int(class_code)] = fold_probability[
                    :, source_index
                ]
        fold_assignment[holdout_index] = fold
    if (fold_assignment < 0).any():
        raise RuntimeError("OOF prediction did not cover every row")
    row_sum = probability.sum(axis=1)
    if np.any(row_sum <= 0):
        raise RuntimeError("OOF stage prediction produced an empty probability row")
    probability /= row_sum[:, None]
    return OOFResult(probability=probability, fold=fold_assignment)


def behavior_support_status(
    *,
    train_positive: int,
    validation_positive: int,
) -> BehaviorSupport:
    if train_positive < 20 or validation_positive < 5:
        return "INSUFFICIENT_LABEL_SUPPORT"
    return "SUPPORTED"


def personal_calibration_status(
    *,
    event_count: int,
    positive_count: int,
    negative_count: int,
    distinct_days: int,
) -> PersonalCalibrationStatus:
    if event_count < 20 or positive_count < 5 or negative_count < 5:
        return "GLOBAL_STD_ONLY"
    if event_count >= 50 and distinct_days >= 3:
        return "CHAMPION_COMPARISON_ALLOWED"
    return "CALIBRATION_CANDIDATE"


def fit_stage_candidates(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    random_state: int,
) -> StageCandidateResult:
    """Fit and select the event gate and conditional five-stage classifier."""

    _require_columns(
        train,
        {
            *feature_names,
            "stage_code",
            "hard_negative",
            "person_key",
            "timestamp_utc",
            "context",
        },
    )
    _require_columns(validation, {*feature_names, "stage_code"})
    train_event = train.assign(
        event_gate=(train["stage_code"].astype(str) != "NO_EVENT").astype(np.int8)
    )
    validation_truth = np.asarray(
        validation["stage_code"].astype(str) != "NO_EVENT", dtype=np.int8
    )
    selected_rows = select_training_rows(train_event, target="event_gate")
    event_models = fit_candidate_models(
        train_event.loc[selected_rows, list(feature_names)].to_numpy(dtype=np.float32),
        train_event.loc[selected_rows, "event_gate"].to_numpy(dtype=np.int8),
        random_state=random_state,
    )

    metric_rows: list[dict[str, object]] = []
    event_scores: list[tuple[float, float, float, str, float]] = []
    duration_hours = max(len(validation) / 3600, 1 / 3600)
    validation_features = validation[list(feature_names)].to_numpy(dtype=np.float32)
    event_thresholds: dict[str, float] = {}
    for model_name, model in event_models.items():
        probability = model.predict_proba(validation_features)[:, 1].astype(np.float64)
        threshold = select_event_threshold(
            validation_truth,
            probability,
            duration_hours=duration_hours,
        )
        event_thresholds[model_name] = threshold
        metrics = evaluate_probabilities(
            validation_truth,
            probability,
            threshold=threshold,
            duration_hours=duration_hours,
        )
        metric_rows.append(
            {
                "head": "event",
                "model_name": model_name,
                "threshold": threshold,
                **{
                    key: value
                    for key, value in metrics.items()
                    if isinstance(value, int | float)
                },
            }
        )
        event_scores.append(
            (
                float(cast(float | int, metrics["event_f1"])),
                float(cast(float | int, metrics["aucpr"])),
                -float(cast(float | int, metrics["false_alerts_per_hour"])),
                model_name,
                threshold,
            )
        )
    _, _, _, selected_event_model, event_threshold = max(event_scores)

    train_stage = train.loc[
        train["stage_code"].astype(str) != "NO_EVENT"
    ].copy()
    validation_stage = validation.loc[
        validation["stage_code"].astype(str) != "NO_EVENT"
    ].copy()
    stage_to_index = {stage: index for index, stage in enumerate(MODEL_STAGE_CODES)}
    unknown_train = set(train_stage["stage_code"].astype(str)) - set(stage_to_index)
    unknown_validation = set(validation_stage["stage_code"].astype(str)) - set(
        stage_to_index
    )
    if unknown_train or unknown_validation:
        raise ValueError(
            f"unsupported stage codes: {sorted(unknown_train | unknown_validation)}"
        )
    stage_models = fit_candidate_models(
        train_stage[list(feature_names)].to_numpy(dtype=np.float32),
        np.asarray(
            train_stage["stage_code"].astype(str).map(stage_to_index),
            dtype=np.int8,
        ),
        random_state=random_state,
    )
    validation_stage_truth = (
        validation_stage["stage_code"]
        .astype(str)
        .map(stage_to_index)
        .to_numpy(dtype=np.int8)
    )
    stage_scores: list[tuple[float, str]] = []
    for model_name, model in stage_models.items():
        probability = model.predict_proba(
            validation_stage[list(feature_names)].to_numpy(dtype=np.float32)
        )
        predicted = np.argmax(probability, axis=1).astype(np.int8)
        macro_f1 = float(
            f1_score(
                validation_stage_truth,
                predicted,
                labels=list(range(len(MODEL_STAGE_CODES))),
                average="macro",
                zero_division=0,
            )
        )
        metric_rows.append(
            {
                "head": "stage",
                "model_name": model_name,
                "threshold": np.nan,
                "macro_f1": macro_f1,
            }
        )
        stage_scores.append((macro_f1, model_name))
    _, selected_stage_model = max(stage_scores)
    return StageCandidateResult(
        event_models=event_models,
        stage_models=stage_models,
        selected_event_model=selected_event_model,
        selected_stage_model=selected_stage_model,
        event_threshold=event_threshold,
        metrics=pd.DataFrame(metric_rows),
    )


def fit_behavior_candidates(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    behavior_codes: Sequence[str],
    random_state: int,
) -> BehaviorCandidateResult:
    """Fit per-behavior one-vs-rest candidates with fixed support gates."""

    _require_columns(train, {*feature_names, *behavior_codes})
    _require_columns(validation, {*feature_names, *behavior_codes})
    models: dict[tuple[str, str], ProbabilityClassifier] = {}
    selected_model_by_behavior: dict[str, str] = {}
    status_by_behavior: dict[str, BehaviorSupport] = {}
    metric_rows: list[dict[str, object]] = []
    train_features = train[list(feature_names)].to_numpy(dtype=np.float32)
    validation_features = validation[list(feature_names)].to_numpy(dtype=np.float32)
    for behavior_offset, behavior_code in enumerate(behavior_codes):
        train_truth = train[behavior_code].to_numpy(dtype=np.int8)
        validation_truth = validation[behavior_code].to_numpy(dtype=np.int8)
        status = behavior_support_status(
            train_positive=int(train_truth.sum()),
            validation_positive=int(validation_truth.sum()),
        )
        if status == "SUPPORTED" and (
            len(np.unique(train_truth)) < 2 or len(np.unique(validation_truth)) < 2
        ):
            status = "INSUFFICIENT_LABEL_SUPPORT"
        status_by_behavior[behavior_code] = status
        if status != "SUPPORTED":
            metric_rows.append(
                {
                    "head": "behavior",
                    "behavior_code": behavior_code,
                    "model_name": None,
                    "status": status,
                    "train_positive": int(train_truth.sum()),
                    "validation_positive": int(validation_truth.sum()),
                }
            )
            continue
        candidates = fit_candidate_models(
            train_features,
            train_truth,
            random_state=random_state + behavior_offset,
        )
        scores: list[tuple[float, float, str]] = []
        for model_name, model in candidates.items():
            models[(behavior_code, model_name)] = model
            probability = model.predict_proba(validation_features)[:, 1].astype(
                np.float64
            )
            aucpr = float(average_precision_score(validation_truth, probability))
            predicted = probability >= 0.5
            macro_f1 = float(
                f1_score(
                    validation_truth,
                    predicted,
                    average="binary",
                    zero_division=0,
                )
            )
            metric_rows.append(
                {
                    "head": "behavior",
                    "behavior_code": behavior_code,
                    "model_name": model_name,
                    "status": status,
                    "train_positive": int(train_truth.sum()),
                    "validation_positive": int(validation_truth.sum()),
                    "aucpr": aucpr,
                    "f1": macro_f1,
                }
            )
            scores.append((aucpr, macro_f1, model_name))
        _, _, selected_model = max(scores)
        selected_model_by_behavior[behavior_code] = selected_model
    return BehaviorCandidateResult(
        models=models,
        selected_model_by_behavior=selected_model_by_behavior,
        status_by_behavior=status_by_behavior,
        metrics=pd.DataFrame(metric_rows),
    )


def _require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
