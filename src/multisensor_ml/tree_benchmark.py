"""XGBoost/LightGBM challenger benchmark with feature-importance stability.

The module keeps optional tree libraries out of the core inference path.  The
``tree-benchmark`` project extra is required only for the challenger run.  A
candidate is selected on the same person split and validation contract as the
existing Logistic/HGB models; locked test data is intentionally not touched.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol, cast

import numpy as np
import pandas as pd


class TreeProbabilityModel(Protocol):
    feature_importances_: Any

    def fit(self, features: object, target: object) -> object: ...

    def predict_proba(self, features: object) -> Any: ...


TREE_MODEL_IDS: Final[tuple[str, str]] = ("xgboost", "lightgbm")
TREE_MODEL_LABELS_KO: Final[dict[str, str]] = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "soft_ensemble": "소프트 앙상블",
}


@dataclass(frozen=True, slots=True)
class TreeBenchmarkConfig:
    """Deterministic, CPU-safe challenger defaults."""

    random_state: int = 20260725
    seed_count: int = 3
    n_estimators: int = 160
    learning_rate: float = 0.05
    max_depth: int = 6
    num_leaves: int = 31
    min_child_samples: int = 50
    subsample: float = 0.9
    colsample_bytree: float = 0.8
    n_jobs: int = 1
    threshold: float = 0.5


DEFAULT_TREE_BENCHMARK_CONFIG: Final[TreeBenchmarkConfig] = TreeBenchmarkConfig()


@dataclass(frozen=True, slots=True)
class TreeBenchmarkResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    importance_summary: pd.DataFrame
    anchor_model: str


def feature_importance_table(
    *,
    model_id: str,
    seed: int,
    feature_names: Sequence[str],
    importances: Sequence[float],
) -> pd.DataFrame:
    """Normalize one model/seed's gain or split importance vector."""

    names = [str(value) for value in feature_names]
    values = np.asarray(importances, dtype="float64")
    if len(names) != len(values):
        raise ValueError("feature_names and importances must have the same length")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    normalized = (
        np.full(len(values), 1.0 / len(values), dtype="float64")
        if total <= 0.0 and len(values)
        else values / total
    )
    order = np.argsort(-normalized, kind="mergesort")
    ranks = np.empty(len(values), dtype="int64")
    ranks[order] = np.arange(1, len(values) + 1, dtype="int64")
    return pd.DataFrame(
        {
            "model_id": str(model_id),
            "seed": int(seed),
            "feature_name": names,
            "importance": values,
            "normalized_importance": normalized,
            "importance_rank": ranks,
        }
    )


def summarize_feature_importance(table: pd.DataFrame) -> pd.DataFrame:
    """Report spread and seed-to-seed variance for each challenger."""

    required = {"model_id", "feature_name", "normalized_importance"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"feature importance table missing columns: {missing}")
    rows: list[dict[str, object]] = []
    for model_id, model_table in table.groupby("model_id", sort=True):
        by_feature = model_table.groupby("feature_name", sort=True)[
            "normalized_importance"
        ]
        mean_importance = by_feature.mean().sort_values(ascending=False, kind="mergesort")
        variance = by_feature.var(ddof=0).fillna(0.0)
        probabilities = mean_importance.to_numpy(dtype="float64")
        entropy_denominator = math.log(max(len(probabilities), 2))
        entropy = float(
            -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, None)))
            / entropy_denominator
        )
        rows.append(
            {
                "model_id": str(model_id),
                "top_feature": str(mean_importance.index[0]),
                "top3_share": float(mean_importance.head(3).sum()),
                "importance_spread_entropy": entropy,
                "importance_variance_mean": float(variance.mean()),
                "importance_variance_max": float(variance.max()),
                "seed_count": int(model_table["seed"].nunique())
                if "seed" in model_table
                else 1,
            }
        )
    if not rows:
        raise ValueError("feature importance table is empty")
    return pd.DataFrame(rows)


def weighted_soft_average(
    probabilities: Mapping[str, Sequence[float] | np.ndarray],
    *,
    weights: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Blend probability vectors without changing the validation contract."""

    if not probabilities:
        raise ValueError("at least one probability vector is required")
    keys = tuple(probabilities)
    vectors = [np.asarray(probabilities[key], dtype="float64") for key in keys]
    length = len(vectors[0])
    if any(vector.ndim != 1 or len(vector) != length for vector in vectors):
        raise ValueError("probability vectors must be one-dimensional and aligned")
    raw_weights = np.asarray(
        [1.0 if weights is None else float(weights.get(key, 0.0)) for key in keys],
        dtype="float64",
    )
    if np.any(raw_weights < 0.0) or not np.isfinite(raw_weights).all():
        raise ValueError("ensemble weights must be finite and non-negative")
    denominator = float(raw_weights.sum())
    if denominator <= 0.0:
        raise ValueError("at least one ensemble weight must be positive")
    result = sum(vector * weight for vector, weight in zip(vectors, raw_weights, strict=True))
    return np.asarray(np.clip(result / denominator, 0.0, 1.0), dtype="float64")


def select_anchor_model(
    metrics: pd.DataFrame,
    stability: pd.DataFrame,
    *,
    aucpr_guardrail: float = 0.02,
) -> str:
    """Choose a stable challenger among models near the best AUCPR.

    Stability is considered only after a candidate is within the declared
    AUCPR guardrail.  This prevents a tiny importance-spread advantage from
    overriding a materially better detector.
    """

    if not 0.0 <= aucpr_guardrail < 1.0:
        raise ValueError("aucpr_guardrail must be in [0, 1)")
    required_metrics = {
        "model_id",
        "aucpr",
        "event_f1",
        "false_alerts_per_hour",
        "calibration_error",
    }
    required_stability = {"model_id", "importance_variance_mean", "top3_share"}
    missing_metrics = sorted(required_metrics.difference(metrics.columns))
    missing_stability = sorted(required_stability.difference(stability.columns))
    if missing_metrics or missing_stability:
        raise ValueError(
            f"anchor inputs missing metrics={missing_metrics}, stability={missing_stability}"
        )
    merged = metrics.merge(stability, on="model_id", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("no common model candidates for anchor selection")
    finite_aucpr = merged.loc[np.isfinite(merged["aucpr"].astype("float64"))].copy()
    if finite_aucpr.empty:
        # Tiny smoke samples can contain no positive row.  Preserve a
        # deterministic stability fallback instead of indexing an empty frame.
        eligible = merged.copy()
    else:
        best_aucpr = float(finite_aucpr["aucpr"].max())
        eligible = finite_aucpr.loc[
            finite_aucpr["aucpr"].ge(best_aucpr * (1.0 - aucpr_guardrail))
        ].copy()
    eligible = eligible.sort_values(
        [
            "importance_variance_mean",
            "top3_share",
            "false_alerts_per_hour",
            "calibration_error",
            "event_f1",
            "aucpr",
            "model_id",
        ],
        ascending=[True, True, True, True, False, False, True],
        kind="mergesort",
    )
    return str(eligible.iloc[0]["model_id"])


def build_tree_candidates(
    config: TreeBenchmarkConfig,
    *,
    seed: int,
) -> dict[str, TreeProbabilityModel]:
    """Build XGBoost and LightGBM lazily, so core inference stays dependency-light."""

    try:
        from lightgbm import LGBMClassifier
        from xgboost import XGBClassifier
    except ImportError as error:  # pragma: no cover - exercised in minimal installs
        raise RuntimeError(
            "tree challengers require the optional dependency group: "
            "uv sync --extra tree-benchmark"
        ) from error
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        min_child_weight=5,
        n_jobs=config.n_jobs,
        random_state=seed,
        verbosity=0,
    )
    lgbm = LGBMClassifier(
        objective="binary",
        n_estimators=config.n_estimators,
        num_leaves=config.num_leaves,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        min_child_samples=config.min_child_samples,
        n_jobs=config.n_jobs,
        random_state=seed,
        verbosity=-1,
    )
    return {
        "xgboost": cast(TreeProbabilityModel, xgb),
        "lightgbm": cast(TreeProbabilityModel, lgbm),
    }


def expected_calibration_error(
    target: Sequence[int] | np.ndarray,
    probability: Sequence[float] | np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Compute a fixed-bin ECE for comparable candidate reports."""

    if bins < 2:
        raise ValueError("bins must be at least 2")
    y = np.asarray(target, dtype="float64")
    p = np.asarray(probability, dtype="float64")
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p):
        raise ValueError("target and probability must be aligned one-dimensional arrays")
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (p >= lower) & (p <= upper if index == bins - 1 else p < upper)
        if mask.any():
            result += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(result)


def _false_alerts_per_hour(
    target: np.ndarray,
    probability: np.ndarray,
    *,
    threshold: float,
    timestamps: Sequence[object] | None,
    groups: Sequence[object] | None,
) -> float:
    if timestamps is None or groups is None:
        return float("nan")
    if len(timestamps) != len(target) or len(groups) != len(target):
        raise ValueError("timestamps and groups must align with target")
    timestamp_values = pd.Series(list(timestamps), dtype="object")
    work = pd.DataFrame(
        {
            "group": list(groups),
            "timestamp": pd.to_datetime(timestamp_values, utc=True, errors="coerce"),
        }
    )
    duration = work.groupby("group", sort=False)["timestamp"].agg(
        lambda values: (values.max() - values.min()).total_seconds()
    )
    hours = float(duration.clip(lower=1.0).sum() / 3600.0)
    false_alerts = int(((target == 0) & (probability >= threshold)).sum())
    return float(false_alerts / hours) if hours > 0.0 else float("nan")


def evaluate_tree_predictions(
    target: Sequence[int] | np.ndarray,
    probability: Sequence[float] | np.ndarray,
    *,
    threshold: float,
    timestamps: Sequence[object] | None = None,
    groups: Sequence[object] | None = None,
) -> dict[str, float]:
    """Return threshold-free ranking and thresholded/calibration metrics."""

    from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, recall_score

    y = np.asarray(target, dtype="int8")
    p = np.asarray(probability, dtype="float64")
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p):
        raise ValueError("target and probability must be aligned one-dimensional arrays")
    predicted = p >= float(threshold)
    aucpr = float(average_precision_score(y, p)) if np.unique(y).size > 1 else float("nan")
    return {
        "aucpr": aucpr,
        "event_recall": float(recall_score(y, predicted, zero_division=0)),
        "event_f1": float(f1_score(y, predicted, zero_division=0)),
        "false_alerts_per_hour": _false_alerts_per_hour(
            y,
            p,
            threshold=threshold,
            timestamps=timestamps,
            groups=groups,
        ),
        "brier_score": float(brier_score_loss(y, p)),
        "calibration_error": expected_calibration_error(y, p),
        "threshold": float(threshold),
        "row_count": float(len(y)),
        "positive_count": float(y.sum()),
    }


def reference_metrics_from_validation(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize existing event-head validation rows for side-by-side reports."""

    required = {
        "head",
        "model_name",
        "role",
        "threshold",
        "aucpr",
        "event_recall",
        "event_f1",
        "false_alerts_per_hour",
        "brier_score",
        "calibration_error",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"reference validation metrics missing columns: {missing}")
    selected = frame.loc[
        frame["head"].eq("event") & frame["role"].eq("validation"),
        [
            "model_name",
            "threshold",
            "aucpr",
            "event_recall",
            "event_f1",
            "false_alerts_per_hour",
            "brier_score",
            "calibration_error",
        ],
    ].copy()
    if selected.empty:
        raise ValueError("existing validation metrics contain no event rows")
    selected["model_id"] = "existing_" + selected.pop("model_name").astype(str)
    selected["model_label_ko"] = selected["model_id"].map(
        {
            "existing_hist_gradient_boosting": "기존 HGB",
            "existing_logistic_regression": "기존 Logistic",
        }
    ).fillna(selected["model_id"])
    selected["evaluation_scope"] = "existing_full_validation"
    selected["data_status"] = "oracle/sanity"
    selected["real_data_status"] = "NOT VERIFIED"
    selected["locked_test_read"] = False
    return selected.reset_index(drop=True)


def fit_tree_challengers(
    train_features: pd.DataFrame,
    train_target: Sequence[int] | np.ndarray,
    validation_features: pd.DataFrame,
    validation_target: Sequence[int] | np.ndarray,
    *,
    config: TreeBenchmarkConfig = DEFAULT_TREE_BENCHMARK_CONFIG,
    validation_timestamps: Sequence[object] | None = None,
    validation_groups: Sequence[object] | None = None,
) -> TreeBenchmarkResult:
    """Fit two challengers across deterministic seeds and add a soft ensemble."""

    if config.seed_count < 1:
        raise ValueError("seed_count must be positive")
    if list(train_features.columns) != list(validation_features.columns):
        raise ValueError("train and validation feature schemas must match exactly")
    feature_names = list(train_features.columns)
    y_train = np.asarray(train_target, dtype="int8")
    y_validation = np.asarray(validation_target, dtype="int8")
    if len(y_train) != len(train_features) or len(y_validation) != len(validation_features):
        raise ValueError("features and targets must align")
    probability_by_model: dict[str, list[np.ndarray]] = {
        model_id: [] for model_id in TREE_MODEL_IDS
    }
    importance_parts: list[pd.DataFrame] = []
    for seed_offset in range(config.seed_count):
        seed = config.random_state + seed_offset
        candidates = build_tree_candidates(config, seed=seed)
        for model_id, model in candidates.items():
            model.fit(train_features, y_train)
            probabilities = np.asarray(
                model.predict_proba(validation_features), dtype="float64"
            )[:, 1]
            probability_by_model[model_id].append(probabilities)
            importance_parts.append(
                feature_importance_table(
                    model_id=model_id,
                    seed=seed,
                    feature_names=feature_names,
                    importances=cast(
                        Sequence[float],
                        np.asarray(model.feature_importances_, dtype="float64"),
                    ),
                )
            )
    importance = pd.concat(importance_parts, ignore_index=True)
    summary = summarize_feature_importance(importance)
    predictions_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    mean_probabilities: dict[str, np.ndarray] = {}
    for model_id, values in probability_by_model.items():
        probabilities = np.mean(np.stack(values, axis=0), axis=0)
        mean_probabilities[model_id] = probabilities
        row: dict[str, object] = {
            "model_id": model_id,
            "model_label_ko": TREE_MODEL_LABELS_KO[model_id],
        }
        row.update(
            evaluate_tree_predictions(
                y_validation,
                probabilities,
                threshold=config.threshold,
                timestamps=validation_timestamps,
                groups=validation_groups,
            )
        )
        metric_rows.append(row)
        predictions_parts.append(
            pd.DataFrame(
                {
                    "model_id": model_id,
                    "probability": probabilities,
                    "event_binary": y_validation,
                }
            )
        )
    ensemble = weighted_soft_average(mean_probabilities)
    ensemble_row: dict[str, object] = {
        "model_id": "soft_ensemble",
        "model_label_ko": TREE_MODEL_LABELS_KO["soft_ensemble"],
    }
    ensemble_row.update(
        evaluate_tree_predictions(
            y_validation,
            ensemble,
            threshold=config.threshold,
            timestamps=validation_timestamps,
            groups=validation_groups,
        )
    )
    metric_rows.append(ensemble_row)
    predictions_parts.append(
        pd.DataFrame(
            {
                "model_id": "soft_ensemble",
                "probability": ensemble,
                "event_binary": y_validation,
            }
        )
    )
    metrics = pd.DataFrame(metric_rows)
    anchor_model = select_anchor_model(
        metrics.loc[metrics["model_id"].isin(TREE_MODEL_IDS)],
        summary,
    )
    metrics["anchor_model"] = anchor_model
    metrics["data_status"] = "oracle/sanity"
    metrics["real_data_status"] = "NOT VERIFIED"
    metrics["locked_test_read"] = False
    predictions = pd.concat(predictions_parts, ignore_index=True)
    return TreeBenchmarkResult(
        metrics=metrics,
        predictions=predictions,
        feature_importance=importance,
        importance_summary=summary,
        anchor_model=anchor_model,
    )
