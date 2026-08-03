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


GPU_TREE_MODEL_IDS: Final[tuple[str, str]] = ("xgboost", "lightgbm")
TREE_MODEL_IDS: Final[tuple[str, str, str]] = (
    "xgboost",
    "lightgbm",
    "extra_trees",
)
TREE_MODEL_LABELS_KO: Final[dict[str, str]] = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "extra_trees": "ExtraTrees",
    "soft_ensemble": "소프트 앙상블",
}

DERIVED_SIGNAL_BASES: Final[tuple[str, ...]] = (
    "autonomic_arousal",
    "motor_activation",
    "cognitive_load",
    "sleep_pressure",
    "sensory_context",
    "recovery_capacity",
    "social_context",
)
FEATURE_GROUP_LABELS_KO: Final[dict[str, str]] = {
    "derived_signal": "생리·상황 파생변수",
    "time": "시간 주기",
    "context": "맥락·각성",
    "other": "기타·품질",
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
    include_extra_trees: bool = False
    extra_trees_n_estimators: int = 64
    use_gpu: bool = False
    gpu_required: bool = False
    gpu_device: str = "cuda"
    time_ablation_aucpr_relative_drop_limit: float = 0.05
    time_ablation_event_recall_absolute_drop_limit: float = 0.05
    min_positive_person_fraction: float = 0.80
    min_repeat_positive_rate: float = 0.67
    permutation_repeats: int = 3

    def validate(self) -> None:
        """Reject an ambiguous GPU request before fitting any candidate."""

        if self.seed_count < 1:
            raise ValueError("seed_count must be positive")
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be positive")
        if self.extra_trees_n_estimators < 1:
            raise ValueError("extra_trees_n_estimators must be positive")
        if self.gpu_required and not self.use_gpu:
            raise ValueError("gpu_required cannot be enabled when use_gpu is false")
        if self.use_gpu and not self.gpu_device.strip():
            raise ValueError("gpu_device must be non-empty when use_gpu is enabled")


DEFAULT_TREE_BENCHMARK_CONFIG: Final[TreeBenchmarkConfig] = TreeBenchmarkConfig()


@dataclass(frozen=True, slots=True)
class TreeBenchmarkResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    importance_summary: pd.DataFrame
    anchor_model: str
    fitted_models: dict[str, tuple[TreeProbabilityModel, ...]] | None = None


def feature_group_map(feature_names: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """Assign model inputs to auditable semantic groups.

    All seven latent-derived factor families are grouped together so the gate
    tests the intended derived-signal layer rather than a single correlated
    window.  Calendar features are kept separate to expose schedule shortcuts.
    """

    groups: dict[str, list[str]] = {
        "derived_signal": [],
        "time": [],
        "context": [],
        "other": [],
    }
    for feature in feature_names:
        name = str(feature)
        base = name.split("__", 1)[0]
        if base in DERIVED_SIGNAL_BASES:
            group_id = "derived_signal"
        elif name.startswith("time_") or name.startswith("weekday_"):
            group_id = "time"
        elif name.startswith("context__") or name == "is_awake":
            group_id = "context"
        else:
            group_id = "other"
        groups[group_id].append(name)
    return {group_id: tuple(values) for group_id, values in groups.items() if values}


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


PERMUTATION_COLUMNS: Final[tuple[str, ...]] = (
    "model_id",
    "seed_index",
    "person_key",
    "group_id",
    "repeat",
    "baseline_aucpr",
    "permuted_aucpr",
    "importance_drop",
    "valid_person",
)


def permutation_importance_by_person(
    models: Mapping[str, Sequence[TreeProbabilityModel]],
    validation_features: pd.DataFrame,
    validation_target: Sequence[int] | np.ndarray,
    validation_groups: Sequence[object],
    feature_groups: Mapping[str, Sequence[str]],
    *,
    repeats: int = 3,
    random_state: int = 20260725,
) -> pd.DataFrame:
    """Measure grouped permutation AUCPR drop within each validation person.

    Permutations stay inside a person so the audit does not turn a person
    split into a cross-person distribution shift.  Each group is permuted as
    a block, preserving correlations among its windows.  A positive drop
    means the model relied on that group for that person's ranking.
    """

    from sklearn.metrics import average_precision_score

    if repeats < 1:
        raise ValueError("repeats must be positive")
    target = np.asarray(validation_target, dtype="int8")
    if len(target) != len(validation_features) or len(validation_groups) != len(target):
        raise ValueError("validation features, target, and groups must align")
    feature_names = list(validation_features.columns)
    positions_by_person: dict[object, list[int]] = {}
    for position, person in enumerate(validation_groups):
        positions_by_person.setdefault(person, []).append(position)
    position_arrays = {
        person: np.asarray(positions, dtype="int64")
        for person, positions in positions_by_person.items()
    }
    rows: list[dict[str, object]] = []
    for model_id, seed_models in models.items():
        for seed_index, model in enumerate(seed_models):
            for person_key, positions in position_arrays.items():
                person_target = target[positions]
                valid_person = np.unique(person_target).size > 1
                if not valid_person:
                    continue
                person_features = validation_features.iloc[positions]
                baseline_probability = np.asarray(
                    model.predict_proba(person_features), dtype="float64"
                )[:, 1]
                baseline_aucpr = float(
                    average_precision_score(person_target, baseline_probability)
                )
                for group_index, (group_id, columns) in enumerate(feature_groups.items()):
                    selected_columns = [name for name in columns if name in feature_names]
                    if not selected_columns:
                        continue
                    column_positions = [feature_names.index(name) for name in selected_columns]
                    original = person_features.to_numpy(dtype="float64", copy=True)
                    for repeat in range(repeats):
                        permutation_seed = (
                            random_state
                            + seed_index * 100_000
                            + group_index * 1_000
                            + repeat
                        )
                        permutation = np.random.default_rng(permutation_seed).permutation(
                            len(positions)
                        )
                        permuted = original.copy()
                        permuted[:, column_positions] = original[
                            permutation[:, None], column_positions
                        ]
                        permuted_probability = np.asarray(
                            model.predict_proba(
                                pd.DataFrame(permuted, columns=feature_names)
                            ),
                            dtype="float64",
                        )[:, 1]
                        permuted_aucpr = float(
                            average_precision_score(person_target, permuted_probability)
                        )
                        rows.append(
                            {
                                "model_id": str(model_id),
                                "seed_index": int(seed_index),
                                "person_key": str(person_key),
                                "group_id": str(group_id),
                                "repeat": int(repeat),
                                "baseline_aucpr": baseline_aucpr,
                                "permuted_aucpr": permuted_aucpr,
                                "importance_drop": baseline_aucpr - permuted_aucpr,
                                "valid_person": True,
                            }
                        )
    return pd.DataFrame(rows, columns=PERMUTATION_COLUMNS)


def summarize_permutation_importance(
    detail: pd.DataFrame,
    *,
    min_positive_person_fraction: float = 0.80,
    min_repeat_positive_rate: float = 0.67,
) -> pd.DataFrame:
    """Summarize repeated positive group contribution by model and group."""

    if not 0.0 <= min_positive_person_fraction <= 1.0:
        raise ValueError("min_positive_person_fraction must be in [0, 1]")
    if not 0.0 <= min_repeat_positive_rate <= 1.0:
        raise ValueError("min_repeat_positive_rate must be in [0, 1]")
    required = {
        "model_id",
        "group_id",
        "person_key",
        "importance_drop",
        "valid_person",
    }
    missing = sorted(required.difference(detail.columns))
    if missing:
        raise ValueError(f"permutation detail missing columns: {missing}")
    valid = detail.loc[detail["valid_person"].astype(bool)].copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "model_id",
                "group_id",
                "mean_importance_drop",
                "median_importance_drop",
                "positive_person_fraction",
                "mean_person_repeat_positive_rate",
                "valid_person_count",
                "repeated_positive",
            ]
        )
    person = (
        valid.groupby(["model_id", "group_id", "person_key"], sort=True)["importance_drop"]
        .agg(
            person_mean_importance_drop="mean",
            person_median_importance_drop="median",
            person_repeat_positive_rate=lambda values: float((values > 0.0).mean()),
        )
        .reset_index()
    )
    person["person_repeated_positive"] = person["person_repeat_positive_rate"].ge(
        min_repeat_positive_rate
    )
    summary = (
        person.groupby(["model_id", "group_id"], sort=True)
        .agg(
            mean_importance_drop=("person_mean_importance_drop", "mean"),
            median_importance_drop=("person_median_importance_drop", "median"),
            positive_person_fraction=("person_repeated_positive", "mean"),
            mean_person_repeat_positive_rate=("person_repeat_positive_rate", "mean"),
            valid_person_count=("person_key", "nunique"),
        )
        .reset_index()
    )
    summary["repeated_positive"] = (
        summary["mean_importance_drop"].gt(0.0)
        & summary["positive_person_fraction"].ge(min_positive_person_fraction)
        & summary["mean_person_repeat_positive_rate"].ge(min_repeat_positive_rate)
    )
    return summary


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


def evaluate_anchor_gate(
    full_metrics: pd.DataFrame,
    time_ablation_metrics: pd.DataFrame,
    permutation_summary: pd.DataFrame,
    *,
    aucpr_relative_drop_limit: float = 0.05,
    event_recall_absolute_drop_limit: float = 0.05,
    derived_group_id: str = "derived_signal",
    min_positive_person_fraction: float = 0.80,
    min_repeat_positive_rate: float = 0.67,
) -> pd.DataFrame:
    """Apply the predeclared anchor requirements to each tree challenger."""

    if not 0.0 <= aucpr_relative_drop_limit < 1.0:
        raise ValueError("aucpr_relative_drop_limit must be in [0, 1)")
    if event_recall_absolute_drop_limit < 0.0:
        raise ValueError("event_recall_absolute_drop_limit must be non-negative")
    required_metrics = {"model_id", "aucpr", "event_recall"}
    missing_full = sorted(required_metrics.difference(full_metrics.columns))
    missing_ablation = sorted(required_metrics.difference(time_ablation_metrics.columns))
    required_permutation = {
        "model_id",
        "group_id",
        "mean_importance_drop",
        "positive_person_fraction",
        "mean_person_repeat_positive_rate",
    }
    missing_permutation = sorted(required_permutation.difference(permutation_summary.columns))
    if missing_full or missing_ablation or missing_permutation:
        raise ValueError(
            "anchor gate inputs missing "
            f"full={missing_full}, ablation={missing_ablation}, "
            f"permutation={missing_permutation}"
        )
    candidate_ids = sorted(
        set(full_metrics["model_id"].astype(str))
        .intersection(time_ablation_metrics["model_id"].astype(str))
    )
    rows: list[dict[str, object]] = []
    for model_id in candidate_ids:
        full_match = full_metrics.loc[full_metrics["model_id"].eq(model_id)]
        ablation_match = time_ablation_metrics.loc[
            time_ablation_metrics["model_id"].eq(model_id)
        ]
        permutation_match = permutation_summary.loc[
            permutation_summary["model_id"].eq(model_id)
            & permutation_summary["group_id"].eq(derived_group_id)
        ]
        reasons: list[str] = []
        if full_match.empty or ablation_match.empty:
            rows.append(
                {
                    "model_id": model_id,
                    "gate_pass": False,
                    "gate_reason": "missing_full_or_time_ablation_metrics",
                }
            )
            continue
        full_row = full_match.iloc[0]
        ablation_row = ablation_match.iloc[0]
        full_aucpr = float(full_row["aucpr"])
        ablation_aucpr = float(ablation_row["aucpr"])
        full_recall = float(full_row["event_recall"])
        ablation_recall = float(ablation_row["event_recall"])
        if np.isfinite(full_aucpr) and np.isfinite(ablation_aucpr):
            aucpr_relative_drop = max(
                0.0,
                (full_aucpr - ablation_aucpr) / max(abs(full_aucpr), 1e-12),
            )
            aucpr_pass = aucpr_relative_drop <= aucpr_relative_drop_limit
        else:
            aucpr_relative_drop = float("nan")
            aucpr_pass = False
        if np.isfinite(full_recall) and np.isfinite(ablation_recall):
            event_recall_drop = max(0.0, full_recall - ablation_recall)
            event_recall_pass = event_recall_drop <= event_recall_absolute_drop_limit
        else:
            event_recall_drop = float("nan")
            event_recall_pass = False
        if permutation_match.empty:
            mean_importance_drop = float("nan")
            positive_person_fraction = 0.0
            mean_repeat_positive_rate = 0.0
            derived_pass = False
        else:
            permutation_row = permutation_match.iloc[0]
            mean_importance_drop = float(permutation_row["mean_importance_drop"])
            positive_person_fraction = float(permutation_row["positive_person_fraction"])
            mean_repeat_positive_rate = float(
                permutation_row["mean_person_repeat_positive_rate"]
            )
            derived_pass = (
                mean_importance_drop > 0.0
                and positive_person_fraction >= min_positive_person_fraction
                and mean_repeat_positive_rate >= min_repeat_positive_rate
            )
        if not aucpr_pass:
            reasons.append("time_ablation_aucpr_drop_exceeded")
        if not event_recall_pass:
            reasons.append("time_ablation_event_recall_drop_exceeded")
        if not derived_pass:
            reasons.append("derived_group_permutation_not_repeated_positive")
        rows.append(
            {
                "model_id": model_id,
                "full_aucpr": full_aucpr,
                "time_ablation_aucpr": ablation_aucpr,
                "aucpr_relative_drop": aucpr_relative_drop,
                "full_event_recall": full_recall,
                "time_ablation_event_recall": ablation_recall,
                "event_recall_drop": event_recall_drop,
                "time_ablation_aucpr_pass": bool(aucpr_pass),
                "time_ablation_event_recall_pass": bool(event_recall_pass),
                "derived_mean_importance_drop": mean_importance_drop,
                "derived_positive_person_fraction": positive_person_fraction,
                "derived_mean_repeat_positive_rate": mean_repeat_positive_rate,
                "derived_permutation_pass": bool(derived_pass),
                "gate_pass": bool(aucpr_pass and event_recall_pass and derived_pass),
                "gate_reason": "PASS" if not reasons else ";".join(reasons),
            }
        )
    return pd.DataFrame(rows)


def select_gated_anchor_model(
    metrics: pd.DataFrame,
    stability: pd.DataFrame,
    gate: pd.DataFrame,
    *,
    aucpr_guardrail: float = 0.02,
) -> str | None:
    """Select a stable candidate only after the full audit gate passes."""

    required = {"model_id", "gate_pass"}
    missing = sorted(required.difference(gate.columns))
    if missing:
        raise ValueError(f"anchor gate missing columns: {missing}")
    approved = gate.loc[gate["gate_pass"].astype(bool), "model_id"].astype(str)
    eligible = metrics.loc[metrics["model_id"].astype(str).isin(set(approved))].copy()
    if eligible.empty:
        return None
    return select_anchor_model(eligible, stability, aucpr_guardrail=aucpr_guardrail)


def build_tree_candidates(
    config: TreeBenchmarkConfig,
    *,
    seed: int,
) -> dict[str, TreeProbabilityModel]:
    """Build GPU-capable boosters and an optional CPU ExtraTrees candidate."""

    config.validate()

    try:
        from lightgbm import LGBMClassifier
        from xgboost import XGBClassifier
    except ImportError as error:  # pragma: no cover - exercised in minimal installs
        raise RuntimeError(
            "tree challengers require the optional dependency group: "
            "uv sync --extra tree-benchmark"
        ) from error
    xgb_parameters: dict[str, object] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "n_estimators": config.n_estimators,
        "max_depth": config.max_depth,
        "learning_rate": config.learning_rate,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "min_child_weight": 5,
        "n_jobs": config.n_jobs,
        "random_state": seed,
        "verbosity": 0,
    }
    if config.use_gpu:
        # XGBoost 2+ uses ``device`` with hist; a successful fit is the GPU
        # execution proof.  We deliberately do not fall back to CPU here.
        xgb_parameters["device"] = config.gpu_device
    xgb = XGBClassifier(**xgb_parameters)
    lgbm_parameters: dict[str, Any] = {
        "objective": "binary",
        "n_estimators": config.n_estimators,
        "num_leaves": config.num_leaves,
        "learning_rate": config.learning_rate,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "min_child_samples": config.min_child_samples,
        "n_jobs": config.n_jobs,
        "random_state": seed,
        "verbosity": -1,
    }
    if config.use_gpu:
        # LightGBM requires a GPU-enabled build.  If the wheel lacks it, the
        # required GPU run fails loudly instead of silently using CPU.
        lgbm_parameters["device_type"] = "gpu"
    lgbm = LGBMClassifier(**lgbm_parameters)
    candidates: dict[str, TreeProbabilityModel] = {
        "xgboost": cast(TreeProbabilityModel, xgb),
        "lightgbm": cast(TreeProbabilityModel, lgbm),
    }
    if config.include_extra_trees:
        try:
            from sklearn.ensemble import ExtraTreesClassifier
        except ImportError as error:  # pragma: no cover - core dependency guard
            raise RuntimeError("ExtraTrees requires scikit-learn") from error
        candidates["extra_trees"] = cast(
            TreeProbabilityModel,
            ExtraTreesClassifier(
                n_estimators=config.extra_trees_n_estimators,
                max_depth=config.max_depth,
                max_features=config.colsample_bytree,
                class_weight="balanced",
                n_jobs=config.n_jobs,
                random_state=seed,
            ),
        )
    return candidates


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
    """Fit the configured challengers across seeds and add a soft ensemble.

    ``use_gpu`` applies only to XGBoost and LightGBM.  ExtraTrees is kept as a
    deliberately labelled CPU reference because scikit-learn ExtraTrees has
    no CUDA implementation.
    """

    config.validate()
    if list(train_features.columns) != list(validation_features.columns):
        raise ValueError("train and validation feature schemas must match exactly")
    feature_names = list(train_features.columns)
    y_train = np.asarray(train_target, dtype="int8")
    y_validation = np.asarray(validation_target, dtype="int8")
    if len(y_train) != len(train_features) or len(y_validation) != len(validation_features):
        raise ValueError("features and targets must align")
    probability_by_model: dict[str, list[np.ndarray]] = {}
    fitted_models: dict[str, list[TreeProbabilityModel]] = {}
    importance_parts: list[pd.DataFrame] = []
    for seed_offset in range(config.seed_count):
        seed = config.random_state + seed_offset
        candidates = build_tree_candidates(config, seed=seed)
        if not probability_by_model:
            probability_by_model = {model_id: [] for model_id in candidates}
            fitted_models = {model_id: [] for model_id in candidates}
        if set(candidates) != set(probability_by_model):
            raise RuntimeError("candidate model set changed across deterministic seeds")
        for model_id, model in candidates.items():
            model.fit(train_features, y_train)
            fitted_models[model_id].append(model)
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
            "execution_device": (
                config.gpu_device
                if config.use_gpu and model_id in GPU_TREE_MODEL_IDS
                else "cpu"
            ),
            "gpu_requested": bool(config.use_gpu and model_id in GPU_TREE_MODEL_IDS),
            "gpu_required": bool(config.gpu_required and model_id in GPU_TREE_MODEL_IDS),
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
        "execution_device": (
            "mixed"
            if config.use_gpu and "extra_trees" in probability_by_model
            else config.gpu_device if config.use_gpu else "cpu"
        ),
        "gpu_requested": bool(config.use_gpu),
        "gpu_required": bool(config.gpu_required),
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
        fitted_models={
            model_id: tuple(models) for model_id, models in fitted_models.items()
        },
    )
