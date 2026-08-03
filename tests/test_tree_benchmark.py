from __future__ import annotations

import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

from multisensor_ml.tree_benchmark import (
    TreeBenchmarkConfig,
    evaluate_anchor_gate,
    feature_group_map,
    feature_importance_table,
    fit_tree_challengers,
    permutation_importance_by_person,
    reference_metrics_from_validation,
    select_anchor_model,
    select_gated_anchor_model,
    summarize_feature_importance,
    summarize_permutation_importance,
    weighted_soft_average,
)


def test_feature_importance_table_is_normalized_and_ranked() -> None:
    table = feature_importance_table(
        model_id="xgboost",
        seed=20260725,
        feature_names=("f_a", "f_b", "f_c"),
        importances=(2.0, 1.0, 0.0),
    )

    assert list(table["feature_name"]) == ["f_a", "f_b", "f_c"]
    assert np.isclose(table["normalized_importance"].sum(), 1.0)
    assert list(table["importance_rank"]) == [1, 2, 3]
    assert table.loc[0, "normalized_importance"] > table.loc[1, "normalized_importance"]


def test_importance_summary_reports_spread_and_seed_variance() -> None:
    first = feature_importance_table(
        model_id="xgboost",
        seed=1,
        feature_names=("f_a", "f_b", "f_c"),
        importances=(8.0, 1.0, 1.0),
    )
    second = feature_importance_table(
        model_id="xgboost",
        seed=2,
        feature_names=("f_a", "f_b", "f_c"),
        importances=(6.0, 2.0, 2.0),
    )

    summary = summarize_feature_importance(pd.concat([first, second], ignore_index=True))

    row = summary.loc[summary["model_id"].eq("xgboost")].iloc[0]
    assert row["top_feature"] == "f_a"
    assert 0.0 < row["top3_share"] <= 1.0
    assert row["importance_variance_mean"] > 0.0
    assert row["importance_spread_entropy"] > 0.0


def test_weighted_soft_average_is_deterministic_and_normalized() -> None:
    result = weighted_soft_average(
        {
            "xgboost": np.array([0.2, 0.8]),
            "lightgbm": np.array([0.6, 0.4]),
        },
        weights={"xgboost": 3.0, "lightgbm": 1.0},
    )

    np.testing.assert_allclose(result, np.array([0.3, 0.7]))


def test_feature_group_map_separates_derived_and_time_features() -> None:
    groups = feature_group_map(
        [
            "autonomic_arousal__mean_300s",
            "sensory_context__std_30s",
            "time_cos",
            "weekday_sin",
            "context__wake_rest",
            "is_awake",
            "quality__missing_ratio",
        ]
    )

    assert groups["derived_signal"] == (
        "autonomic_arousal__mean_300s",
        "sensory_context__std_30s",
    )
    assert groups["time"] == ("time_cos", "weekday_sin")
    assert groups["context"] == ("context__wake_rest", "is_awake")
    assert groups["other"] == ("quality__missing_ratio",)


class _DerivedProbabilityModel:
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        value = features["autonomic_arousal__mean_5s"].to_numpy(dtype="float64")
        probability = np.where(value > 0.5, 0.9, 0.1)
        return np.column_stack([1.0 - probability, probability])


def test_permutation_importance_is_positive_repeatedly_per_person() -> None:
    rows: list[dict[str, object]] = []
    for person_index in range(4):
        for event in (0, 0, 0, 1, 1, 1):
            rows.append(
                {
                    "person_key": f"P{person_index}",
                    "autonomic_arousal__mean_5s": float(event),
                    "time_cos": float(event),
                }
            )
    features = pd.DataFrame(rows).drop(columns="person_key")
    target = np.tile(np.array([0, 0, 0, 1, 1, 1], dtype="int8"), 4)
    groups = np.repeat([f"P{index}" for index in range(4)], 6)
    detail = permutation_importance_by_person(
        {"xgboost": (_DerivedProbabilityModel(),)},
        features,
        target,
        groups,
        {"derived_signal": ("autonomic_arousal__mean_5s",), "time": ("time_cos",)},
        repeats=5,
        random_state=11,
    )
    summary = summarize_permutation_importance(detail)

    derived = summary.loc[
        summary["group_id"].eq("derived_signal") & summary["model_id"].eq("xgboost")
    ].iloc[0]
    assert derived["mean_importance_drop"] > 0.0
    assert derived["positive_person_fraction"] >= 0.75
    assert bool(derived["repeated_positive"]) is True


def test_anchor_gate_requires_time_ablation_and_derived_group_support() -> None:
    full = pd.DataFrame(
        [
            {"model_id": "xgboost", "aucpr": 0.80, "event_recall": 0.70},
            {"model_id": "lightgbm", "aucpr": 0.81, "event_recall": 0.71},
        ]
    )
    ablation = pd.DataFrame(
        [
            {"model_id": "xgboost", "aucpr": 0.79, "event_recall": 0.69},
            {"model_id": "lightgbm", "aucpr": 0.60, "event_recall": 0.70},
        ]
    )
    permutation = pd.DataFrame(
        [
            {
                "model_id": "xgboost",
                "group_id": "derived_signal",
                "mean_importance_drop": 0.02,
                "positive_person_fraction": 1.0,
                "mean_person_repeat_positive_rate": 0.9,
            },
            {
                "model_id": "lightgbm",
                "group_id": "derived_signal",
                "mean_importance_drop": -0.01,
                "positive_person_fraction": 0.5,
                "mean_person_repeat_positive_rate": 0.4,
            },
        ]
    )
    gate = evaluate_anchor_gate(full, ablation, permutation)

    assert bool(gate.loc[gate["model_id"].eq("xgboost"), "gate_pass"].iloc[0]) is True
    assert bool(gate.loc[gate["model_id"].eq("lightgbm"), "gate_pass"].iloc[0]) is False
    stability = pd.DataFrame(
        [
            {"model_id": "xgboost", "importance_variance_mean": 0.01, "top3_share": 0.4},
            {"model_id": "lightgbm", "importance_variance_mean": 0.001, "top3_share": 0.3},
        ]
    )
    gated_metrics = full.assign(
        event_f1=0.7,
        false_alerts_per_hour=0.1,
        calibration_error=0.1,
    )
    assert select_gated_anchor_model(gated_metrics, stability, gate) == "xgboost"


def test_anchor_prefers_stable_model_within_aucpr_guardrail() -> None:
    metrics = pd.DataFrame(
        [
            {
                "model_id": "xgboost",
                "aucpr": 0.82,
                "event_f1": 0.70,
                "false_alerts_per_hour": 0.04,
                "calibration_error": 0.08,
            },
            {
                "model_id": "lightgbm",
                "aucpr": 0.81,
                "event_f1": 0.69,
                "false_alerts_per_hour": 0.03,
                "calibration_error": 0.07,
            },
        ]
    )
    stability = pd.DataFrame(
        [
            {
                "model_id": "xgboost",
                "importance_variance_mean": 0.010,
                "top3_share": 0.70,
            },
            {
                "model_id": "lightgbm",
                "importance_variance_mean": 0.001,
                "top3_share": 0.35,
            },
        ]
    )

    assert select_anchor_model(metrics, stability, aucpr_guardrail=0.02) == "lightgbm"


def test_anchor_uses_stability_fallback_when_smoke_sample_has_no_positive() -> None:
    metrics = pd.DataFrame(
        [
            {
                "model_id": "xgboost",
                "aucpr": np.nan,
                "event_f1": 0.0,
                "false_alerts_per_hour": 0.0,
                "calibration_error": 0.0,
            },
            {
                "model_id": "lightgbm",
                "aucpr": np.nan,
                "event_f1": 0.0,
                "false_alerts_per_hour": 0.0,
                "calibration_error": 0.0,
            },
        ]
    )
    stability = pd.DataFrame(
        [
            {"model_id": "xgboost", "importance_variance_mean": 0.02, "top3_share": 0.50},
            {"model_id": "lightgbm", "importance_variance_mean": 0.01, "top3_share": 0.40},
        ]
    )

    assert select_anchor_model(metrics, stability) == "lightgbm"


def test_xgboost_lightgbm_challengers_and_ensemble_share_contract() -> None:
    rng = np.random.default_rng(7)
    features = pd.DataFrame(rng.normal(size=(40, 4)), columns=["f_a", "f_b", "f_c", "f_d"])
    target = (features["f_a"] + features["f_b"] > 0).astype("int8")
    result = fit_tree_challengers(
        features.iloc[:30],
        target.iloc[:30],
        features.iloc[30:],
        target.iloc[30:],
        config=TreeBenchmarkConfig(seed_count=1, n_estimators=4, n_jobs=1),
    )

    assert set(result.metrics["model_id"]) == {"xgboost", "lightgbm", "soft_ensemble"}
    assert result.anchor_model in {"xgboost", "lightgbm"}
    assert set(result.predictions["model_id"]) == {
        "xgboost",
        "lightgbm",
        "soft_ensemble",
    }
    assert set(result.feature_importance["model_id"]) == {"xgboost", "lightgbm"}


def test_reference_metrics_keep_existing_hgb_and_logistic_rows_separate() -> None:
    source = pd.DataFrame(
        [
            {
                "head": "event",
                "model_name": "hist_gradient_boosting",
                "role": "validation",
                "threshold": 0.99,
                "aucpr": 0.51,
                "event_recall": 0.62,
                "event_f1": 0.74,
                "false_alerts_per_hour": 0.01,
                "brier_score": 0.06,
                "calibration_error": 0.18,
            },
            {
                "head": "stage",
                "model_name": "hist_gradient_boosting",
                "role": "validation",
                "macro_f1": 0.63,
            },
        ]
    )

    result = reference_metrics_from_validation(source)

    assert list(result["model_id"]) == ["existing_hist_gradient_boosting"]
    assert result.loc[0, "evaluation_scope"] == "existing_full_validation"


def test_kaggle_tree_notebook_is_cpu_only_and_locked_test_safe() -> None:
    notebook = nbformat.read("kaggle/09_tree_model_benchmark.ipynb", as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    metadata = json.loads(
        Path("kaggle/09_tree_model_benchmark.kernel-metadata.json").read_text()
    )

    assert "RUN_LOCKED_TEST = False" in source
    assert "USE_GPU = False" in source
    assert "N_JOBS = 1" in source
    assert "NanumGothic" in source
    assert "soft_ensemble" in source
    assert "feature_group_map" in source
    assert "permutation_importance_by_person" in source
    assert "evaluate_anchor_gate" in source
    assert "permutation_summary.parquet" in source
    assert "time_ablation_metrics" in source
    assert metadata["enable_gpu"] is False
    assert metadata["enable_internet"] is False
