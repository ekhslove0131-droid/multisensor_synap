from __future__ import annotations

import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

from multisensor_ml.tree_benchmark import (
    TreeBenchmarkConfig,
    feature_importance_table,
    fit_tree_challengers,
    reference_metrics_from_validation,
    select_anchor_model,
    summarize_feature_importance,
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
    assert metadata["enable_gpu"] is False
    assert metadata["enable_internet"] is False
