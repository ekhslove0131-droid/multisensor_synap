import pandas as pd
import pytest

from multisensor_ml.phase3_evaluation import (
    evaluate_group_uplift,
    evaluate_pattern_predictions,
    phase3_adoption_decision,
    select_smallest_stable_cap,
)


def prediction_frame(probabilities: list[float]) -> pd.DataFrame:
    rows = len(probabilities)
    return pd.DataFrame(
        {
            "dataset_id": ["D1"] * (rows // 2) + ["D2"] * (rows - rows // 2),
            "person_key": ["P1"] * (rows // 2) + ["P2"] * (rows - rows // 2),
            "day_key": ["2026-01-01"] * rows,
            "session_id": ["S1"] * rows,
            "canonical_time": pd.date_range(
                "2026-01-01T00:00:00Z",
                periods=rows,
                freq="s",
            ),
            "event_binary": [0, 1, 1, 0, 0, 1, 1, 0][:rows],
            "probability": probabilities,
            "split_role": ["validation"] * rows,
            "run_id": ["R1"] * (rows // 2) + ["R2"] * (rows - rows // 2),
            "context": ["home"] * (rows // 2) + ["outside"] * (rows - rows // 2),
            "load_state": ["LOW"] * (rows // 2) + ["HIGH"] * (rows - rows // 2),
        }
    )


def test_pattern_metrics_include_accuracy_but_keep_primary_metrics() -> None:
    metrics = evaluate_pattern_predictions(
        prediction_frame([0.1, 0.8, 0.9, 0.2, 0.1, 0.8, 0.9, 0.2]),
        threshold=0.5,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["aucpr"] == 1.0
    assert metrics["event_recall"] == 1.0
    assert metrics["false_alerts_per_hour"] == 0.0


def test_group_uplift_reports_exact_absolute_and_relative_deltas() -> None:
    global_predictions = prediction_frame([0.1, 0.6, 0.4, 0.2, 0.1, 0.6, 0.4, 0.2])
    personal_predictions = prediction_frame([0.1, 0.9, 0.8, 0.2, 0.1, 0.9, 0.8, 0.2])

    result = evaluate_group_uplift(
        global_predictions,
        personal_predictions,
        group_columns=(
            "split_role",
            "run_id",
            "dataset_id",
            "person_key",
            "context",
            "load_state",
        ),
        threshold=0.5,
    )
    validation_aucpr = result.loc[
        (result["group_dimension"] == "split_role")
        & (result["group_value"] == "validation")
        & (result["metric"] == "aucpr")
    ].iloc[0]

    assert validation_aucpr["absolute_delta"] == pytest.approx(
        validation_aucpr["personal_value"] - validation_aucpr["global_value"]
    )
    assert validation_aucpr["relative_delta"] == pytest.approx(
        validation_aucpr["absolute_delta"] / validation_aucpr["global_value"]
    )
    assert set(result["group_dimension"]) == {
        "split_role",
        "run_id",
        "dataset_id",
        "person_key",
        "context",
        "load_state",
    }


def test_group_uplift_marks_single_class_support() -> None:
    global_predictions = prediction_frame([0.1, 0.6, 0.4, 0.2, 0.1, 0.6, 0.4, 0.2])
    personal_predictions = global_predictions.copy()
    global_predictions["single_class"] = global_predictions["event_binary"].map(
        {0: "NEGATIVE", 1: "POSITIVE"}
    )
    personal_predictions["single_class"] = global_predictions["single_class"]

    result = evaluate_group_uplift(
        global_predictions,
        personal_predictions,
        group_columns=("single_class",),
        threshold=0.5,
    )

    assert set(result["support_status"]) == {"INSUFFICIENT_SUPPORT"}


def test_zero_denominator_is_not_computable() -> None:
    global_predictions = prediction_frame([0.1] * 8)
    personal_predictions = prediction_frame([0.9, 0.9, 0.9, 0.1] * 2)

    result = evaluate_group_uplift(
        global_predictions,
        personal_predictions,
        group_columns=("split_role",),
        threshold=0.5,
    )
    event_recall = result.loc[result["metric"] == "event_recall"].iloc[0]

    assert event_recall["global_value"] == 0.0
    assert pd.isna(event_recall["relative_delta"])
    assert event_recall["delta_status"] == "NOT_COMPUTABLE"


def test_uplift_counts_improvements_ties_and_degradations() -> None:
    global_predictions = prediction_frame([0.1, 0.4, 0.4, 0.2, 0.1, 0.9, 0.8, 0.2])
    personal_predictions = prediction_frame([0.1, 0.9, 0.8, 0.2, 0.1, 0.4, 0.3, 0.2])

    result = evaluate_group_uplift(
        global_predictions,
        personal_predictions,
        group_columns=("dataset_id",),
        threshold=0.5,
    )
    supported = result.loc[
        (result["metric"] == "event_recall")
        & result["support_status"].eq("SUPPORTED")
    ]

    assert set(supported["direction"]) == {"IMPROVED", "DEGRADED"}
    assert supported["improvement_count"].iloc[0] == 1
    assert supported["degradation_count"].iloc[0] == 1


def test_selection_ignores_accuracy_and_chooses_smallest_stable_cap() -> None:
    oof = pd.DataFrame(
        {
            "candidate_id": ["P2-025", "P2-050", "P2-100"],
            "weight_cap": [0.25, 0.5, 1.0],
            "aucpr": [0.79, 0.80, 0.81],
            "event_recall": [0.80, 0.80, 0.81],
            "false_alerts_per_hour": [1.0, 1.0, 1.0],
            "calibration_error": [0.1, 0.1, 0.1],
            "accuracy": [0.99, 0.10, 0.05],
        }
    )

    assert select_smallest_stable_cap(oof, relative_guardrail=0.05) == "P2-025"
    changed_accuracy = oof.copy()
    changed_accuracy["accuracy"] = [0.0, 0.0, 1.0]
    assert select_smallest_stable_cap(changed_accuracy) == "P2-025"


def test_adoption_requires_primary_gain_without_safety_regression() -> None:
    assert (
        phase3_adoption_decision(
            {"aucpr_delta": 0.03, "event_recall_delta": 0.02},
            {
                "false_alerts_per_hour_delta": -0.1,
                "calibration_error_delta": 0.0,
                "degradation_count": 0,
            },
        )
        == "ADOPT"
    )
    assert (
        phase3_adoption_decision(
            {"aucpr_delta": 0.03, "event_recall_delta": 0.02},
            {
                "false_alerts_per_hour_delta": 0.2,
                "calibration_error_delta": 0.0,
                "degradation_count": 0,
            },
        )
        == "REJECT"
    )
