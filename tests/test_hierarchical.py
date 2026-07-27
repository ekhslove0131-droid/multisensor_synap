from __future__ import annotations

import numpy as np
import pandas as pd

from multisensor_ml.hierarchical import (
    behavior_support_status,
    decode_stage_sequence,
    fit_behavior_candidates,
    fit_stage_candidates,
    grouped_oof_probabilities,
    grouped_oof_stage_probabilities,
    personal_calibration_status,
)
from multisensor_ml.result_router import route_prediction_ko


def test_stage_decoder_enforces_forward_progression_and_abstains_on_invalid_rows() -> None:
    event_probability = np.array([0.1, 0.9, 0.9, 0.9, 0.9, 0.9])
    stage_probability = np.array(
        [
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.05, 0.05, 0.80, 0.05, 0.05],
            [0.05, 0.80, 0.05, 0.05, 0.05],
            [0.05, 0.05, 0.80, 0.05, 0.05],
            [0.05, 0.05, 0.05, 0.80, 0.05],
            [0.05, 0.05, 0.05, 0.05, 0.80],
        ]
    )

    decoded = decode_stage_sequence(
        event_probability,
        stage_probability,
        event_threshold=0.5,
        valid_mask=np.array([True, True, True, True, False, True]),
    )

    assert decoded == [
        "NO_EVENT",
        "LOW",
        "MEDIUM",
        "HIGH",
        "NOT_DECISIONABLE",
        "RECOVERY",
    ]


def test_grouped_oof_assigns_each_person_to_exactly_one_fold() -> None:
    rng = np.random.default_rng(17)
    groups = np.repeat([f"P{index:02d}" for index in range(8)], 10)
    features = rng.normal(size=(80, 3)).astype(np.float32)
    target = (features[:, 0] > 0).astype(np.int8)

    result = grouped_oof_probabilities(
        features,
        target,
        groups,
        random_state=17,
        folds=4,
    )

    assert np.isfinite(result.probability).all()
    assignment = pd.DataFrame({"group": groups, "fold": result.fold})
    assert assignment.groupby("group")["fold"].nunique().eq(1).all()


def test_grouped_stage_oof_has_five_probabilities_without_person_leakage() -> None:
    rng = np.random.default_rng(19)
    groups = np.repeat([f"P{index:02d}" for index in range(10)], 10)
    target = np.tile(np.arange(5, dtype=np.int8), 20)
    features = np.column_stack(
        [target + rng.normal(0, 0.1, len(target)), rng.normal(size=len(target))]
    ).astype(np.float32)

    result = grouped_oof_stage_probabilities(
        features,
        target,
        groups,
        random_state=19,
        folds=5,
    )

    assert result.probability.shape == (100, 5)
    np.testing.assert_allclose(result.probability.sum(axis=1), 1.0)
    assignment = pd.DataFrame({"group": groups, "fold": result.fold})
    assert assignment.groupby("group")["fold"].nunique().eq(1).all()


def test_behavior_support_requires_train_and_validation_positives() -> None:
    assert (
        behavior_support_status(train_positive=19, validation_positive=10)
        == "INSUFFICIENT_LABEL_SUPPORT"
    )
    assert (
        behavior_support_status(train_positive=20, validation_positive=4)
        == "INSUFFICIENT_LABEL_SUPPORT"
    )
    assert (
        behavior_support_status(train_positive=20, validation_positive=5)
        == "SUPPORTED"
    )


def test_personal_calibration_requires_events_support_and_multiple_days() -> None:
    assert (
        personal_calibration_status(
            event_count=19,
            positive_count=10,
            negative_count=9,
            distinct_days=4,
        )
        == "GLOBAL_STD_ONLY"
    )
    assert (
        personal_calibration_status(
            event_count=20,
            positive_count=5,
            negative_count=15,
            distinct_days=2,
        )
        == "CALIBRATION_CANDIDATE"
    )
    assert (
        personal_calibration_status(
            event_count=50,
            positive_count=20,
            negative_count=30,
            distinct_days=3,
        )
        == "CHAMPION_COMPARISON_ALLOWED"
    )


def test_korean_router_preserves_raw_probability_but_abstains_for_ood() -> None:
    routed = route_prediction_ko(
        {
            "event_probability": 0.87,
            "stage_code": "HIGH",
            "ood_status": "OOD_MONITOR",
            "behavior_probabilities": {"ear_covering": 0.71},
        },
        router_version="ko-v1",
    )

    assert routed["판정"] == "판단 보류"
    assert routed["단계"] == "판단 보류"
    assert routed["raw_event_probability"] == 0.87
    assert routed["raw_behavior_probabilities"] == {"ear_covering": 0.71}


def test_stage_candidates_select_event_and_multiclass_models_on_validation() -> None:
    rng = np.random.default_rng(23)
    stages = ["NO_EVENT", "LOW", "MEDIUM", "HIGH", "DECREASING", "RECOVERY"]

    def make_frame(rows_per_stage: int) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for stage_index, stage in enumerate(stages):
            for row_index in range(rows_per_stage):
                rows.append(
                    {
                        "person_key": f"P{row_index % 8:02d}",
                        "timestamp_utc": pd.Timestamp("2026-01-01", tz="UTC")
                        + pd.Timedelta(seconds=len(rows)),
                        "context": "focused_task",
                        "hard_negative": int(stage == "NO_EVENT" and row_index % 7 == 0),
                        "stage_code": stage,
                        "f1": stage_index + rng.normal(0, 0.1),
                        "f2": rng.normal(),
                    }
                )
        return pd.DataFrame(rows)

    result = fit_stage_candidates(
        make_frame(30),
        make_frame(10),
        feature_names=["f1", "f2"],
        random_state=17,
    )

    assert result.selected_event_model in {
        "logistic_regression",
        "hist_gradient_boosting",
    }
    assert result.selected_stage_model in {
        "logistic_regression",
        "hist_gradient_boosting",
    }
    assert 0 <= result.event_threshold <= 1
    assert set(result.metrics["head"]) == {"event", "stage"}


def test_behavior_candidates_skip_unsupported_labels_and_select_supported_ones() -> None:
    rng = np.random.default_rng(29)
    train = pd.DataFrame(
        {
            "f1": rng.normal(size=80),
            "event_oof_probability": rng.random(80),
            "ear_covering": [1] * 30 + [0] * 50,
            "exit_attempt": [1] * 5 + [0] * 75,
        }
    )
    validation = pd.DataFrame(
        {
            "f1": rng.normal(size=30),
            "event_oof_probability": rng.random(30),
            "ear_covering": [1] * 10 + [0] * 20,
            "exit_attempt": [1] * 2 + [0] * 28,
        }
    )

    result = fit_behavior_candidates(
        train,
        validation,
        feature_names=["f1", "event_oof_probability"],
        behavior_codes=["ear_covering", "exit_attempt"],
        random_state=17,
    )

    assert result.status_by_behavior["ear_covering"] == "SUPPORTED"
    assert result.status_by_behavior["exit_attempt"] == "INSUFFICIENT_LABEL_SUPPORT"
    assert result.selected_model_by_behavior["ear_covering"] in {
        "logistic_regression",
        "hist_gradient_boosting",
    }
    assert "exit_attempt" not in result.selected_model_by_behavior
