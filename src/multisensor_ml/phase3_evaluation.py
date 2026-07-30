from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from multisensor_ml.metrics import evaluate_segmented_probabilities

PRIMARY_GROUP_METRICS: tuple[str, ...] = (
    "aucpr",
    "event_recall",
    "event_f1",
    "false_alerts_per_hour",
    "calibration_error",
    "accuracy",
)
ALIGNMENT_COLUMNS: tuple[str, ...] = (
    "dataset_id",
    "person_key",
    "day_key",
    "session_id",
    "canonical_time",
    "event_binary",
)


def evaluate_pattern_predictions(
    frame: pd.DataFrame,
    *,
    threshold: float,
) -> dict[str, float]:
    result = evaluate_segmented_probabilities(
        frame,
        threshold=threshold,
        truth_column="event_binary",
        probability_column="probability",
    )
    truth = frame["event_binary"].to_numpy(dtype=np.int8)
    predicted = frame["probability"].to_numpy(dtype=np.float64) >= threshold
    metrics = {
        key: float(value)
        for key, value in result.items()
        if isinstance(value, (int, float, np.integer, np.floating))
    }
    metrics["accuracy"] = float(accuracy_score(truth, predicted))
    return metrics


def _validate_aligned_predictions(
    global_predictions: pd.DataFrame,
    personal_predictions: pd.DataFrame,
    group_columns: Sequence[str],
) -> None:
    required = {*ALIGNMENT_COLUMNS, "probability", *group_columns}
    for label, frame in (
        ("global", global_predictions),
        ("personal", personal_predictions),
    ):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{label} predictions missing columns: {missing}")
    pd.testing.assert_frame_equal(
        global_predictions.loc[:, [*ALIGNMENT_COLUMNS, *group_columns]].reset_index(
            drop=True
        ),
        personal_predictions.loc[:, [*ALIGNMENT_COLUMNS, *group_columns]].reset_index(
            drop=True
        ),
        check_dtype=False,
        obj="global and personal prediction alignment",
    )


def _group_rows(
    global_group: pd.DataFrame,
    personal_group: pd.DataFrame,
    *,
    dimension: str,
    value: str,
    threshold: float,
) -> list[dict[str, Any]]:
    if global_group["event_binary"].nunique(dropna=False) < 2:
        return [
            {
                "group_dimension": dimension,
                "group_value": value,
                "metric": metric,
                "global_value": np.nan,
                "personal_value": np.nan,
                "absolute_delta": np.nan,
                "relative_delta": np.nan,
                "support_status": "INSUFFICIENT_SUPPORT",
                "delta_status": "NOT_COMPUTABLE",
                "direction": "NOT_COMPUTABLE",
            }
            for metric in PRIMARY_GROUP_METRICS
        ]

    global_metrics = evaluate_pattern_predictions(global_group, threshold=threshold)
    personal_metrics = evaluate_pattern_predictions(personal_group, threshold=threshold)
    rows: list[dict[str, Any]] = []
    for metric in PRIMARY_GROUP_METRICS:
        global_value = global_metrics[metric]
        personal_value = personal_metrics[metric]
        absolute_delta = personal_value - global_value
        if np.isclose(global_value, 0.0):
            relative_delta = np.nan
            delta_status = "NOT_COMPUTABLE"
        else:
            relative_delta = absolute_delta / global_value
            delta_status = "COMPUTABLE"
        if np.isclose(absolute_delta, 0.0):
            direction = "TIED"
        elif metric in {"false_alerts_per_hour", "calibration_error"}:
            direction = "IMPROVED" if absolute_delta < 0 else "DEGRADED"
        elif absolute_delta > 0:
            direction = "IMPROVED"
        else:
            direction = "DEGRADED"
        rows.append(
            {
                "group_dimension": dimension,
                "group_value": value,
                "metric": metric,
                "global_value": global_value,
                "personal_value": personal_value,
                "absolute_delta": absolute_delta,
                "relative_delta": relative_delta,
                "support_status": "SUPPORTED",
                "delta_status": delta_status,
                "direction": direction,
            }
        )
    return rows


def evaluate_group_uplift(
    global_predictions: pd.DataFrame,
    personal_predictions: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    threshold: float,
) -> pd.DataFrame:
    """Compare a personal candidate against the fixed G0 denominator."""

    _validate_aligned_predictions(
        global_predictions,
        personal_predictions,
        group_columns,
    )
    rows: list[dict[str, Any]] = []
    for dimension in group_columns:
        for value, global_group in global_predictions.groupby(
            dimension,
            sort=True,
            dropna=False,
        ):
            positions = global_group.index
            personal_group = personal_predictions.loc[positions]
            rows.extend(
                _group_rows(
                    global_group,
                    personal_group,
                    dimension=dimension,
                    value=str(value),
                    threshold=threshold,
                )
            )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("group uplift produced no groups")
    for column in ("improvement_count", "tie_count", "degradation_count"):
        result[column] = 0
    supported = result["support_status"].eq("SUPPORTED")
    for _metric, index in result.loc[supported].groupby("metric").groups.items():
        directions = result.loc[index, "direction"]
        result.loc[index, "improvement_count"] = int(directions.eq("IMPROVED").sum())
        result.loc[index, "tie_count"] = int(directions.eq("TIED").sum())
        result.loc[index, "degradation_count"] = int(directions.eq("DEGRADED").sum())
    for column in ("improvement_count", "tie_count", "degradation_count"):
        result[column] = result[column].astype("int64")
    return result


def select_smallest_stable_cap(
    oof_metrics: pd.DataFrame,
    *,
    relative_guardrail: float = 0.05,
) -> str:
    """Select the least-personal candidate near the best primary OOF metrics."""

    required = {
        "candidate_id",
        "weight_cap",
        "aucpr",
        "event_recall",
        "false_alerts_per_hour",
        "calibration_error",
    }
    missing = sorted(required.difference(oof_metrics.columns))
    if missing:
        raise ValueError(f"OOF selection missing columns: {missing}")
    if not 0 <= relative_guardrail < 1:
        raise ValueError("relative guardrail must be in [0, 1)")
    ordered = oof_metrics.sort_values("weight_cap", kind="mergesort")
    best_aucpr = float(ordered["aucpr"].max())
    best_recall = float(ordered["event_recall"].max())
    best_false_alerts = float(ordered["false_alerts_per_hour"].min())
    best_ece = float(ordered["calibration_error"].min())
    stable = ordered.loc[
        ordered["aucpr"].ge(best_aucpr * (1 - relative_guardrail))
        & ordered["event_recall"].ge(best_recall * (1 - relative_guardrail))
        & ordered["false_alerts_per_hour"].le(
            best_false_alerts * (1 + relative_guardrail) + 1e-12
        )
        & ordered["calibration_error"].le(
            best_ece * (1 + relative_guardrail) + 1e-12
        )
    ]
    if stable.empty:
        stable = ordered.sort_values(
            ["aucpr", "event_recall", "weight_cap"],
            ascending=[False, False, True],
            kind="mergesort",
        ).head(1)
    return str(stable.iloc[0]["candidate_id"])


def phase3_adoption_decision(
    metrics: Mapping[str, float],
    uplift: Mapping[str, float | int],
) -> str:
    primary_gain = (
        float(metrics.get("aucpr_delta", 0.0)) > 0
        and float(metrics.get("event_recall_delta", 0.0)) >= 0
    )
    safe = (
        float(uplift.get("false_alerts_per_hour_delta", 0.0)) <= 0
        and float(uplift.get("calibration_error_delta", 0.0)) <= 0
        and int(uplift.get("degradation_count", 0)) == 0
    )
    return "ADOPT" if primary_gain and safe else "REJECT"
