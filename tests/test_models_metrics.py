from __future__ import annotations

import numpy as np
import pandas as pd
from pytest import MonkeyPatch

from multisensor_ml import metrics as metrics_module
from multisensor_ml.metrics import (
    evaluate_probabilities,
    forecast_lead_times,
    select_event_threshold,
)
from multisensor_ml.models import fit_candidate_models, select_training_rows


def test_training_sampling_keeps_all_signal_and_caps_matched_baseline() -> None:
    rows = 100
    frame = pd.DataFrame(
        {
            "person_key": ["run/P001"] * rows,
            "timestamp_utc": pd.date_range("2026-01-01", periods=rows, freq="s", tz="UTC"),
            "context": ["focused_task"] * rows,
            "event_binary": [1] * 5 + [0] * 95,
            "hard_negative": [0] * 10 + [1] * 7 + [0] * 83,
        }
    )

    selected = select_training_rows(frame, target="event_binary", baseline_ratio=3)

    assert selected[:5].all()
    assert selected[10:17].all()
    assert selected.sum() == 5 + 7 + 15
    assert np.array_equal(
        selected,
        select_training_rows(frame, target="event_binary", baseline_ratio=3),
    )


def test_two_candidate_models_fit_and_return_probabilities() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(120, 4)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(np.int8)

    models = fit_candidate_models(x, y, random_state=17)

    assert set(models) == {"hist_gradient_boosting", "logistic_regression"}
    for model in models.values():
        probability = model.predict_proba(x)[:, 1]
        assert probability.shape == (120,)
        assert np.isfinite(probability).all()
        assert ((probability >= 0) & (probability <= 1)).all()


def test_validation_threshold_optimizes_event_f1_and_reports_false_alerts() -> None:
    truth = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0], dtype=np.int8)
    probability = np.array([0.1, 0.2, 0.8, 0.7, 0.65, 0.1, 0.2, 0.9, 0.85, 0.1])

    threshold = select_event_threshold(truth, probability, duration_hours=10 / 3600)
    metrics = evaluate_probabilities(
        truth,
        probability,
        threshold=threshold,
        duration_hours=10 / 3600,
    )

    assert threshold > 0.65
    assert metrics["event_recall"] == 1.0
    assert metrics["false_alerts"] == 0
    assert metrics["false_alerts_per_hour"] == 0.0
    assert 0 <= metrics["brier_score"] <= 1
    assert 0 <= metrics["calibration_error"] <= 1


def test_default_threshold_selection_does_not_rescan_series_per_candidate(
    monkeypatch: MonkeyPatch,
) -> None:
    rng = np.random.default_rng(20260727)
    truth = np.zeros(512, dtype=np.int8)
    truth[30:60] = 1
    truth[210:250] = 1
    truth[440:480] = 1
    probability = rng.random(512)
    probability[truth.astype(bool)] += 0.5
    probability = np.clip(probability, 0, 1).astype(np.float64)
    values = np.unique(np.round(probability, 6))
    expected = select_event_threshold(
        truth,
        probability,
        duration_hours=512 / 3600,
        candidates=values,
    )

    calls = 0
    original = metrics_module._event_counts

    def counted_event_counts(
        event_truth: np.ndarray,
        predicted: np.ndarray,
    ) -> tuple[int, int, int]:
        nonlocal calls
        calls += 1
        return original(event_truth, predicted)

    monkeypatch.setattr(metrics_module, "_event_counts", counted_event_counts)
    actual = select_event_threshold(
        truth,
        probability,
        duration_hours=512 / 3600,
    )

    assert actual == expected
    assert calls <= 1


def test_forecast_lead_time_is_measured_from_first_alert_to_onset() -> None:
    truth = np.array([0, 1, 1, 1, 0, 1, 1, 0], dtype=np.int8)
    probability = np.array([0.1, 0.2, 0.8, 0.9, 0.1, 0.7, 0.8, 0.1])

    leads = forecast_lead_times(truth, probability, threshold=0.5)

    assert leads == [2, 2]
