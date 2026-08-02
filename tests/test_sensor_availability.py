from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression

from multisensor_ml import sensor_availability
from multisensor_ml.sensor_availability import (
    AVAILABILITY_PROFILES,
    availability_feature_names,
    fit_availability_variant,
    load_availability_variant,
    predict_availability_variant,
    sensor_availability_manifest,
)


def test_logistic_loader_restores_removed_multi_class_attribute() -> None:
    model = LogisticRegression()
    if hasattr(model, "multi_class"):
        del model.multi_class

    loaded = sensor_availability._ensure_logistic_compatibility(model)

    assert loaded.multi_class == "auto"


def test_profiles_cover_all_nonempty_combinations_of_three_devices() -> None:
    assert set(AVAILABILITY_PROFILES) == {
        "watch_only",
        "polar_only",
        "muse_only",
        "watch_polar",
        "watch_muse",
        "polar_muse",
        "watch_polar_muse",
    }
    assert all(profile.devices for profile in AVAILABILITY_PROFILES.values())


def test_availability_features_keep_common_context_and_only_selected_factors() -> None:
    columns = [
        "autonomic_arousal__robust_z",
        "motor_activation__robust_z",
        "cognitive_load__robust_z",
        "sensory_context__mean_30s",
        "social_context__slope_60s",
        "time_sin",
        "context__sleep",
    ]

    selected = availability_feature_names(columns, "watch_only")

    assert selected == [
        "autonomic_arousal__robust_z",
        "motor_activation__robust_z",
        "sensory_context__mean_30s",
        "time_sin",
        "context__sleep",
    ]
    assert "cognitive_load__robust_z" not in selected


def test_manifest_marks_availability_benchmark_as_oracle_only() -> None:
    manifest = sensor_availability_manifest(
        profiles={"watch_only": {"selected_event_model": "logistic_regression"}},
        source_dataset_hash="a" * 64,
    )

    assert manifest["status"] == "oracle/sanity"
    assert manifest["real_data_status"] == "NOT VERIFIED"
    assert manifest["locked_test_read"] is False
    assert manifest["profiles"]["watch_only"]["selected_event_model"] == (
        "logistic_regression"
    )


def test_availability_variant_round_trips_event_and_stage_predictions(tmp_path) -> None:
    rows = []
    stage_codes = ("LOW", "MEDIUM", "HIGH", "DECREASING", "RECOVERY")
    for person_index in range(6):
        person = f"p{person_index}"
        for time_index in range(24):
            event = int(time_index % 6 in {3, 4})
            rows.append(
                {
                    "person_key": person,
                    "session_id": "session-a",
                    "timestamp_utc": pd.Timestamp("2026-01-01", tz="UTC")
                    + pd.Timedelta(seconds=time_index),
                    "context": "wake_rest",
                    "hard_negative": int(time_index % 11 == 0 and not event),
                    "stage_code": (
                        stage_codes[time_index % len(stage_codes)]
                        if event
                        else "NO_EVENT"
                    ),
                    "autonomic_arousal__robust_z": float(event * 3 + time_index / 100),
                    "motor_activation__robust_z": float(event * 2),
                    "sensory_context__mean_30s": float(event),
                    "time_sin": float(time_index),
                    "is_awake": 1,
                    "context__wake_rest": 1,
                }
            )
    frame = pd.DataFrame(rows)
    train = frame.loc[frame["person_key"].isin({"p0", "p1", "p2", "p3"})].reset_index(
        drop=True
    )
    validation = frame.loc[frame["person_key"].isin({"p4", "p5"})].reset_index(drop=True)
    result = fit_availability_variant(
        train,
        validation,
        profile_id="watch_only",
        output_root=tmp_path / "watch_only",
        random_state=17,
    )

    loaded = load_availability_variant(result.root)
    predictions = predict_availability_variant(loaded, validation)

    assert len(predictions) == len(validation)
    assert {"event_probability", "predicted_stage"}.issubset(predictions.columns)
    assert predictions["event_probability"].between(0, 1).all()
