from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from multisensor_ml.bundle import load_model_bundle, write_model_bundle
from multisensor_ml.models import fit_candidate_models
from multisensor_ml.stress import StressScenario, apply_stress


def test_stress_is_deterministic_and_does_not_mutate_source() -> None:
    source = pd.DataFrame(
        {
            "autonomic_arousal__robust_z": np.arange(20, dtype=np.float32),
            "motor_activation__robust_z": np.arange(20, dtype=np.float32),
        }
    )
    original = source.copy()
    scenario = StressScenario(kind="gaussian", magnitude=0.1, seed=42)

    first = apply_stress(source, scenario)
    second = apply_stress(source, scenario)

    pd.testing.assert_frame_equal(source, original)
    pd.testing.assert_frame_equal(first, second)
    assert not first.equals(source)


@pytest.mark.parametrize(
    ("scenario", "expected_zero_rows"),
    [
        (StressScenario(kind="block_missing", magnitude=5, seed=3), 5),
        (StressScenario(kind="latent_dropout", magnitude=1, seed=3), 20),
    ],
)
def test_stress_missing_and_axis_dropout_are_explicit(
    scenario: StressScenario,
    expected_zero_rows: int,
) -> None:
    source = pd.DataFrame(
        {
            "autonomic_arousal__robust_z": np.ones(20, dtype=np.float32),
            "motor_activation__robust_z": np.ones(20, dtype=np.float32),
        }
    )

    stressed = apply_stress(source, scenario)

    assert int((stressed == 0).all(axis=1).sum()) == expected_zero_rows


def test_skops_bundle_roundtrip_and_hash_verification(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    features = pd.DataFrame(
        rng.normal(size=(100, 3)).astype(np.float32),
        columns=["f1", "f2", "f3"],
    )
    target = (features["f1"] > 0).astype("int8").to_numpy()
    model = fit_candidate_models(features.to_numpy(), target)["logistic_regression"]
    expected = model.predict_proba(features)[:, 1]

    bundle = write_model_bundle(
        tmp_path / "bundle",
        models={("event_binary", "logistic_regression"): model},
        feature_names=list(features.columns),
        thresholds={("event_binary", "logistic_regression"): 0.5},
        lineage={"dataset_id": "d1", "split_hash": "a" * 64},
        global_baseline=pd.DataFrame({"context": ["focused_task"]}),
        personal_baseline=pd.DataFrame({"person_key": ["run/P001"]}),
        metrics=pd.DataFrame({"metric": ["aucpr"], "value": [1.0]}),
        predictions=pd.DataFrame({"probability": expected}),
        uv_lock=tmp_path / "missing-uv.lock",
    )
    loaded = load_model_bundle(bundle)

    actual = loaded.models[("event_binary", "logistic_regression")].predict_proba(
        features
    )[:, 1]
    np.testing.assert_allclose(actual, expected)
    assert not list(bundle.rglob("*.pkl"))
    assert not list(bundle.rglob("*.joblib"))

    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    model_path = bundle / manifest["models"][0]["path"]
    model_path.write_bytes(model_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_model_bundle(bundle)
