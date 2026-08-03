from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from multisensor_ml.dual_sensor_onnx import feature_names_for_variant
from multisensor_ml.optuna_dual_sensor import (
    DualSensorTuningConfig,
    load_train_validation_from_frames,
)


def test_tuning_config_is_deterministic_and_validation_only() -> None:
    config = DualSensorTuningConfig(
        project_root=Path("/tmp/project"),
        series_id="mvp3-oracle-v1",
        output_root=Path("/tmp/models"),
        n_trials=3,
        seed=42,
        max_train_rows=100,
        max_validation_rows=80,
    )

    assert config.n_trials == 3
    assert config.locked_test_read is False
    assert config.primary_metric == "aucpr"


def test_load_train_validation_never_returns_locked_test() -> None:
    class FakeConfig:
        max_train_rows = 0
        max_validation_rows = 0

    train = pd.DataFrame({"person_key": ["p1"], "event_binary": [1]})
    validation = pd.DataFrame({"person_key": ["p2"], "event_binary": [0]})
    result = load_train_validation_from_frames(train, validation, FakeConfig())

    assert set(result.train["person_key"]) == {"p1"}
    assert set(result.validation["person_key"]) == {"p2"}
    assert set(result.train["person_key"]).isdisjoint(result.validation["person_key"])
    with pytest.raises(ValueError, match="no features"):
        feature_names_for_variant(("autonomic_arousal__robust_z",), "galaxy_watch")
