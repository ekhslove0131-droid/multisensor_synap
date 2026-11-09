from __future__ import annotations

import numpy as np
import pytest
from synapbridge_beta_runtime.cardiac_features import interval_features
from synapbridge_beta_runtime.feature_catalog import (
    feature_contract,
    feature_order,
    schema_sha,
)
from synapbridge_beta_runtime.wave_features import change_rate


def test_change_features_are_rates_over_elapsed_valid_seconds() -> None:
    _, _, rr_change = interval_features(np.array([1000.0, 1100.0, 1200.0]))
    assert rr_change == pytest.approx(200.0 / 2.3)
    assert change_rate(7.0, 1.0, 30.0) == pytest.approx(0.2)


def test_feature_schema_closes_units_transforms_support_and_direction() -> None:
    contract = feature_contract("h10")
    assert contract["order"] == feature_order("h10")
    descriptors = contract["features"]
    assert descriptors["rr_change_60"] == {
        "unit": "ms_per_s",
        "transform": "endpoint_delta_over_elapsed_valid_seconds",
        "window_seconds": 60,
        "minimum_intervals": 2,
        "availability": "PAST_HALF_OPEN_AVAILABLE_AT_DECISION",
        "direction": "signed",
    }
    assert schema_sha("h10") == contract["sha256"]
