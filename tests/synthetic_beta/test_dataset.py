from __future__ import annotations

import pytest
from synapbridge_beta_runtime.feature_catalog import feature_order
from synapbridge_beta_runtime.types import BetaScope

from multisensor_ml.beta_training.cohort import _feature_frame
from multisensor_ml.beta_training.labels import labels_at, temporal_stage_at
from multisensor_ml.beta_training.split import validate_splits


def test_subject_splits_are_disjoint_and_locked_is_separate() -> None:
    splits = validate_splits([
        {"subject_uuid": "train", "split": "TRAIN"},
        {"subject_uuid": "validation", "split": "VALIDATION"},
        {"subject_uuid": "test", "split": "LOCKED_TEST"},
    ])
    assert [split.role for split in splits] == ["TRAIN", "VALIDATION", "LOCKED_TEST"]
    with pytest.raises(ValueError, match="split_subject_overlap"):
        validate_splits([
            {"subject_uuid": "same", "split": "TRAIN"},
            {"subject_uuid": "same", "split": "VALIDATION"},
            {"subject_uuid": "test", "split": "LOCKED_TEST"},
        ])


def test_next_onset_label_excludes_target_already_in_progress() -> None:
    events = [{"subject_uuid": "subject", "onset_ms": 100_000, "offset_ms": 160_000}]
    assert labels_at(events, "subject", 50_000) == (0, 1)
    assert labels_at(events, "subject", 120_000) == (1, None)
    assert labels_at(events, "subject", 170_000) == (0, 0)


def test_reference_temporal_phase_is_evaluation_only_truth() -> None:
    event = {
        "subject_uuid": "subject", "pre_early_start_ms": 40_000,
        "pre_late_start_ms": 70_000, "onset_ms": 100_000,
        "offset_ms": 280_000, "recovery_early_start_ms": 280_000,
        "recovery_late_start_ms": 340_000, "phase_end_ms": 400_000,
    }
    assert [temporal_stage_at([event], "subject", value) for value in
            (40_000, 70_000, 100_000, 280_000, 340_000, 400_000)] == [1, 2, 3, 4, 5, 0]


def test_canonical_json_key_order_is_rebound_to_frozen_catalogue() -> None:
    family = "h10"
    raw = {name: float(index) for index, name in enumerate(reversed(feature_order(family)))}
    row = {"decision_ms": 180_000, "values": dict(sorted(raw.items())),
           "valid_support_ms": [170_000, 180_000], "quality": 1.0,
           "schema_sha": "a" * 64}
    frame = _feature_frame(
        BetaScope("10000000-0000-4000-8000-000000000001", 1, ""),
        "20000000-0000-4000-8000-000000000001", family, row,
    )
    assert tuple(frame.values) == feature_order(family)
