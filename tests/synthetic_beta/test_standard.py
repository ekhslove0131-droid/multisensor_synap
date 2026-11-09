from __future__ import annotations

from types import MappingProxyType

import pytest
from synapbridge_beta_runtime.feature_types import FeatureFrame, SourceScore
from synapbridge_beta_runtime.standard import (
    calibrate_reference,
    combine_deviation,
    fit_standard,
    score_reference,
)
from synapbridge_beta_runtime.types import BetaScope


def _frames(*, count: int = 180, duplicate: bool = False) -> list[FeatureFrame]:
    scope = BetaScope(
        "10000000-0000-4000-8000-000000000001", 1,
        "20000000-0000-4000-8000-000000000001",
    )
    result = [FeatureFrame(
        scope, (index + 1) * 10_000, "h10",
        MappingProxyType({"hr_median_180": 70.0 + index % 5,
                          "rr_rmssd_180": 20.0 + index % 7}),
        (index * 10_000, (index + 1) * 10_000), 1.0, "a" * 64,
    ) for index in range(count)]
    return [*result, result[-1]] if duplicate else result


def test_source_rms_is_not_time_integral() -> None:
    scores = [SourceScore("watch", {}, 3.0, {}, 1.0, None),
              SourceScore("h10", {}, 4.0, {}, 1.0, None)]
    assert combine_deviation(scores) == pytest.approx(3.5355339059)
    assert combine_deviation([]) is None


def test_train_standard_and_personal_calibration_use_unique_1800_seconds() -> None:
    frames = _frames()
    standard = fit_standard(frames, "h10")
    reference = calibrate_reference(standard, frames)
    score = score_reference(frames[-1], reference)
    assert standard.feature_order == ("hr_median_180", "rr_rmssd_180")
    assert reference.scored_valid_seconds == 1800
    assert score.z_rms is not None
    with pytest.raises(ValueError, match="calibration_support_overlap"):
        calibrate_reference(standard, _frames(duplicate=True))


def test_personal_calibration_rejects_mixed_subject_scope() -> None:
    frames = _frames()
    standard = fit_standard(frames, "h10")
    changed = frames[-1]
    frames[-1] = FeatureFrame(
        BetaScope(changed.scope.run_uuid, changed.scope.epoch,
                  "20000000-0000-4000-8000-000000000099"),
        changed.decision_ms, changed.family, changed.values,
        changed.valid_support_ms, changed.quality, changed.schema_sha,
    )
    with pytest.raises(ValueError, match="calibration_scope_mismatch"):
        calibrate_reference(standard, frames)
