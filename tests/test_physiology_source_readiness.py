import pytest

from multisensor_ml.physiology_source_readiness import (
    FeatureSpec,
    SourceWindow,
    audit_source_windows,
    validate_timesfm25_context,
)


def window(start=0, *, quality=True, values=(70.0,), capture="capture-a"):
    return SourceWindow("subject-a", capture, start, start + 10000, values, quality,
                        True, True, "ELIGIBLE_NO_PATTERN")


def test_h10_is_auditable_without_watch_or_muse_and_never_training_ready():
    result = audit_source_windows("h10", (FeatureSpec("hr", "HR"),),
                                  (window(), window(10000)))
    assert result["valid_seconds"] == 20
    assert result["longest_valid_run_seconds"] == 20
    assert result["fit_call_count"] == 0
    assert result["training_ready"] is False
    assert "subject-a" not in str(result)
    assert "capture-a" not in str(result)


@pytest.mark.parametrize("origin", ["ACC", "GYRO", "BATTERY", "CONTACT", "QUALITY"])
def test_quality_or_motion_cannot_enter_predictor_even_under_neutral_name(origin):
    with pytest.raises(ValueError, match="physiology"):
        audit_source_windows("watch", (FeatureSpec("neutral", origin),), ())


def test_missing_features_and_quality_do_not_count_as_valid_time():
    result = audit_source_windows("h10", (FeatureSpec("hr", "HR"),),
                                  (window(quality=None), window(10000, values=(float("nan"),)),
                                   window(20000), window(40000)))
    assert result["valid_seconds"] == 20
    assert result["longest_valid_run_seconds"] == 10
    assert result["excluded_reasons"] == {"QUALITY_UNVERIFIED": 1, "FEATURE_INVALID": 1}


def test_captures_are_not_stitched_and_overlaps_are_rejected():
    specs = (FeatureSpec("hr", "HR"),)
    result = audit_source_windows("h10", specs, (window(), window(10000, capture="capture-b")))
    assert result["longest_valid_run_seconds"] == 10
    with pytest.raises(ValueError, match=r"duplicate|overlap"):
        audit_source_windows("h10", specs, (window(), window()))


def test_eeg_keeps_deviation_semantics_and_no_shared_duration_gate():
    result = audit_source_windows("muse", (FeatureSpec("alpha_relative", "EEG"),), ())
    assert result["output_semantics"] == "PERSONAL_PHYSIOLOGY_DEVIATION"
    assert result["minimum_training_seconds"] is None
    assert result["status"] == "NO_SOURCE_WINDOWS"


@pytest.mark.parametrize("times,values", [([0, 20000], [1., 2.]),
                                          ([0, 10000], [1., float("nan")])])
def test_timesfm_input_cannot_silently_interpolate(times, values):
    with pytest.raises(ValueError):
        validate_timesfm25_context(times, values, grid_ms=10000)


def test_timesfm_validation_does_not_require_32_points_or_run_forecast():
    assert validate_timesfm25_context([0, 10000], [1., 2.], grid_ms=10000) == (1., 2.)
