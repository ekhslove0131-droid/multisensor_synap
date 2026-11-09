"""Local per-source audit preparation; never admits a cohort or invokes fitting.

Inputs are normalized by a future Cloud adapter. This is not an authorized-view
schema, raw decoder, or a production feature-schema approval mechanism.
"""

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

PHYSIOLOGY_ORIGINS = {
    "watch": frozenset({"HR", "IBI", "HRV", "EDA", "PPG", "SKIN_TEMPERATURE"}),
    "h10": frozenset({"HR", "RR", "HRV", "ECG"}),
    "muse": frozenset({"EEG", "PPG"}),
}


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    origin: str


@dataclass(frozen=True)
class SourceWindow:
    training_subject_uuid: str
    training_capture_set_uuid: str
    start_ms: int
    end_ms: int
    values: tuple[float | None, ...]
    quality_valid: bool | None
    clock_trusted: bool | None
    lineage_verified: bool | None
    standard_eligibility: str


def _finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)  # type: ignore[arg-type]


def audit_source_windows(
    source: str, features: Sequence[FeatureSpec], windows: Sequence[SourceWindow]
) -> dict[str, object]:
    """Count independently provided source windows without inferring stress truth.

    Durations are union durations inside each subject/capture group, summed across
    groups. Overlap across captures is not interpreted as additional unique person
    time; the receipt explicitly declares that grain.
    """
    if source not in PHYSIOLOGY_ORIGINS:
        raise ValueError("unsupported independent source")
    names = [feature.name for feature in features]
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("feature names must be nonempty and unique")
    if any(feature.origin not in PHYSIOLOGY_ORIGINS[source] for feature in features):
        raise ValueError("predictor origin must be source-specific physiology")
    reasons: Counter[str] = Counter()
    ends: dict[tuple[str, str], int] = {}
    owners: dict[str, str] = {}
    runs: dict[tuple[str, str], int] = {}
    valid_ms = longest_ms = observed_ms = 0
    for row in sorted(windows, key=lambda r: (
        r.training_subject_uuid, r.training_capture_set_uuid, r.start_ms
    )):
        if not row.training_subject_uuid or not row.training_capture_set_uuid:
            raise ValueError("missing pseudonymous grouping")
        if owners.setdefault(row.training_capture_set_uuid, row.training_subject_uuid) != (
            row.training_subject_uuid
        ):
            raise ValueError("capture set has multiple subjects")
        if (type(row.start_ms) is not int or type(row.end_ms) is not int
                or row.start_ms < 0 or row.end_ms <= row.start_ms):
            raise ValueError("invalid corrected-time bounds")
        key = (row.training_subject_uuid, row.training_capture_set_uuid)
        previous_end = ends.get(key)
        if previous_end is not None and row.start_ms < previous_end:
            raise ValueError("duplicate or overlapping source windows")
        ends[key] = row.end_ms
        duration = row.end_ms - row.start_ms
        observed_ms += duration
        failures = []
        if row.quality_valid is not True:
            failures.append("QUALITY_UNVERIFIED")
        if row.clock_trusted is not True:
            failures.append("CLOCK_UNTRUSTED")
        if row.lineage_verified is not True:
            failures.append("LINEAGE_UNVERIFIED")
        if row.standard_eligibility != "ELIGIBLE_NO_PATTERN":
            failures.append("STANDARD_ELIGIBILITY_UNVERIFIED")
        if len(row.values) != len(features) or not all(_finite(v) for v in row.values):
            failures.append("FEATURE_INVALID")
        if failures:
            reasons.update(failures)
            runs[key] = 0
            continue
        valid_ms += duration
        runs[key] = (runs.get(key, 0) if previous_end == row.start_ms else 0) + duration
        longest_ms = max(longest_ms, runs[key])
    return {
        "contract_status": "LOCAL_AUDIT_PREPARATION",
        "source": source,
        "status": "SOURCE_AUDIT_ONLY" if windows else "NO_SOURCE_WINDOWS",
        "duration_grain": "SUM_OF_SUBJECT_CAPTURE_DURATIONS",
        "observed_seconds": observed_ms / 1000,
        "valid_seconds": valid_ms / 1000,
        "longest_valid_run_seconds": longest_ms / 1000,
        "excluded_reasons": dict(sorted(reasons.items())),
        "reason_count_unit": "WINDOWS_OVERLAPPING_PREDICATES",
        "output_semantics": "PERSONAL_PHYSIOLOGY_DEVIATION",
        "minimum_training_seconds": None,
        "training_status": "BLOCKED_NO_REAL_COHORT",
        "training_ready": False,
        "fit_call_count": 0,
    }


def validate_timesfm25_context(
    timestamps_ms: Sequence[int], values: Sequence[float], *, grid_ms: int
) -> tuple[float, ...]:
    """Pre-execution check only; prevents upstream implicit NaN interpolation."""
    if type(grid_ms) is not int or grid_ms <= 0:
        raise ValueError("invalid grid")
    if not values or len(values) != len(timestamps_ms) or len(values) > 16384:
        raise ValueError("invalid context length")
    if not all(type(t) is int and t >= 0 for t in timestamps_ms):
        raise ValueError("invalid timestamps")
    if any(b - a != grid_ms for a, b in pairwise(timestamps_ms)):
        raise ValueError("irregular grid; interpolation is forbidden")
    if not all(_finite(value) for value in values):
        raise ValueError("nonfinite context; interpolation is forbidden")
    return tuple(float(value) for value in values)
