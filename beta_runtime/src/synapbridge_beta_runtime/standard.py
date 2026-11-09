"""TRAIN-only source floors and personal calibration references."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Sequence
from types import MappingProxyType

import numpy as np

from .codec import canonical_bytes
from .feature_catalog import standard_feature_order
from .feature_types import FeatureFrame, SourceScore
from .model_types import SourceReference, SourceStandardModel


def fit_standard(train_calibration: Sequence[FeatureFrame], family: str) -> SourceStandardModel:
    frames = _frames(train_calibration, family)
    order = tuple(name for name in standard_feature_order(family)
                  if all(name in frame.values for frame in frames))
    by_subject: dict[str, list[FeatureFrame]] = defaultdict(list)
    for frame in frames:
        by_subject[frame.scope.subject_uuid].append(frame)
    floors: dict[str, float] = {}
    for name in order:
        scales = []
        for subject_frames in by_subject.values():
            values = np.array([frame.values[name] for frame in subject_frames], dtype=float)
            scale = 1.4826 * np.median(np.abs(values - np.median(values)))
            if scale > 0 and math.isfinite(float(scale)):
                scales.append(float(scale))
        if scales:
            floors[name] = float(np.quantile(scales, 0.1, method="linear"))
    selected = tuple(name for name in order if name in floors)
    if len(selected) < 2:
        raise ValueError("standard_features_insufficient")
    return SourceStandardModel(family, MappingProxyType({name: floors[name] for name in selected}),
                               selected, frames[0].schema_sha)


def calibrate_reference(model: SourceStandardModel,
                        personal_calibration: Sequence[FeatureFrame]) -> SourceReference:
    frames = _frames(personal_calibration, model.family)
    if len({frame.scope for frame in frames}) != 1:
        raise ValueError("calibration_scope_mismatch")
    if any(frame.schema_sha != model.schema_sha for frame in frames):
        raise ValueError("calibration_schema_mismatch")
    intervals = sorted(frame.valid_support_ms for frame in frames)
    unique_ms = 0
    previous_end = -1
    for start, end in intervals:
        if start < previous_end:
            raise ValueError("calibration_support_overlap")
        unique_ms += end - start
        previous_end = end
    if unique_ms < 1_800_000:
        raise ValueError("calibration_support_insufficient")
    centers, scales = {}, {}
    for name in model.feature_order:
        values = np.array([frame.values[name] for frame in frames], dtype=float)
        center = float(np.median(values))
        mad = float(1.4826 * np.median(np.abs(values - center)))
        epsilon = max(1e-12, abs(center) * 1e-6)
        centers[name] = center
        scales[name] = max(mad, model.floors[name], epsilon)
    body = {"family": model.family, "centers": centers, "scales": scales,
            "feature_order": model.feature_order, "schema_sha": model.schema_sha,
            "scored_valid_seconds": unique_ms // 1000,
            "support_kind": "QUALITY_QUALIFIED_FEATURE_STRIDE"}
    digest = hashlib.sha256(canonical_bytes(body)).hexdigest()
    return SourceReference(model.family, MappingProxyType(centers), MappingProxyType(scales),
                           model.feature_order, digest, unique_ms // 1000,
                           "QUALITY_QUALIFIED_FEATURE_STRIDE")


def score_reference(frame: FeatureFrame, reference: SourceReference) -> SourceScore:
    if frame.family != reference.family:
        raise ValueError("reference_family_mismatch")
    signed = {name: (frame.values[name] - reference.centers[name]) / reference.scales[name]
              for name in reference.feature_order if name in frame.values}
    mask = {name: name in signed for name in reference.feature_order}
    if len(signed) < 2:
        return SourceScore(frame.family, MappingProxyType(signed), None,
                           MappingProxyType(mask), frame.quality, "FEATURES_INSUFFICIENT")
    rms = math.sqrt(sum(value * value for value in signed.values()) / len(signed))
    return SourceScore(frame.family, MappingProxyType(signed), rms,
                       MappingProxyType(mask), frame.quality, None)


def combine_deviation(scores: Sequence[SourceScore]) -> float | None:
    valid = [score.z_rms for score in scores if score.z_rms is not None]
    return math.sqrt(sum(value * value for value in valid) / len(valid)) if valid else None


def _frames(values: Sequence[FeatureFrame], family: str) -> list[FeatureFrame]:
    frames = list(values)
    if not frames or any(frame.family != family or frame.quality < 0.8 for frame in frames):
        raise ValueError("standard_input_invalid")
    return frames
