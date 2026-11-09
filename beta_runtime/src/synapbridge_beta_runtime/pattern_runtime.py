"""Shared exact-source-set assembly for beta training and inference."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence

import numpy as np

from .codec import canonical_bytes
from .feature_catalog import feature_order
from .feature_types import FeatureFrame
from .model_types import ModelFrame, SourceReference
from .standard import combine_deviation, score_reference

SOURCE_ORDER = ("watch", "h10", "muse")


def canonical_source_set(values: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(values)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("source_set_invalid")
    result = tuple(family for family in SOURCE_ORDER if family in requested)
    if len(result) != len(requested):
        raise ValueError("source_set_invalid")
    return result


def assemble_model_frame(
    history: Sequence[FeatureFrame],
    references: Mapping[str, SourceReference],
    source_set: Sequence[str],
) -> ModelFrame | None:
    families = canonical_source_set(source_set)
    if any(family not in references for family in families):
        return None
    scopes = {frame.scope for frame in history if frame.family in families}
    if len(scopes) != 1:
        raise ValueError("model_frame_scope_mismatch")
    by_family = {family: [frame for frame in history if frame.family == family]
                 for family in families}
    if any(not frames for frames in by_family.values()):
        return None
    decision = max(frame.decision_ms for frame in by_family[families[0]])
    current, previous = {}, {}
    for family, frames in by_family.items():
        current[family] = next((frame for frame in frames if frame.decision_ms == decision), None)
        previous[family] = next(
            (frame for frame in frames if frame.decision_ms == decision - 60_000), None
        )
    if any(current[family] is None or previous[family] is None for family in families):
        return None
    names: list[str] = []
    values: list[float] = []
    current_scores = []
    qualities: list[float] = []
    source_schema_shas: dict[str, str] = {}
    for family in families:
        frame = current[family]
        prior = previous[family]
        assert frame is not None and prior is not None
        reference = references[family]
        score = score_reference(frame, reference)
        prior_score = score_reference(prior, reference)
        if score.z_rms is None or prior_score.z_rms is None:
            return None
        expected_features = feature_order(family)
        if tuple(frame.values) != expected_features:
            return None
        for name in expected_features:
            names.append(f"{family}.feature.{name}")
            values.append(float(frame.values[name]))
        for name in reference.feature_order:
            if name not in score.signed_z:
                return None
            names.append(f"{family}.signed_z.{name}")
            values.append(float(score.signed_z[name]))
        names.extend((f"{family}.z_rms", f"{family}.z_rms_delta_60",
                      f"{family}.quality"))
        values.extend((score.z_rms, (score.z_rms - prior_score.z_rms) / 60,
                       frame.quality))
        current_scores.append(score)
        qualities.append(frame.quality)
        source_schema_shas[family] = frame.schema_sha
    numeric = np.asarray(values, dtype=np.float32)
    if not np.isfinite(numeric).all():
        raise ValueError("model_frame_nonfinite")
    schema_body = {"version": "synthetic_beta_model_frame_v1",
                   "source_set": families, "feature_names": names,
                   "source_feature_schema_shas": source_schema_shas}
    deviation = combine_deviation(current_scores)
    if deviation is None or not math.isfinite(deviation):
        return None
    return ModelFrame(
        decision,
        families,
        tuple(names),
        numeric,
        hashlib.sha256(canonical_bytes(schema_body)).hexdigest(),
        deviation,
        min(qualities),
    )
