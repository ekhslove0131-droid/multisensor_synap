"""Feature and source-reference types shared by trainer and runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .types import BetaScope


@dataclass(frozen=True, slots=True)
class FeatureFrame:
    scope: BetaScope
    decision_ms: int
    family: str
    values: Mapping[str, float]
    valid_support_ms: tuple[int, int]
    quality: float
    schema_sha: str


@dataclass(frozen=True, slots=True)
class SourceScore:
    family: str
    signed_z: Mapping[str, float]
    z_rms: float | None
    valid_feature_mask: Mapping[str, bool]
    quality: float
    reason: str | None
