"""Immutable numeric model contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .types import LocalArtifact


@dataclass(frozen=True, slots=True)
class SourceStandardModel:
    family: str
    floors: Mapping[str, float]
    feature_order: tuple[str, ...]
    schema_sha: str


@dataclass(frozen=True, slots=True)
class SourceReference:
    family: str
    centers: Mapping[str, float]
    scales: Mapping[str, float]
    feature_order: tuple[str, ...]
    calibration_sha: str
    scored_valid_seconds: int
    support_kind: str = "QUALITY_QUALIFIED_FEATURE_STRIDE"


@dataclass(frozen=True, slots=True)
class PatternScores:
    current: float
    onset60: float


@dataclass(frozen=True, slots=True)
class BetaModelBundle:
    manifest_sha: str
    standard_refs: tuple[str, ...]
    pattern_refs: tuple[str, ...]
    policy_ref: str


@dataclass(frozen=True, slots=True)
class ModelFrame:
    decision_ms: int
    source_set: tuple[str, ...]
    feature_names: tuple[str, ...]
    values: npt.NDArray[np.float32]
    schema_sha: str
    current_deviation: float
    quality: float


@dataclass(frozen=True, slots=True)
class PatternModelArtifact:
    source_set: tuple[str, ...]
    head: str
    model: LocalArtifact
    threshold: float
    feature_names: tuple[str, ...]
    feature_schema_sha: str


@dataclass(frozen=True, slots=True)
class BetaStagePolicy:
    version: str
    d_on: float
    d_off: float
    sha256: str


@dataclass(frozen=True, slots=True)
class LoadedBetaRuntime:
    bundle: BetaModelBundle
    source_references: Mapping[str, SourceReference]
    source_unready_reasons: Mapping[str, str]
    source_schema_shas: Mapping[str, str]
    pattern_models: Mapping[tuple[str, ...], Mapping[str, PatternModelArtifact]]
    policies: Mapping[tuple[str, ...], BetaStagePolicy]
    calibration_bounds: Mapping[str, tuple[int, int]]
    kernel_sha: str
    bundle_sha: str
