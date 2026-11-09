"""Frozen configuration and deterministic synthetic identities."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from uuid import UUID

import yaml

FAMILIES = ("watch", "h10", "muse")


def deterministic_uuid4(*parts: object) -> str:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:16]
    return str(UUID(bytes=digest, version=4))


@dataclass(frozen=True, slots=True)
class BetaConfig:
    seed: int
    subject_count: int
    duration_seconds: int
    prefill_seconds: int
    calibration_seconds: int
    scored_seconds: int
    train_subjects: int
    validation_subjects: int
    test_subjects: int
    events_per_subject: int

    def __post_init__(self) -> None:
        if (self.seed < 0 or self.subject_count < 1 or self.duration_seconds < 30
                or self.prefill_seconds + self.calibration_seconds + self.scored_seconds
                != self.duration_seconds
                or self.train_subjects + self.validation_subjects + self.test_subjects
                != self.subject_count or self.events_per_subject < 1
                or self.duration_seconds % 10):
            raise ValueError("beta_config_invalid")

    @classmethod
    def load(cls, path: Path) -> BetaConfig:
        value = yaml.safe_load(path.read_text())
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("beta_config_invalid")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class BetaSignalState:
    run_uuid: str
    epoch: int
    subject_uuid: str
    source_uuids: Mapping[str, str]
    seed: int
    timeline_origin_ms: int
    subject_index: int
    split: str
    baseline_hr: float
    response_kind: str
    events_ms: tuple[tuple[int, int], ...]

    @staticmethod
    def fixture(seed: int = 4242) -> BetaSignalState:
        subject = deterministic_uuid4(seed, "fixture-subject")
        sources = {family: deterministic_uuid4(seed, subject, family) for family in FAMILIES}
        return BetaSignalState(
            deterministic_uuid4(seed, "fixture-run"), 1, subject,
            MappingProxyType(sources), seed, 0, 0, "TRAIN", 72.0, "RESPONSIVE",
            ((20_000, 40_000),),
        )


@dataclass(frozen=True, slots=True)
class LocalObservedManifest:
    manifest_path: Path
    observed_manifest_path: Path
    truth_path: Path
    manifest_sha256: str
