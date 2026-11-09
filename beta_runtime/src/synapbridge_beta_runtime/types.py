"""Strict source types without truth, labels, or cloud references."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from urllib.parse import unquote, urlparse

import numpy as np
import numpy.typing as npt

Float32Array = npt.NDArray[np.float32]
Int64Array = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


def local_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost") or not parsed.path:
        raise ValueError("local_artifact_uri_invalid")
    return Path(unquote(parsed.path))


@dataclass(frozen=True, slots=True)
class BetaScope:
    run_uuid: str
    epoch: int
    subject_uuid: str


@dataclass(frozen=True, slots=True)
class LocalArtifact:
    uri: str
    sha256: str


@dataclass(frozen=True, slots=True)
class SignalSamples:
    values: Float32Array
    time_us: Int64Array
    available_ms: Int64Array
    sequence: Int64Array
    valid: BoolArray
    unit: str
    nominal_fs_hz: float | None


@dataclass(frozen=True, slots=True)
class NativePart:
    scope: BetaScope
    source_uuid: str
    family: str
    signals: Mapping[str, SignalSamples]
    auxiliary: Mapping[str, SignalSamples]
    start_ms: int
    end_ms: int

    @staticmethod
    def freeze_signals(signals: dict[str, SignalSamples]) -> Mapping[str, SignalSamples]:
        return MappingProxyType(dict(signals))

    @staticmethod
    def freeze_auxiliary(auxiliary: Mapping[str, SignalSamples]) -> Mapping[str, SignalSamples]:
        return MappingProxyType(dict(auxiliary))
