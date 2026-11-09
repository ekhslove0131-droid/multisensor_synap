"""Raw-free output markers for local beta verification."""

from __future__ import annotations

from dataclasses import dataclass

from .types import BetaScope


@dataclass(frozen=True, slots=True)
class BetaOutput:
    scope: BetaScope
    decision_ms: int
    source_set: tuple[str, ...]
    current_deviation: float | None
    unit: str
    status: str
