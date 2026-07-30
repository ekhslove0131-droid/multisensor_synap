from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

BaselineVariant = Literal[
    "G0",
    "P1-025",
    "P1-050",
    "P1-100",
    "P2-025",
    "P2-050",
    "P2-100",
    "P2+CL",
    "P2+CL-B",
]
PHASE3_CANDIDATE_IDS: tuple[BaselineVariant, ...] = (
    "G0",
    "P1-025",
    "P1-050",
    "P1-100",
    "P2-025",
    "P2-050",
    "P2-100",
    "P2+CL",
    "P2+CL-B",
)
EXPECTED_SPLIT_COUNTS = {"train": 24, "validation": 6, "locked_test": 6}
EXPECTED_WEIGHT_CAPS = (0.0, 0.25, 0.5, 1.0)
EXPECTED_CUMULATIVE_HORIZONS = (1800, 21600, 86400, 259200)


class CumulativeLoadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    horizons_sec: tuple[int, ...] = EXPECTED_CUMULATIVE_HORIZONS
    baseline_influence_cap: float = 0.10

    @model_validator(mode="after")
    def fixed_contract(self) -> CumulativeLoadConfig:
        if self.horizons_sec != EXPECTED_CUMULATIVE_HORIZONS:
            raise ValueError("cumulative load horizons are fixed")
        if self.baseline_influence_cap != 0.10:
            raise ValueError("cumulative baseline influence cap is fixed at 0.10")
        return self


class Phase3Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["goal1.5/phase3-personal-pattern/v1"]
    series_id: str = Field(min_length=1)
    split_counts: dict[str, int]
    warmup_sec: int
    lookback_sec: int
    refresh_sec: int
    weight_caps: tuple[float, ...]
    cumulative_horizons_sec: tuple[int, ...]
    baseline_load_influence_cap: float
    run_locked_test: Literal[False]
    raw_root: Path = Path("data/raw")
    outcome_root: Path = Path("data/outcomes")
    registry_root: Path = Path("data/registry")
    artifact_root: Path = Path("artifacts/phase3")
    random_state: int = 20260730

    @model_validator(mode="after")
    def fixed_experiment_contract(self) -> Phase3Config:
        if self.split_counts != EXPECTED_SPLIT_COUNTS:
            raise ValueError("Phase 3 split counts are fixed at 24/6/6")
        if (self.warmup_sec, self.lookback_sec, self.refresh_sec) != (1800, 21600, 60):
            raise ValueError("Phase 3 baseline timing contract is fixed")
        if self.weight_caps != EXPECTED_WEIGHT_CAPS:
            raise ValueError("Phase 3 weight caps are fixed")
        if self.cumulative_horizons_sec != EXPECTED_CUMULATIVE_HORIZONS:
            raise ValueError("Phase 3 cumulative horizons are fixed")
        if self.baseline_load_influence_cap != 0.10:
            raise ValueError("Phase 3 baseline load influence is fixed at 0.10")
        return self
