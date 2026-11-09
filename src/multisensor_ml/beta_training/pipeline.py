"""Frozen beta candidate matrix and bounded training orchestration."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml
from synapbridge_beta_runtime.codec import canonical_bytes

from .samples import SOURCE_SETS


@dataclass(frozen=True, slots=True)
class BetaTrainConfig:
    source_sets: tuple[tuple[str, ...], ...] = SOURCE_SETS
    heads: tuple[str, ...] = ("current", "onset60")
    algorithms: tuple[str, ...] = ("LR", "HGB")
    thresholds: tuple[float, ...] = tuple(index / 20 for index in range(1, 20))
    current_sensitivity_gate: float = 0.70
    onset_sensitivity_gate: float = 0.60
    far_gate_per_hour: float = 1.0
    d_on_grid: tuple[float, ...] = (1.0, 1.5, 2.0)
    locked_bootstrap_replicates: int = 1000
    allowed_use: str = "BETA_ONLY"
    model_role: str = "BETA_SELECTED"
    random_state: int = 4242

    def __post_init__(self) -> None:
        allowed = {"watch", "h10", "muse"}
        if (self.source_sets != SOURCE_SETS
                or self.heads != ("current", "onset60")
                or self.algorithms != ("LR", "HGB")
                or self.thresholds != tuple(index / 20 for index in range(1, 20))
                or self.current_sensitivity_gate != 0.70
                or self.onset_sensitivity_gate != 0.60
                or self.far_gate_per_hour != 1.0
                or self.d_on_grid != (1.0, 1.5, 2.0)
                or self.locked_bootstrap_replicates != 1000
                or self.allowed_use != "BETA_ONLY"
                or self.model_role != "BETA_SELECTED"
                or any(not set(source_set) <= allowed for source_set in self.source_sets)
                or self.random_state != 4242):
            raise ValueError("beta_train_config_invalid")

    @classmethod
    def fixture(cls) -> BetaTrainConfig:
        return cls()

    @classmethod
    def load(cls, path: Path) -> BetaTrainConfig:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        required = {"schema_version", "realm", "random_state", "algorithms", "heads",
                    "source_sets", "threshold_grid", "current_sensitivity_gate",
                    "onset_sensitivity_gate", "far_gate_per_hour", "d_on_grid",
                    "locked_bootstrap_replicates", "allowed_use", "model_role"}
        if (not isinstance(value, dict) or set(value) != required
                or value["schema_version"] != "synthetic_beta_train_config_v1"
                or value["realm"] != "SYNTHETIC_BETA"):
            raise ValueError("beta_train_config_invalid")
        return cls(
            source_sets=tuple(tuple(item) for item in value["source_sets"]),
            heads=tuple(value["heads"]), algorithms=tuple(value["algorithms"]),
            thresholds=tuple(float(item) for item in value["threshold_grid"]),
            current_sensitivity_gate=float(value["current_sensitivity_gate"]),
            onset_sensitivity_gate=float(value["onset_sensitivity_gate"]),
            far_gate_per_hour=float(value["far_gate_per_hour"]),
            d_on_grid=tuple(float(item) for item in value["d_on_grid"]),
            locked_bootstrap_replicates=int(value["locked_bootstrap_replicates"]),
            allowed_use=str(value["allowed_use"]), model_role=str(value["model_role"]),
            random_state=int(value["random_state"]),
        )

    def sha256(self) -> str:
        return hashlib.sha256(canonical_bytes({
            "source_sets": self.source_sets, "heads": self.heads,
            "algorithms": self.algorithms, "thresholds": self.thresholds,
            "current_sensitivity_gate": self.current_sensitivity_gate,
            "onset_sensitivity_gate": self.onset_sensitivity_gate,
            "far_gate_per_hour": self.far_gate_per_hour,
            "d_on_grid": self.d_on_grid,
            "locked_bootstrap_replicates": self.locked_bootstrap_replicates,
            "allowed_use": self.allowed_use, "model_role": self.model_role,
            "random_state": self.random_state,
        })).hexdigest()


def expected_fit_attempts(config: BetaTrainConfig) -> int:
    return len(config.source_sets) * len(config.heads) * len(config.algorithms)


def bundle_model_role(status: str, selected_count: int) -> str | None:
    if status == "BETA_SELECTED" and selected_count == 14:
        return "BETA_SELECTED"
    if status == "NO_QUALIFIED_BETA_MODEL" and 0 <= selected_count < 14:
        return None
    raise ValueError("beta_bundle_status_invalid")
