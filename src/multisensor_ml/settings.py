from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Goal15Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["goal1.5/config/v1"]
    series_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    generator_config: Path
    seeds: tuple[int, ...] = Field(min_length=1)
    generator_overrides: dict[str, object] = Field(default_factory=dict)
    data_root: Path = Path("data")
    artifact_root: Path = Path("artifacts")
    random_state: int = 20260725
    locked_test_audit_reason: str = Field(min_length=8)


class FactoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["goal1.5/synthetic-factory/v1"]
    series_id: str = Field(min_length=1)
    generator_config: Path
    behavior_ontology: Path
    seeds: tuple[int, ...] = Field(min_length=1)
    generator_overrides: dict[str, object] = Field(default_factory=dict)
    data_root: Path = Path("data")
    outcome_root: Path = Path("data/outcomes")
    random_state: int = 20260725


class TrainingRegistryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["goal1.5/training-registry-config/v1"]
    registry_path: Path
    artifact_root: Path
    outcome_root: Path
    data_root: Path
    series_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    input_receipt: Path
    random_state: int = 20260725
    korean_router_version: str = Field(min_length=1)


def load_goal15_config(path: Path) -> Goal15Config:
    source = path.resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Goal 1.5 config root must be a mapping")
    config = Goal15Config.model_validate(payload)
    base = source.parent
    return config.model_copy(
        update={
            "generator_config": (base / config.generator_config).resolve(),
            "data_root": (base / config.data_root).resolve(),
            "artifact_root": (base / config.artifact_root).resolve(),
        }
    )


def load_factory_config(path: Path) -> FactoryConfig:
    source = path.resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("synthetic factory config root must be a mapping")
    config = FactoryConfig.model_validate(payload)
    base = source.parent
    return config.model_copy(
        update={
            "generator_config": (base / config.generator_config).resolve(),
            "behavior_ontology": (base / config.behavior_ontology).resolve(),
            "data_root": (base / config.data_root).resolve(),
            "outcome_root": (base / config.outcome_root).resolve(),
        }
    )


def load_training_registry_config(path: Path) -> TrainingRegistryConfig:
    source = path.resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("training registry config root must be a mapping")
    config = TrainingRegistryConfig.model_validate(payload)
    base = source.parent
    return config.model_copy(
        update={
            "registry_path": (base / config.registry_path).resolve(),
            "artifact_root": (base / config.artifact_root).resolve(),
            "outcome_root": (base / config.outcome_root).resolve(),
            "data_root": (base / config.data_root).resolve(),
            "input_receipt": (base / config.input_receipt).resolve(),
        }
    )
