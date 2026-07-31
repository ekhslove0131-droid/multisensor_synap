from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

PACKAGE_SCHEMA = "goal1.5/kaggle-hierarchical-model-package/v1"
MODEL_STATUS = "oracle/sanity"
REAL_DATA_STATUS = "NOT VERIFIED"
DEVICE_SYNCHRONIZATION_STATUS = "NOT_AVAILABLE_TRUTH_ONLY"
FRAMEWORK = "scikitLearn"

BEHAVIOR_CODES: tuple[str, ...] = (
    "ear_covering",
    "exit_attempt",
    "head_turn_away",
    "motion_freeze",
    "movement_reduction",
    "repetitive_body_movement",
    "repetitive_hand_movement",
    "repetitive_object_contact",
    "sustained_pressure_or_contact",
    "withdrawal_movement",
)


class KaggleModelPackageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["goal1.5/kaggle-model-package-config/v1"]
    project_root: Path
    series_id: Literal["mvp3-oracle-v1"]
    output_root: Path
    owner_slug: Literal["bjcoding"]
    model_slug: Literal["multisensor-goal15-hierarchical"]
    framework: Literal["scikitLearn"]
    variation_slug: Literal["oracle-sanity-v1"]
    source_dataset_handle: Literal["bjcoding/multisensor-goal15-oracle-mvp3"]
    source_dataset_version: int = Field(ge=1)
    license_name: Literal["Apache 2.0"]
    notebook_slug: Literal["multisensor-goal15-hierarchical-reproduction"]
    is_private: Literal[True]
    run_training: Literal[False]
    run_locked_test: Literal[False]
    use_gpu: Literal[False]
    sample_split_role: Literal["train", "validation"]
    sample_rows: int = Field(ge=60, le=3600)


def load_kaggle_model_package_config(path: Path) -> KaggleModelPackageConfig:
    source = path.resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Kaggle model package config must be a mapping")
    config = KaggleModelPackageConfig.model_validate(payload)
    repository_root = source.parent.parent
    project_root = (
        config.project_root
        if config.project_root.is_absolute()
        else repository_root / config.project_root
    ).resolve()
    output_root = (
        config.output_root
        if config.output_root.is_absolute()
        else project_root / config.output_root
    ).resolve()
    return config.model_copy(
        update={"project_root": project_root, "output_root": output_root}
    )
