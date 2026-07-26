"""Cross-field validation lives on frozen Pydantic models.

This module is a stable import boundary for callers that need to validate a
fully resolved mapping without depending on the loader.
"""

from __future__ import annotations

from multisensor_synth.config.models import ProjectConfig


def validate_resolved_config(payload: object) -> ProjectConfig:
    return ProjectConfig.model_validate(payload)
