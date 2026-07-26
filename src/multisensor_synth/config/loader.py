from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from multisensor_synth.config.models import ProjectConfig


class ConfigLoadError(ValueError):
    """Configuration source or inheritance could not be resolved."""


@dataclass(frozen=True)
class LoadedConfig:
    config: ProjectConfig
    source_path: Path
    source_text: str
    source_data: dict[str, object]
    resolved_data: dict[str, object]
    config_sha256: str


def _deep_merge(
    parent: dict[str, object], child: dict[str, object]
) -> dict[str, object]:
    merged = dict(parent)
    for key, value in child.items():
        previous = merged.get(key)
        if isinstance(previous, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(
                cast(dict[str, object], previous),
                cast(dict[str, object], value),
            )
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path, seen: frozenset[Path]) -> tuple[dict[str, object], str]:
    resolved_path = path.resolve()
    if resolved_path in seen:
        raise ConfigLoadError(f"cyclic config extends detected at {resolved_path}")
    source_text = resolved_path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(source_text)
    if not isinstance(parsed, dict):
        raise ConfigLoadError(f"config root must be a mapping: {resolved_path}")
    source_data = cast(dict[str, object], dict(parsed))
    extends = source_data.pop("extends", None)
    if extends is None:
        return source_data, source_text
    if not isinstance(extends, str):
        raise ConfigLoadError("extends must be a relative path string")
    parent_data, _ = _read_yaml(resolved_path.parent / extends, seen | {resolved_path})
    return _deep_merge(parent_data, source_data), source_text


def load_config(path: str | Path) -> LoadedConfig:
    source_path = Path(path).resolve()
    raw_resolved, source_text = _read_yaml(source_path, frozenset())
    config = ProjectConfig.model_validate(raw_resolved)
    resolved_data = config.model_dump(mode="json", by_alias=False)
    canonical = json.dumps(
        resolved_data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    source_parsed = yaml.safe_load(source_text)
    if not isinstance(source_parsed, dict):
        raise ConfigLoadError(f"config root must be a mapping: {source_path}")
    return LoadedConfig(
        config=config,
        source_path=source_path,
        source_text=source_text,
        source_data=cast(dict[str, object], dict(source_parsed)),
        resolved_data=resolved_data,
        config_sha256=hashlib.sha256(canonical).hexdigest(),
    )


def export_json_schema(path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(ProjectConfig.model_json_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
