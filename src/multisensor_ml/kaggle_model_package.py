from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import cast

import skops.io as sio

from multisensor_ml.kaggle_model_contracts import (
    BEHAVIOR_CODES,
    MODEL_STATUS,
    PACKAGE_SCHEMA,
    REAL_DATA_STATUS,
    KaggleModelPackageConfig,
)
from multisensor_ml.registry import sha256_file

FORBIDDEN_SUFFIXES = {".pkl", ".pickle", ".joblib"}


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return cast(dict[str, object], payload)


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"required package source missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _require_scope(manifest: dict[str, object], source: Path) -> None:
    if manifest.get("status") != MODEL_STATUS:
        raise ValueError(f"source is not oracle/sanity: {source}")
    if manifest.get("real_data_status") != REAL_DATA_STATUS:
        raise ValueError(f"source real-data status is invalid: {source}")


def _copy_model_group(
    source_root: Path,
    destination_root: Path,
    *,
    group: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    manifest = _read_json(source_root / "manifest.json")
    _require_scope(manifest, source_root / "manifest.json")
    _copy(source_root / "manifest.json", destination_root / "manifest.json")
    _copy(source_root / "feature_schema.json", destination_root / "feature_schema.json")
    entries = cast(list[dict[str, object]], manifest.get("models", []))
    copied: list[dict[str, object]] = []
    for entry in entries:
        relative = str(entry["path"])
        source_model = source_root / relative
        destination_model = destination_root / relative
        _copy(source_model, destination_model)
        discovered = sorted(sio.get_untrusted_types(file=source_model))
        declared = sorted(cast(list[str], entry.get("unknown_types", [])))
        if discovered != declared:
            raise ValueError(f"source skops unknown type mismatch: {source_model}")
        copied.append(
            {
                "group": group,
                "head": entry.get("head"),
                "behavior_code": entry.get("behavior_code"),
                "model_name": entry.get("model_name"),
                "path": str(destination_model.relative_to(destination_root.parent)),
                "unknown_types": declared,
            }
        )
    return manifest, copied


def collect_hierarchical_payload(
    config: KaggleModelPackageConfig,
    payload_root: Path,
    wheel: Path,
) -> dict[str, object]:
    root = payload_root.resolve()
    if root.exists():
        raise FileExistsError(f"payload already exists: {root}")
    root.mkdir(parents=True)
    project = config.project_root
    prepared = project / "data/prepared" / config.series_id
    registry = project / "artifacts/registry" / config.series_id

    prepared_manifest = _read_json(prepared / "manifest.json")
    _require_scope(prepared_manifest, prepared / "manifest.json")
    _copy(prepared / "global_baseline.json", root / "baseline/global_baseline.json")
    _copy(
        prepared / "personal_baseline.parquet",
        root / "baseline/personal_baseline.parquet",
    )
    _copy(prepared / "manifest.json", root / "baseline/prepared_manifest.json")

    type_source = registry / "types"
    type_manifest = _read_json(type_source / "manifest.json")
    _require_scope(type_manifest, type_source / "manifest.json")
    for name in (
        "manifest.json",
        "standard_type_model.skops",
        "standard_type_preprocessing.json",
        "person_standard_types.parquet",
        "ood_status.parquet",
    ):
        _copy(type_source / name, root / "standard_type" / name)
    type_model = root / "standard_type/standard_type_model.skops"
    model_entries: list[dict[str, object]] = [
        {
            "group": "standard_type",
            "head": None,
            "behavior_code": None,
            "model_name": "standard_type",
            "path": str(type_model.relative_to(root)),
            "unknown_types": sorted(sio.get_untrusted_types(file=type_model)),
        }
    ]

    stage_manifest, stage_models = _copy_model_group(
        registry / "stage-model", root / "stage_model", group="stage_model"
    )
    behavior_manifest, behavior_models = _copy_model_group(
        registry / "behavior-model", root / "behavior_model", group="behavior_model"
    )
    model_entries.extend(stage_models)
    model_entries.extend(behavior_models)
    selected_behaviors = cast(dict[str, str], behavior_manifest["selected_models"])
    if set(selected_behaviors) != set(BEHAVIOR_CODES):
        raise ValueError("selected behavior model mapping is incomplete")

    _copy(wheel.resolve(), root / "wheel" / wheel.name)
    _copy(project / "uv.lock", root / "uv.lock")
    forbidden = [path for path in root.rglob("*") if path.suffix in FORBIDDEN_SUFFIXES]
    if forbidden:
        raise ValueError(f"forbidden model files: {forbidden}")

    files = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    manifest: dict[str, object] = {
        "schema_version": PACKAGE_SCHEMA,
        "series_id": config.series_id,
        "status": MODEL_STATUS,
        "real_data_status": REAL_DATA_STATUS,
        "candidate": True,
        "locked_test_read": False,
        "selected_event_model": stage_manifest["selected_event_model"],
        "selected_stage_model": stage_manifest["selected_stage_model"],
        "event_threshold": stage_manifest["event_threshold"],
        "selected_behavior_models": selected_behaviors,
        "standard_types": type_manifest["type_names"],
        "selected_k": type_manifest["selected_k"],
        "models": model_entries,
        "files": files,
    }
    (root / "payload_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verify_payload(root)
    return manifest


def verify_payload(payload_root: Path) -> dict[str, object]:
    root = payload_root.resolve()
    manifest = _read_json(root / "payload_manifest.json")
    if manifest.get("schema_version") != PACKAGE_SCHEMA:
        raise ValueError("unsupported payload schema")
    files = cast(dict[str, str], manifest["files"])
    for relative, expected in files.items():
        if sha256_file(root / relative) != expected:
            raise ValueError(f"payload hash mismatch: {relative}")
    for entry in cast(list[dict[str, object]], manifest["models"]):
        path = root / str(entry["path"])
        discovered = sorted(sio.get_untrusted_types(file=path))
        declared = sorted(cast(list[str], entry["unknown_types"]))
        if discovered != declared:
            raise ValueError(f"skops unknown type mismatch: {entry['path']}")
    if any(path.suffix in FORBIDDEN_SUFFIXES for path in root.rglob("*")):
        raise ValueError("forbidden pickle extension in payload")
    return manifest
