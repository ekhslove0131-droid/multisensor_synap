from __future__ import annotations

import gzip
import json
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import numpy as np
import pandas as pd
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
KOREAN_DOCUMENTS = (
    "MODEL_CARD_KO.md",
    "KAGGLE_REPRODUCTION_KO.md",
    "FEATURE_LABEL_GUIDE_KO.md",
    "EXPERIMENT_LESSONS_KO.md",
)


@dataclass(frozen=True, slots=True)
class KaggleModelPackage:
    root: Path
    extracted: Path
    archive: Path
    manifest: Path
    checksums: Path


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


def _refresh_payload_hashes(root: Path) -> dict[str, object]:
    manifest_path = root / "payload_manifest.json"
    manifest = _read_json(manifest_path)
    manifest["files"] = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != manifest_path
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _write_deterministic_archive(source: Path, destination: Path) -> None:
    with (
        destination.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            info = archive.gettarinfo(
                str(path), arcname=str(path.relative_to(source))
            )
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            info.mode = 0o644
            with path.open("rb") as handle:
                archive.addfile(info, handle)


def _extract_verified_archive(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        unsafe = [
            member.name
            for member in members
            if member.name.startswith("/")
            or ".." in Path(member.name).parts
            or not (member.isfile() or member.isdir())
        ]
        if unsafe:
            raise ValueError(f"unsafe archive members: {unsafe}")
        bundle.extractall(destination, members=members, filter="data")


def _select_sample(config: KaggleModelPackageConfig) -> tuple[pd.DataFrame, dict[str, object]]:
    prepared = config.project_root / "data/prepared" / config.series_id
    manifest = _read_json(prepared / "manifest.json")
    people = cast(list[dict[str, object]], manifest["people"])
    candidates = sorted(
        (entry for entry in people if entry["split_role"] == config.sample_split_role),
        key=lambda entry: str(entry["person_key"]),
    )
    if not candidates:
        raise ValueError("sample split contains no people")
    entry = candidates[0]
    frame = pd.read_parquet(prepared / str(entry["path"]))
    if "session_id" not in frame:
        frame["session_id"] = f"dataset-{entry['dataset_id']}"
    event_binary = frame.get("event_binary", pd.Series(0, index=frame.index))
    positive = np.flatnonzero(event_binary.to_numpy() == 1)
    anchor = int(positive[0]) if len(positive) else 0
    start = max(0, anchor - min(300, config.sample_rows // 2))
    if start + config.sample_rows > len(frame):
        start = max(0, len(frame) - config.sample_rows)
    selected = frame.iloc[start : start + config.sample_rows].copy()
    feature_names = [str(value) for value in cast(list[object], manifest["feature_names"])]
    columns = ["person_key", "session_id", "timestamp_utc", *feature_names]
    selected = selected[columns]
    metadata: dict[str, object] = {
        "sample_split_role": config.sample_split_role,
        "sample_rows": len(selected),
        "person_key": str(entry["person_key"]),
        "dataset_id": str(entry["dataset_id"]),
        "anchor_row": anchor,
        "locked_test_read": False,
    }
    return selected, metadata


def build_kaggle_model_package(
    config: KaggleModelPackageConfig, wheel: Path
) -> KaggleModelPackage:
    from multisensor_ml.kaggle_reproduce import (
        compare_expected,
        load_hierarchical_package,
        predict_hierarchical,
    )

    root = config.output_root.resolve()
    if root.exists():
        raise FileExistsError(f"Kaggle model package already exists: {root}")
    root.mkdir(parents=True)
    extracted = root / "extracted"
    collect_hierarchical_payload(config, extracted, wheel)
    sample, sample_metadata = _select_sample(config)
    sample_root = extracted / "sample"
    sample_root.mkdir()
    sample.to_parquet(sample_root / "sample_input.parquet", index=False)
    loaded = load_hierarchical_package(extracted)
    expected = predict_hierarchical(loaded, sample)
    expected.to_parquet(sample_root / "expected_output.parquet", index=False)
    (sample_root / "sample_manifest.json").write_text(
        json.dumps(sample_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _refresh_payload_hashes(extracted)
    verify_payload(extracted)
    comparison = compare_expected(
        predict_hierarchical(load_hierarchical_package(extracted), sample), expected
    )
    if comparison.status != "REPRODUCED":
        raise ValueError("packaged sample did not reproduce")

    archive = root / "model_payload.tar.gz"
    _write_deterministic_archive(extracted, archive)
    manifest_path = root / "model_manifest.json"
    outer_manifest: dict[str, object] = {
        "schema_version": PACKAGE_SCHEMA,
        "status": MODEL_STATUS,
        "real_data_status": REAL_DATA_STATUS,
        "release_status": "candidate",
        "archive_path": archive.name,
        "archive_sha256": sha256_file(archive),
        "archive_size_bytes": archive.stat().st_size,
        "source_dataset_handle": config.source_dataset_handle,
        "source_dataset_version": config.source_dataset_version,
        "reproduction_status": comparison.status,
        **sample_metadata,
    }
    manifest_path.write_text(
        json.dumps(outer_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    document_root = config.project_root / "docs/kaggle_model"
    if not document_root.is_dir():
        document_root = Path.cwd() / "docs/kaggle_model"
    document_paths: list[Path] = []
    for name in KOREAN_DOCUMENTS:
        destination = root / name
        _copy(document_root / name, destination)
        document_paths.append(destination)
    model_metadata = root / "model-metadata.json"
    model_metadata.write_text(
        json.dumps(
            {
                "ownerSlug": config.owner_slug,
                "title": "Multisensor Goal 1.5 Hierarchical Synthetic Model",
                "slug": config.model_slug,
                "subtitle": "개인 기준선, STD-A, 사건, 5단계, 행동 10개 합성 모델",
                "isPrivate": True,
                "description": (
                    "기존 Phase 1 계층형 모델의 CPU 재현 패키지입니다. "
                    "합성 oracle/sanity 후보이며 실제 데이터 성능은 NOT VERIFIED입니다."
                ),
                "publishTime": "",
                "provenanceSources": "",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    instance_metadata = root / "model-instance-metadata.json"
    instance_metadata.write_text(
        json.dumps(
            {
                "ownerSlug": config.owner_slug,
                "modelSlug": config.model_slug,
                "instanceSlug": config.variation_slug,
                "framework": config.framework,
                "overview": (
                    "개인 기준선과 STD-A를 거쳐 사건, 5단계, 행동 10개를 "
                    "순차 추론하는 scikit-learn 모델 모음"
                ),
                "usage": (
                    "KAGGLE_REPRODUCTION_KO.md에 따라 validation 샘플의 "
                    "expected output을 CPU에서 재현합니다. 새 학습은 수행하지 않습니다."
                ),
                "fineTunable": False,
                "licenseName": config.license_name,
                "trainingData": [],
                "modelInstanceType": "Unspecified",
                "baseModelInstanceId": 0,
                "externalBaseModelUrl": "",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    checksums = root / "SHA256SUMS"
    checksums.write_text(
        "".join(
            f"{sha256_file(path)}  {path.name}\n"
            for path in (
                archive,
                manifest_path,
                *document_paths,
            )
        ),
        encoding="utf-8",
    )
    package = KaggleModelPackage(root, extracted, archive, manifest_path, checksums)
    verify_kaggle_model_package(root)
    return package


def verify_kaggle_model_package(package_root: Path) -> dict[str, object]:
    from multisensor_ml.kaggle_reproduce import (
        compare_expected,
        load_hierarchical_package,
        predict_hierarchical,
    )

    root = package_root.resolve()
    manifest = _read_json(root / "model_manifest.json")
    archive = root / str(manifest["archive_path"])
    with TemporaryDirectory(prefix="multisensor-model-verify-") as temporary:
        effective_archive = archive
        expanded = root / "model_payload"
        if not effective_archive.is_file() and expanded.is_dir():
            effective_archive = Path(temporary) / archive.name
            _write_deterministic_archive(expanded, effective_archive)
        if sha256_file(effective_archive) != manifest["archive_sha256"]:
            raise ValueError("outer package hash mismatch: model_payload.tar.gz")
        for line in (root / "SHA256SUMS").read_text(
            encoding="utf-8"
        ).splitlines():
            checksum_expected, name = line.split("  ", maxsplit=1)
            checked_path = root / name
            if name == archive.name and not checked_path.is_file():
                checked_path = effective_archive
            if sha256_file(checked_path) != checksum_expected:
                raise ValueError(f"outer package hash mismatch: {name}")

        extracted = root / "extracted"
        if expanded.is_dir():
            extracted = expanded
        elif not extracted.is_dir():
            extracted = Path(temporary)
            _extract_verified_archive(effective_archive, extracted)
        verify_payload(extracted)
        sample = pd.read_parquet(extracted / "sample/sample_input.parquet")
        expected = pd.read_parquet(extracted / "sample/expected_output.parquet")
        actual = predict_hierarchical(load_hierarchical_package(extracted), sample)
        result = compare_expected(actual, expected)
    if result.status != "REPRODUCED":
        raise ValueError(
            f"sample reproduction failed: {result.first_mismatch_column}"
        )
    manifest["reproduction_status"] = result.status
    return manifest
