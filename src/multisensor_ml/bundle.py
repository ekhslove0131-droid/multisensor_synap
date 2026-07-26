from __future__ import annotations

import importlib.metadata
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import skops.io as sio

from multisensor_ml.models import ProbabilityClassifier
from multisensor_ml.registry import sha256_file

ModelKey = tuple[str, str]


@dataclass(frozen=True, slots=True)
class LoadedBundle:
    root: Path
    manifest: dict[str, object]
    models: dict[ModelKey, ProbabilityClassifier]


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False),
        path,
        compression="zstd",
    )


def write_model_bundle(
    root: Path,
    *,
    models: dict[ModelKey, ProbabilityClassifier],
    feature_names: list[str],
    thresholds: dict[ModelKey, float],
    lineage: dict[str, object],
    global_baseline: pd.DataFrame,
    personal_baseline: pd.DataFrame,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    uv_lock: Path,
    synchronization: pd.DataFrame | None = None,
    pr_curve: pd.DataFrame | None = None,
) -> Path:
    """Write a hash-addressed, non-pickle model bundle."""

    root.mkdir(parents=True, exist_ok=False)
    model_dir = root / "models"
    model_dir.mkdir()
    model_entries: list[dict[str, object]] = []
    for (target, model_name), model in sorted(models.items()):
        relative = Path("models") / f"{target}__{model_name}.skops"
        sio.dump(model, root / relative)
        trusted_types = sorted(sio.get_untrusted_types(file=root / relative))
        model_entries.append(
            {
                "target": target,
                "model_name": model_name,
                "path": str(relative),
                "trusted_types": trusted_types,
                "threshold": thresholds[(target, model_name)],
            }
        )

    (root / "global_baseline.json").write_text(
        json.dumps(global_baseline.to_dict(orient="records"), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    _write_parquet(personal_baseline, root / "personal_baseline.parquet")
    _write_parquet(metrics, root / "metrics.parquet")
    _write_parquet(predictions, root / "predictions.parquet")
    if pr_curve is None:
        pr_curve = pd.DataFrame(
            {
                "target": pd.Series(dtype="str"),
                "model_name": pd.Series(dtype="str"),
                "precision": pd.Series(dtype="float64"),
                "recall": pd.Series(dtype="float64"),
                "threshold": pd.Series(dtype="float64"),
            }
        )
    _write_parquet(pr_curve, root / "pr_curve.parquet")
    if synchronization is None:
        synchronization = pd.DataFrame(
            [
                {
                    "dataset_id": "synthetic_truth_oracle",
                    "session_id": None,
                    "person_key": None,
                    "device_pair": None,
                    "reference_device": "Polar H10",
                    "offset_ms": None,
                    "drift_ppm": None,
                    "jitter_ms": None,
                    "physiological_lag_ms": None,
                    "overlap_sec": None,
                    "correlation": None,
                    "corrected_time_axis": "UTC",
                    "watch_ecg_policy": "calibration_only",
                    "status": "NOT_AVAILABLE_TRUTH_ONLY",
                }
            ]
        )
    _write_parquet(synchronization, root / "synchronization.parquet")
    (root / "real_data_audit.json").write_text(
        json.dumps(
            {
                "real_accuracy": "NOT VERIFIED",
                "device_synchronization": "NOT VERIFIED",
                "oracle_synchronization": "NOT_AVAILABLE_TRUTH_ONLY",
                "cardiac_reference": "Polar H10",
                "watch_ecg_policy": "calibration_only",
                "clock_axis": "corrected UTC",
                "physiological_lag_separate_from_clock_error": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "feature_schema.json").write_text(
        json.dumps(
            {
                "schema_version": "goal1.5/features/v1",
                "features": feature_names,
                "forbidden_persistence": [".pkl", ".pickle", ".joblib"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    versions = {
        package: importlib.metadata.version(package)
        for package in (
            "multisensor-ml",
            "numpy",
            "pandas",
            "pyarrow",
            "scikit-learn",
            "skops",
        )
    }
    (root / "dependency_versions.json").write_text(
        json.dumps(versions, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if uv_lock.exists():
        shutil.copy2(uv_lock, root / "uv.lock")

    files = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "bundle_schema": "goal1.5/oracle-model-bundle/v1",
        "status": "oracle/sanity",
        "real_data_status": "NOT VERIFIED",
        "lineage": lineage,
        "models": model_entries,
        "files": files,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


def load_model_bundle(root: Path) -> LoadedBundle:
    """Verify every hash and skops type declaration before loading any estimator."""

    manifest = cast(
        dict[str, object],
        json.loads((root / "manifest.json").read_text(encoding="utf-8")),
    )
    files = cast(dict[str, str], manifest["files"])
    for relative, expected in files.items():
        actual = sha256_file(root / relative)
        if actual != expected:
            raise ValueError(f"bundle hash mismatch: {relative}")

    loaded: dict[ModelKey, ProbabilityClassifier] = {}
    entries = cast(list[dict[str, object]], manifest["models"])
    for entry in entries:
        path = root / str(entry["path"])
        declared = sorted(cast(list[str], entry["trusted_types"]))
        discovered = sorted(sio.get_untrusted_types(file=path))
        if discovered != declared:
            raise ValueError(f"skops unknown type declaration mismatch: {entry['path']}")
        estimator = sio.load(path, trusted=declared)
        loaded[(str(entry["target"]), str(entry["model_name"]))] = cast(
            ProbabilityClassifier, estimator
        )
    return LoadedBundle(root=root, manifest=manifest, models=loaded)
