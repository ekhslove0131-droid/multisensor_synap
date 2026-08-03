"""Validation-only Optuna tuning and ONNX export for two sensor views."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from multisensor_ml.dual_sensor_onnx import (
    SENSOR_VARIANTS,
    OnnxBundle,
    feature_names_for_variant,
)
from multisensor_ml.models import select_training_rows
from multisensor_ml.tree_benchmark import evaluate_tree_predictions


@dataclass(frozen=True, slots=True)
class DualSensorTuningConfig:
    project_root: Path
    series_id: str
    output_root: Path
    n_trials: int = 12
    seed: int = 20260725
    max_train_rows: int = 600_000
    max_validation_rows: int = 120_000
    n_jobs: int = 1
    locked_test_read: bool = False
    primary_metric: str = "aucpr"

    def __post_init__(self) -> None:
        if self.n_trials < 1:
            raise ValueError("n_trials must be positive")
        if self.max_train_rows < 0 or self.max_validation_rows < 0:
            raise ValueError("row caps must be non-negative")
        if self.n_jobs != 1:
            raise ValueError("n_jobs must remain 1 for deterministic final tuning")
        if self.locked_test_read:
            raise ValueError("locked_test_read must remain false")
        if self.primary_metric != "aucpr":
            raise ValueError("primary_metric must be aucpr")


@dataclass(frozen=True, slots=True)
class LoadedTrainValidation:
    train: pd.DataFrame
    validation: pd.DataFrame
    feature_names: tuple[str, ...]
    manifest: dict[str, object]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cap_rows(frame: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    if cap <= 0 or len(frame) <= cap:
        return frame.reset_index(drop=True)
    positive = frame["event_binary"].astype(bool).to_numpy()
    hard_negative = frame["hard_negative"].astype(bool).to_numpy()
    must_keep = np.flatnonzero(positive | hard_negative)
    if len(must_keep) > cap:
        rng = np.random.default_rng(seed)
        positives = np.flatnonzero(positive)
        negatives = np.flatnonzero(hard_negative & ~positive)
        positive_quota = min(len(positives), max(1, cap * 4 // 5))
        negative_quota = min(len(negatives), cap - positive_quota)
        selected = np.concatenate(
            [
                rng.choice(positives, positive_quota, replace=False),
                rng.choice(negatives, negative_quota, replace=False),
            ]
        )
    else:
        selected = must_keep
    remaining = cap - len(selected)
    if remaining > 0:
        candidates = np.setdiff1d(
            np.arange(len(frame), dtype="int64"), selected, assume_unique=True
        )
        rng = np.random.default_rng(seed + 7919)
        selected = np.concatenate(
            [
                selected,
                rng.choice(candidates, min(remaining, len(candidates)), replace=False),
            ]
        )
    return frame.iloc[np.sort(selected)].reset_index(drop=True)


def load_train_validation(
    project_root: Path,
    series_id: str,
    *,
    max_train_rows: int,
    max_validation_rows: int,
    seed: int = 20260725,
) -> LoadedTrainValidation:
    """Load train and validation files only; never open a locked-test file."""

    root = project_root.resolve() / "data" / "prepared" / series_id
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("prepared manifest must be an object")
    if manifest.get("locked_test_read", False) is not False:
        raise ValueError("prepared manifest is already marked locked_test_read")
    feature_names = tuple(str(value) for value in cast(list[object], manifest["feature_names"]))
    columns = list(
        dict.fromkeys(
            [
                "person_key",
                "timestamp_utc",
                "context",
                "event_binary",
                "hard_negative",
                *feature_names,
            ]
        )
    )
    frames: dict[str, list[pd.DataFrame]] = {"train": [], "validation": []}
    entries = cast(list[dict[str, object]], manifest["people"])
    for offset, entry in enumerate(entries):
        role = str(entry["split_role"])
        if role not in frames:
            continue
        path = root / str(entry["path"])
        frame = pd.read_parquet(path, columns=columns)
        if role == "train":
            frame = frame.loc[
                select_training_rows(frame, target="event_binary", baseline_ratio=3)
            ]
            frame = _cap_rows(frame, max_train_rows // 24 if max_train_rows else 0, seed + offset)
        else:
            frame = _cap_rows(
                frame, max_validation_rows // 6 if max_validation_rows else 0, seed + offset
            )
        frames[role].append(frame.reset_index(drop=True))
    if not frames["train"] or not frames["validation"]:
        raise ValueError("both train and validation people are required")
    train = pd.concat(frames["train"], ignore_index=True)
    validation = pd.concat(frames["validation"], ignore_index=True)
    if set(train["person_key"].astype(str)).intersection(
        set(validation["person_key"].astype(str))
    ):
        raise ValueError("train and validation person keys overlap")
    return LoadedTrainValidation(train, validation, feature_names, manifest)


def load_train_validation_from_frames(
    train: pd.DataFrame, validation: pd.DataFrame, config: object
) -> LoadedTrainValidation:
    """Small pure helper used by tests and local adapter checks."""

    if set(train["person_key"].astype(str)).intersection(
        set(validation["person_key"].astype(str))
    ):
        raise ValueError("train and validation person keys overlap")
    names = tuple(
        str(column)
        for column in train.columns
        if column not in {"person_key", "event_binary"}
    )
    return LoadedTrainValidation(train.copy(), validation.copy(), names, {})


def _trial_params(trial: Any) -> dict[str, object]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 80, 240, step=40),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63, step=8),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 160, step=20),
        "subsample": trial.suggest_float("subsample", 0.75, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _make_lightgbm(params: Mapping[str, object], seed: int) -> Any:
    try:
        from lightgbm import LGBMClassifier
    except ImportError as error:  # pragma: no cover - optional extra guard
        raise RuntimeError("install the tree-benchmark extra for LightGBM") from error
    return LGBMClassifier(
        objective="binary",
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
        device_type="cpu",
        force_col_wise=True,
        subsample_freq=1,
        **cast(dict[str, Any], dict(params)),
    )


def _export_lightgbm_onnx(model: Any, feature_count: int, path: Path) -> None:
    try:
        import onnx
        import onnxmltools  # type: ignore[import-untyped]
        from onnxmltools.convert.common.data_types import (  # type: ignore[import-untyped]
            FloatTensorType,
        )
    except ImportError as error:  # pragma: no cover - optional extra guard
        raise RuntimeError("install the onnx-serving extra to export ONNX") from error
    graph = onnxmltools.convert_lightgbm(
        model,
        initial_types=[("features", FloatTensorType([None, feature_count]))],
        target_opset=15,
        zipmap=False,
    )
    onnx.checker.check_model(graph)
    path.write_bytes(graph.SerializeToString())


def _fit_and_score(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_names: Sequence[str],
    params: Mapping[str, object],
    seed: int,
) -> tuple[Any, dict[str, float]]:
    model = _make_lightgbm(params, seed)
    model.fit(train[list(feature_names)], train["event_binary"].astype("int8"))
    probability = np.asarray(
        model.predict_proba(validation[list(feature_names)])[:, 1], dtype="float64"
    )
    score = evaluate_tree_predictions(
        validation["event_binary"].to_numpy(dtype="int8"),
        probability,
        threshold=0.5,
        timestamps=validation["timestamp_utc"].tolist()
        if "timestamp_utc" in validation
        else None,
        groups=validation["person_key"].tolist()
        if "person_key" in validation
        else None,
    )
    return model, {key: float(value) for key, value in score.items() if isinstance(value, float)}


def tune_sensor_variant(
    loaded: LoadedTrainValidation,
    variant: str,
    config: DualSensorTuningConfig,
) -> dict[str, object]:
    """Tune LightGBM on validation AUCPR and write one ONNX bundle."""

    try:
        import optuna
    except ImportError as error:  # pragma: no cover - optional extra guard
        raise RuntimeError("install the onnx-serving extra for Optuna") from error
    feature_names = feature_names_for_variant(loaded.feature_names, variant)
    train = loaded.train
    validation = loaded.validation

    def objective(trial: Any) -> float:
        params = _trial_params(trial)
        model = _make_lightgbm(params, config.seed)
        model.fit(train[list(feature_names)], train["event_binary"].astype("int8"))
        probability = np.asarray(
            model.predict_proba(validation[list(feature_names)])[:, 1], dtype="float64"
        )
        from sklearn.metrics import average_precision_score

        return float(
            average_precision_score(validation["event_binary"].to_numpy(dtype="int8"), probability)
        )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.seed),
        study_name=f"{config.series_id}-{variant}-aucpr",
    )
    study.optimize(objective, n_trials=config.n_trials, n_jobs=1, show_progress_bar=False)
    best_params = cast(dict[str, object], study.best_trial.params)
    model, metrics = _fit_and_score(train, validation, feature_names, best_params, config.seed)
    output = config.output_root.resolve() / variant
    output.mkdir(parents=True, exist_ok=True)
    onnx_path = output / "model.onnx"
    _export_lightgbm_onnx(model, len(feature_names), onnx_path)
    manifest: dict[str, object] = {
        "schema_version": "goal1.5/onnx-bundle/v1",
        "model_id": f"{config.series_id}-optuna-lightgbm-{variant}",
        "variant": variant,
        "feature_names": list(feature_names),
        "input_name": "features",
        "model_path": "model.onnx",
        "model_scope": "oracle/sanity",
        "real_data_status": "NOT VERIFIED",
        "locked_test_read": False,
        "threshold": 0.5,
        "primary_metric": "aucpr",
        "validation_metrics": metrics,
        "validation_people": 6,
        "validation_rows": len(validation),
        "train_people": 24,
        "train_rows": len(train),
        "source_dataset_id": loaded.manifest.get("dataset_id"),
        "source_manifest_sha256": _sha256(
            config.project_root.resolve()
            / "data"
            / "prepared"
            / config.series_id
            / "manifest.json"
        ),
        "optuna": {
            "n_trials": config.n_trials,
            "best_value_aucpr": float(study.best_value),
            "best_params": best_params,
            "seed": config.seed,
        },
        "onnx": {"opset": 15, "execution_provider": "CPUExecutionProvider"},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "feature_schema.json").write_text(
        json.dumps(
            {
                "variant": variant,
                "feature_names": list(feature_names),
                "source": "synthetic_latent_proxy",
                "note": "Replace with observed device adapter before real deployment.",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "optuna_study.json").write_text(
        json.dumps(
            {
                "study_name": study.study_name,
                "best_value_aucpr": study.best_value,
                "best_params": best_params,
                "trials": [
                    {
                        "number": trial.number,
                        "value": trial.value,
                        "state": str(trial.state),
                        "params": trial.params,
                    }
                    for trial in study.trials
                ],
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    # Reload through the same safe runtime used by Cloud Run and verify one row.
    loaded_bundle = OnnxBundle.load(output)
    loaded_bundle.predict_one(cast(dict[str, object], validation.iloc[0].to_dict()))
    manifest["model_sha256"] = _sha256(onnx_path)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def tune_dual_sensor_models(config: DualSensorTuningConfig) -> dict[str, object]:
    loaded = load_train_validation(
        config.project_root,
        config.series_id,
        max_train_rows=config.max_train_rows,
        max_validation_rows=config.max_validation_rows,
        seed=config.seed,
    )
    model_manifests = {
        variant: tune_sensor_variant(loaded, variant, config)
        for variant in SENSOR_VARIANTS
    }
    root = config.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    bundle_files = {
        variant: {
            name: _sha256(root / variant / name)
            for name in ("model.onnx", "manifest.json", "feature_schema.json", "optuna_study.json")
        }
        for variant in SENSOR_VARIANTS
    }
    summary: dict[str, object] = {
        "schema_version": "goal1.5/dual-sensor-optuna/v1",
        "status": "READY_FOR_SYNTHETIC_SHADOW",
        "series_id": config.series_id,
        "primary_metric": "aucpr",
        "model_scope": "oracle/sanity",
        "real_data_status": "NOT VERIFIED",
        "locked_test_read": False,
        "variants": model_manifests,
        "bundle_files": bundle_files,
        "source_split": {"train_people": 24, "validation_people": 6, "locked_test_read": False},
    }
    (root / "experiment_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
