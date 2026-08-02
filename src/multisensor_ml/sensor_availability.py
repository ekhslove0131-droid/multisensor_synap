"""Sensor-availability model contracts for the oracle/sanity benchmark.

The profiles are feature-ablation experiments, not claims that a Watch can
measure EEG or ECG.  A future observed-data adapter must emit the same causal
feature names before a profile is allowed to run on real data.
"""

from __future__ import annotations

import gzip
import json
import shutil
import tarfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import skops.io as sio
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from multisensor_ml.contracts import assert_oracle_columns
from multisensor_ml.hierarchical import (
    MODEL_STAGE_CODES,
    decode_stage_sequence,
)
from multisensor_ml.metrics import evaluate_probabilities, select_event_threshold
from multisensor_ml.models import ProbabilityClassifier, select_training_rows
from multisensor_ml.registry import sha256_file

COMMON_FEATURE_PREFIXES: Final[tuple[str, ...]] = (
    "time_",
    "weekday_",
    "context__",
)
COMMON_FEATURES: Final[frozenset[str]] = frozenset(
    {"is_awake", "history_sufficient"}
)


@dataclass(frozen=True, slots=True)
class AvailabilityProfile:
    profile_id: str
    devices: tuple[str, ...]
    factor_prefixes: tuple[str, ...]


AVAILABILITY_PROFILES: Final[dict[str, AvailabilityProfile]] = {
    "watch_only": AvailabilityProfile(
        "watch_only",
        ("Galaxy Watch8",),
        ("autonomic_arousal", "motor_activation", "sensory_context", "recovery_capacity"),
    ),
    "polar_only": AvailabilityProfile(
        "polar_only", ("Polar H10",), ("autonomic_arousal", "recovery_capacity")
    ),
    "muse_only": AvailabilityProfile(
        "muse_only", ("Muse S",), ("cognitive_load", "motor_activation", "sensory_context")
    ),
    "watch_polar": AvailabilityProfile(
        "watch_polar",
        ("Galaxy Watch8", "Polar H10"),
        ("autonomic_arousal", "motor_activation", "sensory_context", "recovery_capacity"),
    ),
    "watch_muse": AvailabilityProfile(
        "watch_muse",
        ("Galaxy Watch8", "Muse S"),
        (
            "autonomic_arousal",
            "motor_activation",
            "cognitive_load",
            "sensory_context",
            "recovery_capacity",
        ),
    ),
    "polar_muse": AvailabilityProfile(
        "polar_muse",
        ("Polar H10", "Muse S"),
        (
            "autonomic_arousal",
            "cognitive_load",
            "motor_activation",
            "sensory_context",
            "recovery_capacity",
        ),
    ),
    "watch_polar_muse": AvailabilityProfile(
        "watch_polar_muse",
        ("Galaxy Watch8", "Polar H10", "Muse S"),
        (
            "autonomic_arousal",
            "motor_activation",
            "cognitive_load",
            "sleep_pressure",
            "sensory_context",
            "recovery_capacity",
            "social_context",
        ),
    ),
}


def availability_feature_names(columns: list[str], profile_id: str) -> list[str]:
    """Select causal feature columns for one availability profile."""

    try:
        profile = AVAILABILITY_PROFILES[profile_id]
    except KeyError as error:
        raise ValueError(f"unknown availability profile: {profile_id}") from error
    assert_oracle_columns(columns)
    selected = [
        column
        for column in columns
        if column in COMMON_FEATURES
        or column.startswith(COMMON_FEATURE_PREFIXES)
        or column.startswith(profile.factor_prefixes)
    ]
    if not selected:
        raise ValueError(f"availability profile has no usable features: {profile_id}")
    return selected


def sensor_availability_manifest(
    *, profiles: dict[str, dict[str, object]], source_dataset_hash: str
) -> dict[str, object]:
    """Build the immutable outer manifest shared by local and Kaggle bundles."""

    if len(source_dataset_hash) != 64:
        raise ValueError("source_dataset_hash must be a SHA-256 digest")
    unknown = sorted(set(profiles).difference(AVAILABILITY_PROFILES))
    if unknown:
        raise ValueError(f"unknown availability profiles: {unknown}")
    return {
        "schema_version": "goal1.5/sensor-availability-model-package/v1",
        "status": "oracle/sanity",
        "real_data_status": "NOT VERIFIED",
        "device_synchronization_status": "NOT_AVAILABLE_TRUTH_ONLY",
        "locked_test_read": False,
        "source_dataset_hash": source_dataset_hash,
        "profiles": profiles,
    }


@dataclass(frozen=True, slots=True)
class AvailabilityVariantArtifacts:
    root: Path
    manifest_json: Path
    selected_event_model: str
    selected_stage_model: str
    event_threshold: float


@dataclass(frozen=True, slots=True)
class LoadedAvailabilityVariant:
    root: Path
    manifest: dict[str, object]
    event_model: ProbabilityClassifier
    stage_model: ProbabilityClassifier
    event_threshold: float
    feature_names: tuple[str, ...]


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return cast(dict[str, object], payload)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path, compression="zstd")


def _dump_models(root: Path, prefix: str, models: dict[str, object]) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for name, model in sorted(models.items()):
        relative = f"{prefix}__{name}.skops"
        sio.dump(model, root / relative)
        entries.append(
            {
                "head": prefix,
                "model_name": name,
                "path": relative,
                "unknown_types": sorted(sio.get_untrusted_types(file=root / relative)),
            }
        )
    return entries


def _ensure_logistic_compatibility(model: object) -> ProbabilityClassifier:
    """Restore a removed sklearn attribute for cross-minor-version inference."""

    if isinstance(model, LogisticRegression) and not hasattr(model, "multi_class"):
        # scikit-learn 1.9 removed the constructor attribute while 1.6 still
        # reads it during predict_proba.  ``auto`` is the historical default.
        model.multi_class = "auto"
    return cast(ProbabilityClassifier, model)


def fit_availability_variant(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    profile_id: str,
    output_root: Path,
    random_state: int = 20260725,
) -> AvailabilityVariantArtifacts:
    """Fit validation-selected event and five-stage heads for one sensor profile."""

    root = output_root.resolve()
    if root.exists():
        raise FileExistsError(f"availability artifact already exists: {root}")
    feature_names = availability_feature_names(
        [str(column) for column in train.columns], profile_id
    )
    train_event = train.assign(
        event_gate=(train["stage_code"].astype(str) != "NO_EVENT").astype(np.int8)
    )
    selected_rows = select_training_rows(train_event, target="event_gate")
    event_model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=random_state, solver="lbfgs"
    )
    event_model.fit(
        train_event.loc[selected_rows, feature_names].to_numpy(dtype=np.float32),
        train_event.loc[selected_rows, "event_gate"].to_numpy(dtype=np.int8),
    )
    validation_matrix = validation[feature_names].to_numpy(dtype=np.float32)
    validation_truth = (validation["stage_code"].astype(str) != "NO_EVENT").astype(np.int8)
    validation_truth_array = cast(NDArray[np.int8], validation_truth.to_numpy(dtype=np.int8))
    event_probability = event_model.predict_proba(validation_matrix)[:, 1].astype(np.float64)
    threshold = select_event_threshold(
        validation_truth_array,
        event_probability,
        duration_hours=max(len(validation) / 3600, 1 / 3600),
    )
    event_metrics = evaluate_probabilities(
        validation_truth_array,
        event_probability,
        threshold=threshold,
        duration_hours=max(len(validation) / 3600, 1 / 3600),
    )
    train_stage = train.loc[train["stage_code"].astype(str) != "NO_EVENT"]
    validation_stage = validation.loc[validation["stage_code"].astype(str) != "NO_EVENT"]
    stage_to_index = {stage: index for index, stage in enumerate(MODEL_STAGE_CODES)}
    stage_model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=random_state, solver="lbfgs"
    )
    stage_model.fit(
        train_stage[feature_names].to_numpy(dtype=np.float32),
        train_stage["stage_code"].astype(str).map(stage_to_index).to_numpy(dtype=np.int8),
    )
    stage_probability = stage_model.predict_proba(
        validation_stage[feature_names].to_numpy(dtype=np.float32)
    )
    stage_truth = validation_stage["stage_code"].astype(str).map(stage_to_index).to_numpy(
        dtype=np.int8
    )
    stage_macro_f1 = float(
        f1_score(
            stage_truth,
            np.argmax(stage_probability, axis=1),
            labels=list(range(len(MODEL_STAGE_CODES))),
            average="macro",
            zero_division=0,
        )
    )
    metrics = pd.DataFrame(
        [
            {
                "head": "event",
                "model_name": "logistic_regression",
                "threshold": threshold,
                **{
                    key: value
                    for key, value in event_metrics.items()
                    if isinstance(value, int | float)
                },
            },
            {
                "head": "stage",
                "model_name": "logistic_regression",
                "threshold": np.nan,
                "macro_f1": stage_macro_f1,
            },
        ]
    )
    root.mkdir(parents=True)
    model_entries = [
        *_dump_models(root, "event", {"logistic_regression": event_model}),
        *_dump_models(root, "stage", {"logistic_regression": stage_model}),
    ]
    _write_parquet(metrics, root / "validation_metrics.parquet")
    feature_schema = {
        "schema_version": "goal1.5/sensor-availability-features/v1",
        "profile_id": profile_id,
        "devices": list(AVAILABILITY_PROFILES[profile_id].devices),
        "features": feature_names,
        "source": "oracle_latent_feature_ablation",
        "forbidden": [
            "active_target_*",
            "event_id",
            "event_intensity_truth",
            "hard_negative_id",
            "intensity_truth",
            "participant_truth_*",
            "stage_code",
        ],
    }
    (root / "feature_schema.json").write_text(
        json.dumps(feature_schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest: dict[str, object] = {
        "schema_version": "goal1.5/sensor-availability-variant/v1",
        "profile_id": profile_id,
        "devices": list(AVAILABILITY_PROFILES[profile_id].devices),
        "status": "oracle/sanity",
        "real_data_status": "NOT VERIFIED",
        "locked_test_read": False,
        "selected_event_model": "logistic_regression",
        "selected_stage_model": "logistic_regression",
        "event_threshold": threshold,
        "models": model_entries,
        "files": {},
    }
    manifest["files"] = {
        path.name: sha256_file(path)
        for path in sorted(root.iterdir())
        if path.is_file()
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return AvailabilityVariantArtifacts(
        root=root,
        manifest_json=manifest_path,
        selected_event_model="logistic_regression",
        selected_stage_model="logistic_regression",
        event_threshold=threshold,
    )


def load_availability_variant(root: Path) -> LoadedAvailabilityVariant:
    """Safely load a variant and inspect every ``.skops`` unknown type list."""

    resolved = root.resolve()
    manifest = _read_json(resolved / "manifest.json")
    if manifest.get("status") != "oracle/sanity":
        raise ValueError("availability variant is outside oracle/sanity scope")
    schema = _read_json(resolved / "feature_schema.json")
    entries = cast(list[dict[str, object]], manifest["models"])

    def load(head: str, model_name: str) -> ProbabilityClassifier:
        entry = next(
            value
            for value in entries
            if value["head"] == head and value["model_name"] == model_name
        )
        model_path = resolved / str(entry["path"])
        discovered = sorted(sio.get_untrusted_types(file=model_path))
        declared = sorted(cast(list[str], entry["unknown_types"]))
        if discovered != declared:
            raise ValueError(f"skops unknown type mismatch: {model_path}")
        return _ensure_logistic_compatibility(sio.load(model_path, trusted=declared))

    return LoadedAvailabilityVariant(
        root=resolved,
        manifest=manifest,
        event_model=load("event", str(manifest["selected_event_model"])),
        stage_model=load("stage", str(manifest["selected_stage_model"])),
        event_threshold=float(cast(float | int | str, manifest["event_threshold"])),
        feature_names=tuple(str(value) for value in cast(list[object], schema["features"])),
    )


def predict_availability_variant(
    package: LoadedAvailabilityVariant, frame: pd.DataFrame
) -> pd.DataFrame:
    """Predict event and causally decoded stages for one availability profile."""

    required = {"person_key", "session_id", "timestamp_utc", *package.feature_names}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"availability input missing columns: {missing}")
    if frame.duplicated(["person_key", "session_id", "timestamp_utc"]).any():
        raise ValueError("duplicate availability inference identity")
    ordered = frame.sort_values(
        ["person_key", "session_id", "timestamp_utc"], kind="stable"
    ).reset_index(drop=True)
    matrix = ordered[list(package.feature_names)].to_numpy(dtype=np.float32)
    event_probability = package.event_model.predict_proba(matrix)[:, 1]
    stage_probability = package.stage_model.predict_proba(matrix)
    if stage_probability.shape[1] != len(MODEL_STAGE_CODES):
        raise ValueError("availability stage model must expose five classes")
    decoded: list[str] = []
    for _, group in ordered.groupby(["person_key", "session_id"], sort=False).groups.items():
        positions = np.asarray(list(group), dtype=int)
        decoded.extend(
            decode_stage_sequence(
                np.asarray(event_probability[positions], dtype=np.float64),
                np.asarray(stage_probability[positions], dtype=np.float64),
                event_threshold=package.event_threshold,
            )
        )
    output = ordered[["person_key", "session_id", "timestamp_utc"]].copy()
    output["event_probability"] = event_probability
    output["predicted_event"] = event_probability >= package.event_threshold
    output["predicted_stage"] = decoded
    return output


def _sample_person_frame(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame
    # Keep every event/hard-negative row and deterministically thin ordinary
    # baseline rows.  The model never sees a random, unrecorded sample.
    important = frame["stage_code"].astype(str).ne("NO_EVENT") | frame[
        "hard_negative"
    ].astype(bool)
    important_rows = frame.loc[important]
    remaining = max(max_rows - len(important_rows), 0)
    baseline = frame.loc[~important]
    if remaining == 0:
        return important_rows.iloc[:max_rows]
    sampled = baseline.iloc[
        np.linspace(0, len(baseline) - 1, num=min(remaining, len(baseline)), dtype=int)
    ]
    return pd.concat([important_rows, sampled], ignore_index=True).sort_values(
        ["person_key", "session_id", "timestamp_utc"], kind="stable"
    )


def load_synthetic_availability_frames(
    project_root: Path,
    series_id: str,
    *,
    max_rows_per_person: int = 20_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only train/validation persons and attach independent stage labels."""

    prepared_root = project_root / "data" / "prepared" / series_id
    outcome_root = project_root / "data" / "outcomes" / series_id
    prepared_manifest = _read_json(prepared_root / "manifest.json")
    all_features = [str(value) for value in cast(list[object], prepared_manifest["feature_names"])]
    assert_oracle_columns(all_features)
    stages = pq.read_table(outcome_root / "outcome_stages.parquet").to_pandas()
    people = cast(list[dict[str, object]], prepared_manifest["people"])
    split_frames: dict[str, list[pd.DataFrame]] = {"train": [], "validation": []}
    for entry in people:
        role = str(entry["split_role"])
        if role not in split_frames:
            continue
        columns = [
            "person_key",
            "run_id",
            "person_id",
            "timestamp_utc",
            "context",
            "hard_negative",
            *all_features,
        ]
        frame = pd.read_parquet(prepared_root / str(entry["path"]), columns=columns)
        if "session_id" not in frame:
            frame["session_id"] = frame["run_id"].astype(str)
        labels = stages.loc[
            (stages["run_id"].astype(str) == str(entry["run_id"]))
            & (stages["person_id"].astype(str) == str(entry["person_id"]))
        ][["run_id", "person_id", "timestamp_utc", "stage_code"]]
        frame = frame.merge(
            labels,
            on=["run_id", "person_id", "timestamp_utc"],
            how="left",
            validate="one_to_one",
        )
        if frame["stage_code"].isna().any():
            raise ValueError(f"stage labels do not cover {entry['person_key']}")
        split_frames[role].append(_sample_person_frame(frame, max_rows_per_person))
    if not split_frames["train"] or not split_frames["validation"]:
        raise ValueError("availability benchmark requires train and validation people")
    return (
        pd.concat(split_frames["train"], ignore_index=True),
        pd.concat(split_frames["validation"], ignore_index=True),
    )


def _write_deterministic_archive(source: Path, destination: Path) -> None:
    with (
        destination.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            info = archive.gettarinfo(str(path), arcname=str(path.relative_to(source)))
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            info.mode = 0o644
            with path.open("rb") as handle:
                archive.addfile(info, handle)


def build_sensor_availability_package(
    project_root: Path,
    *,
    series_id: str,
    output_root: Path,
    profile_ids: Sequence[str] = tuple(AVAILABILITY_PROFILES),
    max_rows_per_person: int = 20_000,
    wheel_path: Path | None = None,
) -> Path:
    """Build a deterministic multi-profile scikit-learn Kaggle Model package."""

    root = output_root.resolve()
    if root.exists():
        raise FileExistsError(f"availability package already exists: {root}")
    train, validation = load_synthetic_availability_frames(
        project_root.resolve(), series_id, max_rows_per_person=max_rows_per_person
    )
    source_manifest = project_root / "data" / "prepared" / series_id / "manifest.json"
    source_hash = sha256_file(source_manifest)
    payload = root / "extracted"
    profiles_root = payload / "profiles"
    profiles_root.mkdir(parents=True)
    wheel_files: list[str] = []
    if wheel_path is not None:
        wheel = wheel_path.resolve()
        if not wheel.is_file() or wheel.suffix != ".whl":
            raise FileNotFoundError(f"availability runtime wheel is missing: {wheel}")
        wheelhouse = sorted(wheel.parent.glob("*.whl"))
        if not wheelhouse:
            raise FileNotFoundError(f"availability wheelhouse is empty: {wheel.parent}")
        for wheel_file in wheelhouse:
            destination = payload / "wheel" / wheel_file.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(wheel_file, destination)
            wheel_files.append(wheel_file.name)
    profile_manifests: dict[str, dict[str, object]] = {}
    for profile_id in profile_ids:
        if profile_id not in AVAILABILITY_PROFILES:
            raise ValueError(f"unknown availability profile: {profile_id}")
        profile = fit_availability_variant(
            train,
            validation,
            profile_id=profile_id,
            output_root=profiles_root / profile_id,
        )
        profile_manifests[profile_id] = _read_json(profile.manifest_json)

    # One small validation sample per profile makes the Kaggle readback
    # executable without training or touching locked_test.
    sample_root = payload / "sample"
    sample_root.mkdir()
    sample = validation.sort_values(["person_key", "timestamp_utc"], kind="stable").head(600)
    for profile_id in profile_ids:
        features = availability_feature_names(
            [str(column) for column in sample.columns], profile_id
        )
        sample_columns = [
            "person_key",
            "run_id",
            "person_id",
            "session_id",
            "timestamp_utc",
            *features,
        ]
        sample_frame = sample.copy()
        if "session_id" not in sample_frame:
            sample_frame["session_id"] = sample_frame["run_id"].astype(str)
        sample_frame[sample_columns].to_parquet(
            sample_root / f"{profile_id}__input.parquet", index=False
        )
        predicted = predict_availability_variant(
            load_availability_variant(profiles_root / profile_id),
            sample_frame[sample_columns],
        )
        predicted.to_parquet(sample_root / f"{profile_id}__expected.parquet", index=False)
    payload_manifest = sensor_availability_manifest(
        profiles=profile_manifests, source_dataset_hash=source_hash
    )
    payload_manifest["max_rows_per_person"] = max_rows_per_person
    payload_manifest["sample_rows"] = len(sample)
    payload_manifest["files"] = {
        str(path.relative_to(payload)): sha256_file(path)
        for path in sorted(payload.rglob("*"))
        if path.is_file()
    }
    (payload / "payload_manifest.json").write_text(
        json.dumps(payload_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    archive = root / "model_payload.tar.gz"
    _write_deterministic_archive(payload, archive)
    outer = {
        "schema_version": "goal1.5/sensor-availability-model-package/v1",
        "status": "oracle/sanity",
        "real_data_status": "NOT VERIFIED",
        "device_synchronization_status": "NOT_AVAILABLE_TRUTH_ONLY",
        "release_status": "candidate",
        "archive_path": archive.name,
        "archive_sha256": sha256_file(archive),
        "archive_size_bytes": archive.stat().st_size,
        "source_dataset_id": series_id,
        "source_dataset_manifest_sha256": source_hash,
        "profile_ids": list(profile_ids),
        "wheel_files": wheel_files,
        "locked_test_read": False,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "model_manifest.json").write_text(
        json.dumps(outer, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "model-metadata.json").write_text(
        json.dumps(
            {
                "ownerSlug": "bjcoding",
                "title": "Multisensor Goal 1.5 Sensor Availability Models",
                "slug": "multisensor-goal15-availability",
                "subtitle": "3종 센서의 1종·2종·3종 가용성별 사건·5단계 모델",
                "isPrivate": True,
                "description": (
                    "합성 oracle/sanity feature-ablation 모델입니다. "
                    "실제 데이터 성능은 NOT VERIFIED입니다."
                ),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "model-instance-metadata.json").write_text(
        json.dumps(
            {
                "ownerSlug": "bjcoding",
                "modelSlug": "multisensor-goal15-availability",
                "instanceSlug": "oracle-sanity-v1",
                "framework": "scikitLearn",
                "overview": "센서 가용성 프로파일별 사건 gate와 5단계 상태 decoder",
                "usage": "KAGGLE_AVAILABILITY_MODEL_KO.md의 재현 절차를 사용합니다.",
                "fineTunable": False,
                "licenseName": "Apache 2.0",
                "trainingData": [],
                "modelInstanceType": "Unspecified",
                "baseModelInstanceId": 0,
                "externalBaseModelUrl": "",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "SHA256SUMS").write_text(
        f"{sha256_file(archive)}  {archive.name}\n"
        f"{sha256_file(root / 'model_manifest.json')}  model_manifest.json\n",
        encoding="utf-8",
    )
    return root


def verify_sensor_availability_package(package_root: Path) -> dict[str, object]:
    """Verify hashes, safe model loading, and sample prediction identity."""

    root = package_root.resolve()
    outer = _read_json(root / "model_manifest.json")
    archive = root / str(outer["archive_path"])
    if sha256_file(archive) != outer["archive_sha256"]:
        raise ValueError("availability package archive hash mismatch")
    payload = root / "extracted"
    manifest = _read_json(payload / "payload_manifest.json")
    files = cast(dict[str, str], manifest["files"])
    for relative, expected_hash in files.items():
        if sha256_file(payload / relative) != expected_hash:
            raise ValueError(f"availability payload hash mismatch: {relative}")
    for profile_id in cast(list[str], outer["profile_ids"]):
        profile_root = payload / "profiles" / profile_id
        loaded = load_availability_variant(profile_root)
        sample = pd.read_parquet(payload / "sample" / f"{profile_id}__input.parquet")
        expected_frame = pd.read_parquet(
            payload / "sample" / f"{profile_id}__expected.parquet"
        )
        actual = predict_availability_variant(loaded, sample)
        if list(actual.columns) != list(expected_frame.columns) or not actual.equals(
            expected_frame
        ):
            numeric = actual.select_dtypes(include="number").columns
            if not np.allclose(
                actual[numeric].to_numpy(),
                expected_frame[numeric].to_numpy(),
                atol=1e-6,
            ):
                raise ValueError(f"availability sample reproduction failed: {profile_id}")
    forbidden = [
        str(path)
        for path in root.rglob("*")
        if path.suffix in {".pkl", ".pickle", ".joblib"}
    ]
    if forbidden:
        raise ValueError(f"forbidden model files: {forbidden}")
    return outer
