from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import skops.io as sio
from sklearn.metrics import f1_score

from multisensor_ml.contracts import assert_oracle_columns
from multisensor_ml.hierarchical import (
    MODEL_STAGE_CODES,
    decode_stage_sequence,
    fit_behavior_candidates,
    grouped_oof_probabilities,
    personal_calibration_status,
)
from multisensor_ml.metrics import evaluate_probabilities, select_event_threshold
from multisensor_ml.models import (
    ProbabilityClassifier,
    fit_candidate_models,
    select_training_rows,
)
from multisensor_ml.outcomes import BEHAVIOR_CODES
from multisensor_ml.registry import sha256_file

STAGE_MODEL_SCHEMA = "goal1.5/hierarchical-stage-model/v1"
BEHAVIOR_MODEL_SCHEMA = "goal1.5/behavior-model/v1"


@dataclass(frozen=True, slots=True)
class StageModelArtifacts:
    root: Path
    manifest_json: Path
    selected_event_model: str
    selected_stage_model: str
    event_threshold: float


@dataclass(frozen=True, slots=True)
class BehaviorModelArtifacts:
    root: Path
    manifest_json: Path


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False),
        path,
        compression="zstd",
    )


def _read_json(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def _membership_context(
    standard_type_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    memberships = pq.read_table(
        standard_type_root / "person_standard_types.parquet"
    ).to_pandas()
    ood = pq.read_table(standard_type_root / "ood_status.parquet").to_pandas()
    type_features = sorted(
        column for column in memberships.columns if column.startswith("STD-")
    )
    return memberships, ood, type_features


def _add_membership(
    frame: pd.DataFrame,
    *,
    person_key: str,
    memberships: pd.DataFrame,
    type_features: list[str],
) -> pd.DataFrame:
    match = memberships.loc[memberships["person_key"].astype(str) == person_key]
    if len(match) != 1:
        raise ValueError(f"missing or duplicate standard-type membership: {person_key}")
    output = frame.copy()
    for feature in type_features:
        output[feature] = np.float32(match.iloc[0][feature])
    return output


def _read_labeled_person(
    prepared_root: Path,
    outcome_root: Path,
    entry: dict[str, object],
    *,
    feature_names: list[str],
    memberships: pd.DataFrame,
    type_features: list[str],
) -> pd.DataFrame:
    columns = [
        "person_key",
        "run_id",
        "person_id",
        "timestamp_utc",
        "context",
        "hard_negative",
        *feature_names,
    ]
    frame = pq.read_table(
        prepared_root / str(entry["path"]),
        columns=columns,
    ).to_pandas()
    stages = pq.read_table(
        outcome_root / "outcome_stages.parquet",
        filters=[
            ("run_id", "=", str(entry["run_id"])),
            ("person_id", "=", str(entry["person_id"])),
        ],
    ).to_pandas()
    joined = frame.merge(
        stages[["run_id", "person_id", "timestamp_utc", "event_id", "stage_code"]],
        on=["run_id", "person_id", "timestamp_utc"],
        how="left",
        validate="one_to_one",
    )
    if joined["stage_code"].isna().any():
        raise ValueError(f"stage labels do not cover person: {entry['person_key']}")
    return _add_membership(
        joined,
        person_key=str(entry["person_key"]),
        memberships=memberships,
        type_features=type_features,
    )


def _aligned_stage_probability(
    model: ProbabilityClassifier,
    features: np.ndarray,
) -> np.ndarray:
    probability = model.predict_proba(features)
    if probability.shape[1] != len(MODEL_STAGE_CODES):
        raise ValueError("conditional stage model does not expose five classes")
    return probability.astype(np.float64)


def _dump_models(
    root: Path,
    *,
    prefix: str,
    models: dict[str, ProbabilityClassifier],
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for name, model in sorted(models.items()):
        relative = f"{prefix}__{name}.skops"
        sio.dump(model, root / relative)
        unknown = sorted(sio.get_untrusted_types(file=root / relative))
        entries.append(
            {
                "head": prefix,
                "model_name": name,
                "path": relative,
                "unknown_types": unknown,
            }
        )
    return entries


def fit_stage_model_artifacts(
    prepared_root: Path,
    outcome_root: Path,
    standard_type_root: Path,
    output_root: Path,
    *,
    random_state: int,
) -> StageModelArtifacts:
    """Fit two validation-selected heads and retain full validation predictions."""

    root = output_root.resolve()
    if root.exists():
        raise FileExistsError(f"stage model artifact root already exists: {root}")
    root.mkdir(parents=True)
    prepared_manifest = _read_json(prepared_root / "manifest.json")
    base_features = [
        str(value)
        for value in cast(list[object], prepared_manifest["feature_names"])
    ]
    assert_oracle_columns(base_features)
    people = cast(list[dict[str, object]], prepared_manifest["people"])
    memberships, ood, type_features = _membership_context(standard_type_root)
    feature_names = [*base_features, *type_features]

    training_parts: list[pd.DataFrame] = []
    train_entries = [entry for entry in people if entry["split_role"] == "train"]
    validation_entries = [
        entry for entry in people if entry["split_role"] == "validation"
    ]
    for entry in train_entries:
        frame = _read_labeled_person(
            prepared_root,
            outcome_root,
            entry,
            feature_names=base_features,
            memberships=memberships,
            type_features=type_features,
        )
        frame["event_gate"] = (
            frame["stage_code"].astype(str) != "NO_EVENT"
        ).astype(np.int8)
        selected = select_training_rows(frame, target="event_gate")
        training_parts.append(frame.loc[selected])
    training = pd.concat(training_parts, ignore_index=True)
    event_models = dict(
        fit_candidate_models(
            training[feature_names].to_numpy(dtype=np.float32),
            training["event_gate"].to_numpy(dtype=np.int8),
            random_state=random_state,
        )
    )
    event_training = training.loc[training["event_gate"].astype(bool)].copy()
    stage_to_index = {stage: index for index, stage in enumerate(MODEL_STAGE_CODES)}
    stage_models = dict(
        fit_candidate_models(
            event_training[feature_names].to_numpy(dtype=np.float32),
            np.asarray(
                event_training["stage_code"].astype(str).map(stage_to_index),
                dtype=np.int8,
            ),
            random_state=random_state,
        )
    )

    validation_people: list[pd.DataFrame] = []
    event_truth_parts: list[np.ndarray] = []
    event_probability_parts: dict[str, list[np.ndarray]] = {
        name: [] for name in event_models
    }
    stage_truth_parts: list[np.ndarray] = []
    stage_probability_parts: dict[str, list[np.ndarray]] = {
        name: [] for name in stage_models
    }
    for entry in validation_entries:
        frame = _read_labeled_person(
            prepared_root,
            outcome_root,
            entry,
            feature_names=base_features,
            memberships=memberships,
            type_features=type_features,
        )
        frame["event_gate"] = (
            frame["stage_code"].astype(str) != "NO_EVENT"
        ).astype(np.int8)
        validation_people.append(frame)
        matrix = frame[feature_names].to_numpy(dtype=np.float32)
        truth = frame["event_gate"].to_numpy(dtype=np.int8)
        event_truth_parts.append(truth)
        for name, model in event_models.items():
            event_probability_parts[name].append(
                model.predict_proba(matrix)[:, 1].astype(np.float64)
            )
        event_mask = truth.astype(bool)
        stage_truth_parts.append(
            frame.loc[event_mask, "stage_code"]
            .astype(str)
            .map(stage_to_index)
            .to_numpy(dtype=np.int8)
        )
        for name, model in stage_models.items():
            stage_probability_parts[name].append(
                _aligned_stage_probability(model, matrix[event_mask])
            )

    event_truth = np.concatenate(event_truth_parts)
    duration_hours = max(len(event_truth) / 3600, 1 / 3600)
    metric_rows: list[dict[str, object]] = []
    event_scores: list[tuple[float, float, float, str, float]] = []
    thresholds: dict[str, float] = {}
    for name, parts in event_probability_parts.items():
        probability = np.concatenate(parts)
        threshold = select_event_threshold(
            event_truth,
            probability,
            duration_hours=duration_hours,
        )
        thresholds[name] = threshold
        metrics = evaluate_probabilities(
            event_truth,
            probability,
            threshold=threshold,
            duration_hours=duration_hours,
        )
        scalar = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, int | float)
        }
        metric_rows.append(
            {
                "head": "event",
                "model_name": name,
                "role": "validation",
                "threshold": threshold,
                **scalar,
            }
        )
        event_scores.append(
            (
                float(cast(float | int, metrics["event_f1"])),
                float(cast(float | int, metrics["aucpr"])),
                -float(cast(float | int, metrics["false_alerts_per_hour"])),
                name,
                threshold,
            )
        )
    _, _, _, selected_event, selected_threshold = max(event_scores)

    stage_truth = np.concatenate(stage_truth_parts)
    stage_scores: list[tuple[float, str]] = []
    for name, parts in stage_probability_parts.items():
        probability = np.concatenate(parts)
        predicted = np.argmax(probability, axis=1)
        macro_f1 = float(
            f1_score(
                stage_truth,
                predicted,
                labels=list(range(len(MODEL_STAGE_CODES))),
                average="macro",
                zero_division=0,
            )
        )
        metric_rows.append(
            {
                "head": "stage",
                "model_name": name,
                "role": "validation",
                "threshold": np.nan,
                "macro_f1": macro_f1,
            }
        )
        stage_scores.append((macro_f1, name))
    _, selected_stage = max(stage_scores)

    prediction_parts: list[pd.DataFrame] = []
    for frame, event_probability in zip(
        validation_people,
        event_probability_parts[selected_event],
        strict=True,
    ):
        matrix = frame[feature_names].to_numpy(dtype=np.float32)
        stage_probability = _aligned_stage_probability(
            stage_models[selected_stage], matrix
        )
        ood_row = ood.loc[
            ood["person_key"].astype(str) == str(frame["person_key"].iloc[0])
        ]
        ood_status = (
            str(ood_row.iloc[0]["status"])
            if len(ood_row) == 1
            else "NOT_DECISIONABLE"
        )
        valid = np.full(
            len(frame),
            ood_status not in {"NOT_DECISIONABLE", "RETRAIN_CANDIDATE"},
            dtype=bool,
        )
        decoded = decode_stage_sequence(
            event_probability,
            stage_probability,
            event_threshold=selected_threshold,
            valid_mask=valid,
        )
        retained = frame[
            [
                "person_key",
                "run_id",
                "person_id",
                "timestamp_utc",
                "event_id",
                "stage_code",
            ]
        ].copy()
        retained["split_role"] = "validation"
        retained["event_probability"] = event_probability
        retained["predicted_stage"] = decoded
        retained["ood_status"] = ood_status
        for index, stage in enumerate(MODEL_STAGE_CODES):
            retained[f"probability_{stage}"] = stage_probability[:, index]
        prediction_parts.append(retained)

    model_entries = [
        *_dump_models(root, prefix="event", models=event_models),
        *_dump_models(root, prefix="stage", models=stage_models),
    ]
    _write_parquet(pd.DataFrame(metric_rows), root / "validation_metrics.parquet")
    _write_parquet(
        pd.concat(prediction_parts, ignore_index=True),
        root / "validation_predictions.parquet",
    )
    feature_schema = {
        "schema_version": "goal1.5/hierarchical-features/v1",
        "features": feature_names,
        "source": "causal_prepared_plus_standard_type_soft_membership",
        "forbidden": [
            "hidden_archetype",
            "intensity_truth",
            "event_intensity_truth",
            "active_target_*",
            "event_id",
            "stage_code",
        ],
    }
    (root / "feature_schema.json").write_text(
        json.dumps(feature_schema, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files = {
        path.name: sha256_file(path)
        for path in sorted(root.iterdir())
        if path.is_file()
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": STAGE_MODEL_SCHEMA,
                "status": "oracle/sanity",
                "real_data_status": "NOT VERIFIED",
                "selected_event_model": selected_event,
                "selected_stage_model": selected_stage,
                "event_threshold": selected_threshold,
                "validation_person_count": len(validation_entries),
                "training_person_count": len(train_entries),
                "models": model_entries,
                "files": files,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return StageModelArtifacts(
        root=root,
        manifest_json=manifest_path,
        selected_event_model=selected_event,
        selected_stage_model=selected_stage,
        event_threshold=selected_threshold,
    )


def _load_selected_stage_models(
    stage_root: Path,
) -> tuple[dict[str, object], ProbabilityClassifier, ProbabilityClassifier]:
    manifest = _read_json(stage_root / "manifest.json")
    entries = cast(list[dict[str, object]], manifest["models"])
    selected_event = str(manifest["selected_event_model"])
    selected_stage = str(manifest["selected_stage_model"])

    def load(head: str, name: str) -> ProbabilityClassifier:
        entry = next(
            value
            for value in entries
            if value["head"] == head and value["model_name"] == name
        )
        path = stage_root / str(entry["path"])
        discovered = sorted(sio.get_untrusted_types(file=path))
        declared = sorted(cast(list[str], entry["unknown_types"]))
        if discovered != declared:
            raise ValueError(f"skops unknown type mismatch: {path}")
        return cast(ProbabilityClassifier, sio.load(path, trusted=declared))

    return manifest, load("event", selected_event), load("stage", selected_stage)


def _event_feature_frame(
    prepared_root: Path,
    outcome_root: Path,
    standard_type_root: Path,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    prepared_manifest = _read_json(prepared_root / "manifest.json")
    people = cast(list[dict[str, object]], prepared_manifest["people"])
    base_features = [
        str(value)
        for value in cast(list[object], prepared_manifest["feature_names"])
    ]
    assert_oracle_columns(base_features)
    memberships, ood, type_features = _membership_context(standard_type_root)
    events = pq.read_table(outcome_root / "outcome_events.parquet").to_pandas()
    behaviors = pq.read_table(outcome_root / "outcome_behaviors.parquet").to_pandas()
    behavior_wide = (
        behaviors.pivot_table(
            index=["run_id", "person_id", "event_id"],
            columns="behavior_code",
            values="label_value",
            aggfunc="max",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    for code in sorted(BEHAVIOR_CODES):
        if code not in behavior_wide:
            behavior_wide[code] = np.int8(0)
    rows: list[pd.DataFrame] = []
    entry_index = {
        (str(entry["run_id"]), str(entry["person_id"])): entry for entry in people
    }
    for (run_id, person_id), person_events in events.groupby(
        ["run_id", "person_id"], sort=True
    ):
        entry = entry_index[(str(run_id), str(person_id))]
        timeline = pq.read_table(
            prepared_root / str(entry["path"]),
            columns=[
                "person_key",
                "run_id",
                "person_id",
                "timestamp_utc",
                *base_features,
            ],
        ).to_pandas()
        timeline = _add_membership(
            timeline,
            person_key=str(entry["person_key"]),
            memberships=memberships,
            type_features=type_features,
        )
        time_ns = (
            pd.to_datetime(timeline["timestamp_utc"], utc=True)
            .dt.as_unit("ns")
            .astype("int64")
            .to_numpy()
        )
        requested = (
            person_events["start_time_ns"].astype("int64").to_numpy()
            - 1_000_000_000
        )
        indices = np.searchsorted(time_ns, requested, side="right") - 1
        indices = np.clip(indices, 0, len(timeline) - 1)
        sampled = timeline.iloc[indices].reset_index(drop=True)
        sampled["event_id"] = person_events["event_id"].astype(str).to_numpy()
        sampled["is_target"] = person_events["is_target"].astype(bool).to_numpy()
        sampled["event_time_utc"] = pd.to_datetime(
            person_events["start_time_ns"].astype("int64"), unit="ns", utc=True
        ).to_numpy()
        sampled["split_role"] = str(entry["split_role"])
        rows.append(sampled)
    event_frame = pd.concat(rows, ignore_index=True)
    event_frame = event_frame.merge(
        behavior_wide[
            ["run_id", "person_id", "event_id", *sorted(BEHAVIOR_CODES)]
        ],
        on=["run_id", "person_id", "event_id"],
        how="left",
        validate="one_to_one",
    )
    event_frame[list(sorted(BEHAVIOR_CODES))] = event_frame[
        list(sorted(BEHAVIOR_CODES))
    ].fillna(0).astype(np.int8)
    event_frame = event_frame.merge(
        ood[["person_key", "status"]].rename(columns={"status": "ood_status"}),
        on="person_key",
        how="left",
        validate="many_to_one",
    )
    return event_frame, [*base_features, *type_features], ood


def fit_behavior_model_artifacts(
    prepared_root: Path,
    outcome_root: Path,
    standard_type_root: Path,
    stage_model_root: Path,
    output_root: Path,
    *,
    random_state: int,
) -> BehaviorModelArtifacts:
    """Fit leak-safe event-level behavior heads using person-grouped OOF model-1 input."""

    root = output_root.resolve()
    if root.exists():
        raise FileExistsError(f"behavior model artifact root already exists: {root}")
    root.mkdir(parents=True)
    stage_manifest, event_model, stage_model = _load_selected_stage_models(
        stage_model_root
    )
    event_frame, stage_features, _ = _event_feature_frame(
        prepared_root,
        outcome_root,
        standard_type_root,
    )
    train = event_frame.loc[event_frame["split_role"] == "train"].copy()
    validation = event_frame.loc[event_frame["split_role"] == "validation"].copy()
    train_matrix = train[stage_features].to_numpy(dtype=np.float32)
    oof = grouped_oof_probabilities(
        train_matrix,
        train["is_target"].to_numpy(dtype=np.int8),
        train["person_key"].astype(str).tolist(),
        random_state=random_state,
    )
    train["event_oof_probability"] = oof.probability
    train["event_oof_fold"] = oof.fold
    validation_matrix = validation[stage_features].to_numpy(dtype=np.float32)
    validation["event_oof_probability"] = event_model.predict_proba(
        validation_matrix
    )[:, 1]
    behavior_features = [*stage_features, "event_oof_probability"]
    behavior_codes = sorted(BEHAVIOR_CODES)
    fitted = fit_behavior_candidates(
        train,
        validation,
        feature_names=behavior_features,
        behavior_codes=behavior_codes,
        random_state=random_state,
    )

    model_entries: list[dict[str, object]] = []
    for (code, model_name), model in sorted(fitted.models.items()):
        relative = f"behavior__{code}__{model_name}.skops"
        sio.dump(model, root / relative)
        model_entries.append(
            {
                "behavior_code": code,
                "model_name": model_name,
                "path": relative,
                "unknown_types": sorted(sio.get_untrusted_types(file=root / relative)),
                "selected": fitted.selected_model_by_behavior.get(code) == model_name,
            }
        )
    _write_parquet(fitted.metrics, root / "validation_metrics.parquet")
    _write_parquet(
        train[
            [
                "person_key",
                "run_id",
                "person_id",
                "event_id",
                "event_oof_probability",
                "event_oof_fold",
            ]
        ],
        root / "model1_train_oof.parquet",
    )

    audit_rows: list[dict[str, object]] = []
    for role, frame in (("validation", validation),):
        matrix = frame[behavior_features].to_numpy(dtype=np.float32)
        stage_matrix = frame[stage_features].to_numpy(dtype=np.float32)
        stage_probability = _aligned_stage_probability(stage_model, stage_matrix)
        event_probability = frame["event_oof_probability"].to_numpy(dtype=np.float64)
        event_threshold = float(
            cast(str | float | int, stage_manifest["event_threshold"])
        )
        predicted_stage = [
            (
                MODEL_STAGE_CODES[int(np.argmax(stage_probability[index]))]
                if event_probability[index] >= event_threshold
                else "NO_EVENT"
            )
            for index in range(len(frame))
        ]
        records = frame.reset_index(drop=True).to_dict(orient="records")
        for index, record in enumerate(records):
            for code in behavior_codes:
                status = fitted.status_by_behavior[code]
                selected_name = fitted.selected_model_by_behavior.get(code)
                probability = np.nan
                if selected_name is not None:
                    probability = float(
                        fitted.models[(code, selected_name)].predict_proba(
                            matrix[index : index + 1]
                        )[0, 1]
                    )
                audit_rows.append(
                    {
                        "person_key": str(record["person_key"]),
                        "run_id": str(record["run_id"]),
                        "person_id": str(record["person_id"]),
                        "event_id": str(record["event_id"]),
                        "event_time_utc": record["event_time_utc"],
                        "split_role": role,
                        "event_probability": float(event_probability[index]),
                        "predicted_stage": predicted_stage[index],
                        "ood_status": str(record["ood_status"]),
                        "behavior_code": code,
                        "observed_label": int(record[code]),
                        "behavior_probability": probability,
                        "support_status": status,
                        "selected_model": selected_name,
                    }
                )
    audit = pd.DataFrame(audit_rows)
    _write_parquet(audit, root / "event_prediction_audit.parquet")

    calibration_rows: list[dict[str, object]] = []
    for person_key, person in event_frame.groupby("person_key", sort=True):
        distinct_days = int(
            pd.to_datetime(person["event_time_utc"], utc=True).dt.date.nunique()
        )
        for code in behavior_codes:
            positive = int(person[code].sum())
            count = len(person)
            calibration_rows.append(
                {
                    "person_key": str(person_key),
                    "behavior_code": code,
                    "event_count": count,
                    "positive_count": positive,
                    "negative_count": count - positive,
                    "distinct_days": distinct_days,
                    "calibration_status": personal_calibration_status(
                        event_count=count,
                        positive_count=positive,
                        negative_count=count - positive,
                        distinct_days=distinct_days,
                    ),
                }
            )
    _write_parquet(
        pd.DataFrame(calibration_rows),
        root / "personal_calibration_status.parquet",
    )
    (root / "feature_schema.json").write_text(
        json.dumps(
            {
                "schema_version": "goal1.5/behavior-features/v1",
                "features": behavior_features,
                "model1_train_input": "person_grouped_out_of_fold_event_probability",
                "forbidden": [
                    "is_target",
                    "event_type",
                    "event_id",
                    "hidden_archetype",
                    "intensity_truth",
                    "review_tag",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    files = {
        path.name: sha256_file(path)
        for path in sorted(root.iterdir())
        if path.is_file()
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": BEHAVIOR_MODEL_SCHEMA,
                "status": "oracle/sanity",
                "real_data_status": "NOT VERIFIED",
                "models": model_entries,
                "support_status": dict(fitted.status_by_behavior),
                "selected_models": dict(fitted.selected_model_by_behavior),
                "model1_train_probability": "person_grouped_out_of_fold",
                "individual_models_created": False,
                "files": files,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return BehaviorModelArtifacts(root=root, manifest_json=manifest_path)
