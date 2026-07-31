from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
import skops.io as sio

from multisensor_ml.hierarchical import MODEL_STAGE_CODES, decode_stage_sequence
from multisensor_ml.kaggle_model_contracts import BEHAVIOR_CODES
from multisensor_ml.kaggle_model_package import verify_payload
from multisensor_ml.models import ProbabilityClassifier
from multisensor_ml.result_router import route_prediction_ko


@dataclass(frozen=True, slots=True)
class LoadedHierarchicalPackage:
    root: Path
    manifest: dict[str, object]
    event_model: ProbabilityClassifier
    stage_model: ProbabilityClassifier
    behavior_models: dict[str, ProbabilityClassifier]
    event_threshold: float
    stage_features: tuple[str, ...]
    behavior_features: tuple[str, ...]
    memberships: pd.DataFrame
    ood_status: pd.DataFrame
    forbidden_patterns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReproductionResult:
    status: Literal["REPRODUCED", "FAILED"]
    compared_rows: int
    first_mismatch_column: str | None
    first_mismatch_key: str | None


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return cast(dict[str, object], payload)


def _safe_load(root: Path, entry: dict[str, object]) -> ProbabilityClassifier:
    path = root / str(entry["path"])
    discovered = sorted(sio.get_untrusted_types(file=path))
    declared = sorted(cast(list[str], entry["unknown_types"]))
    if discovered != declared:
        raise ValueError(f"skops unknown type mismatch: {entry['path']}")
    return cast(ProbabilityClassifier, sio.load(path, trusted=declared))


def load_hierarchical_package(payload_root: Path) -> LoadedHierarchicalPackage:
    root = payload_root.resolve()
    manifest = verify_payload(root)
    models = cast(list[dict[str, object]], manifest["models"])

    def selected(
        group: str, *, head: str | None = None, code: str | None = None
    ) -> dict[str, object]:
        if head == "event":
            name = str(manifest["selected_event_model"])
        elif head == "stage":
            name = str(manifest["selected_stage_model"])
        elif code is not None:
            name = str(cast(dict[str, str], manifest["selected_behavior_models"])[code])
        else:
            raise ValueError("selected model identity is incomplete")
        return next(
            entry
            for entry in models
            if entry["group"] == group
            and entry.get("head") == head
            and entry.get("behavior_code") == code
            and entry.get("model_name") == name
        )

    event_model = _safe_load(root, selected("stage_model", head="event"))
    stage_model = _safe_load(root, selected("stage_model", head="stage"))
    behavior_models = {
        code: _safe_load(root, selected("behavior_model", code=code))
        for code in BEHAVIOR_CODES
    }
    stage_schema = _read_json(root / "stage_model/feature_schema.json")
    behavior_schema = _read_json(root / "behavior_model/feature_schema.json")
    return LoadedHierarchicalPackage(
        root=root,
        manifest=manifest,
        event_model=event_model,
        stage_model=stage_model,
        behavior_models=behavior_models,
        event_threshold=float(cast(float | int | str, manifest["event_threshold"])),
        stage_features=tuple(str(value) for value in cast(list[object], stage_schema["features"])),
        behavior_features=tuple(
            str(value) for value in cast(list[object], behavior_schema["features"])
        ),
        memberships=pd.read_parquet(root / "standard_type/person_standard_types.parquet"),
        ood_status=pd.read_parquet(root / "standard_type/ood_status.parquet"),
        forbidden_patterns=tuple(
            str(value) for value in cast(list[object], stage_schema.get("forbidden", []))
        ),
    )


def prepare_hierarchical_input(
    package: LoadedHierarchicalPackage, frame: pd.DataFrame
) -> pd.DataFrame:
    forbidden = sorted(
        column
        for column in frame.columns
        if any(fnmatch.fnmatch(column, pattern) for pattern in package.forbidden_patterns)
    )
    if forbidden:
        raise ValueError(f"forbidden model input columns: {forbidden}")
    required_identity = {"person_key", "session_id", "timestamp_utc"}
    missing_identity = sorted(required_identity.difference(frame.columns))
    if missing_identity:
        raise ValueError(f"missing inference identity columns: {missing_identity}")
    if frame.duplicated(list(required_identity)).any():
        raise ValueError("duplicate inference identity")
    joined = frame.merge(
        package.memberships,
        on="person_key",
        how="left",
        validate="many_to_one",
    ).merge(
        package.ood_status.rename(columns={"status": "ood_status"}),
        on="person_key",
        how="left",
        validate="many_to_one",
    )
    if joined["ood_status"].isna().any() or joined["STD-A"].isna().any():
        raise ValueError("unknown person has no frozen STD-A membership")
    missing_features = sorted(set(package.stage_features).difference(joined.columns))
    if missing_features:
        raise ValueError(f"missing stage features: {missing_features}")
    return joined.sort_values(
        ["person_key", "session_id", "timestamp_utc"], kind="stable"
    ).reset_index(drop=True)


def predict_hierarchical(
    package: LoadedHierarchicalPackage, frame: pd.DataFrame
) -> pd.DataFrame:
    prepared = prepare_hierarchical_input(package, frame)
    stage_matrix = prepared[list(package.stage_features)].to_numpy(dtype=np.float32)
    event_probability = package.event_model.predict_proba(stage_matrix)[:, 1].astype(float)
    stage_probability = package.stage_model.predict_proba(stage_matrix).astype(float)
    if stage_probability.shape[1] != len(MODEL_STAGE_CODES):
        raise ValueError("stage model does not expose five classes")
    valid = ~prepared["ood_status"].isin({"NOT_DECISIONABLE", "RETRAIN_CANDIDATE"})
    predicted_stage: list[str] = []
    for _, indices in prepared.groupby(["person_key", "session_id"], sort=False).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        predicted_stage.extend(
            decode_stage_sequence(
                event_probability[positions],
                stage_probability[positions],
                event_threshold=package.event_threshold,
                valid_mask=valid.to_numpy(dtype=bool)[positions],
            )
        )
    prepared = prepared.assign(event_oof_probability=event_probability)
    missing_behavior = sorted(set(package.behavior_features).difference(prepared.columns))
    if missing_behavior:
        raise ValueError(f"missing behavior features: {missing_behavior}")
    behavior_matrix = prepared[list(package.behavior_features)].to_numpy(dtype=np.float32)
    output = prepared[["person_key", "session_id", "timestamp_utc", "ood_status"]].copy()
    output["event_probability"] = event_probability
    output["event_threshold"] = package.event_threshold
    output["predicted_event"] = event_probability >= package.event_threshold
    output["predicted_stage"] = predicted_stage
    output["std_type"] = "STD-A"
    output["std_a_membership"] = prepared["STD-A"].to_numpy(dtype=float)
    for code in BEHAVIOR_CODES:
        output[code] = package.behavior_models[code].predict_proba(behavior_matrix)[:, 1]
    routed = [
        route_prediction_ko(
            {
                "event_probability": float(row["event_probability"]),
                "stage_code": str(row["predicted_stage"]),
                "ood_status": str(row["ood_status"]),
                "behavior_probabilities": {code: float(row[code]) for code in BEHAVIOR_CODES},
            },
            router_version="ko-v1",
        )
        for row in output.to_dict(orient="records")
    ]
    output["판정"] = [str(value["판정"]) for value in routed]
    output["단계"] = [str(value["단계"]) for value in routed]
    output["분포상태"] = [str(value["분포상태"]) for value in routed]
    return output


def compare_expected(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    probability_atol: float = 1e-6,
) -> ReproductionResult:
    if list(actual.columns) != list(expected.columns) or len(actual) != len(expected):
        return ReproductionResult("FAILED", min(len(actual), len(expected)), "schema", None)
    key_columns = ["person_key", "session_id", "timestamp_utc"]
    for column in actual.columns:
        left = actual[column]
        right = expected[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            equal = np.isclose(
                left.to_numpy(),
                right.to_numpy(),
                atol=probability_atol,
                rtol=0,
                equal_nan=True,
            )
        else:
            equal = left.astype(str).to_numpy() == right.astype(str).to_numpy()
        if not bool(np.all(equal)):
            index = int(np.flatnonzero(~equal)[0])
            key = "|".join(str(actual.iloc[index][name]) for name in key_columns)
            return ReproductionResult("FAILED", len(actual), column, key)
    return ReproductionResult("REPRODUCED", len(actual), None, None)
