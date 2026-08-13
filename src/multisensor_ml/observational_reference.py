"""Training-only simple reference contract for continuous Watch forecasts.

This module never selects a reference or a MASE denominator from validation or
locked-test rows.  It is separate from the five-stage pattern model.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Final, cast

import numpy as np
import pandas as pd

from multisensor_ml.observational_contract import (
    BASELINE_VERSION,
    FEATURE_NAMES,
    FEATURE_SCHEMA_SHA256,
    FEATURE_SCHEMA_VERSION,
    LOAD_FORMULA_VERSION,
    QUALITY_FORMULA_VERSION,
    TARGET_VERSION,
    build_observational_frame,
    canonical_feature_schema,
)

REFERENCE_CONTRACT_VERSION: Final[str] = "observational-training-reference/v1"
REFERENCE_RELEASE: Final[str] = "observational_standard_30m_watch_reference_training_v1"
CALCULATION_CODE_VERSION: Final[str] = "observational_reference_training_only_v1"
REFERENCE_METHODS: Final[tuple[str, str]] = (
    "persistence",
    "rolling_median_300",
)
REFERENCE_FEATURE: Final[dict[str, str]] = {
    "persistence": "watch_load_raw",
    "rolling_median_300": "watch_load_median_300",
}
IN_DISTRIBUTION: Final[str] = "IN_DISTRIBUTION"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _normalized_generated_at(value: str) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("generated_at must include a timezone")
    return timestamp.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _training_split_digest(frame: pd.DataFrame) -> str:
    required = {
        "person_key",
        "session_id",
        "corrected_utc",
        "split_role",
        "standard_target",
        "target_valid",
        "feature_decisionable",
        "future_valid_fraction",
        "future_max_gap_seconds",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"split assignments missing columns: {missing}")
    train = frame.loc[frame["split_role"].astype(str).eq("train")].copy()
    if train.empty:
        raise ValueError("split assignments contain no training rows")
    train["corrected_utc"] = pd.to_datetime(
        train["corrected_utc"], utc=True, errors="raise"
    )
    train = train.sort_values(
        ["person_key", "session_id", "corrected_utc"], kind="stable"
    )
    if train.duplicated(["person_key", "session_id", "corrected_utc"]).any():
        raise ValueError("training split contains duplicate identity rows")
    digest = hashlib.sha256()
    digest_columns = [
        "person_key",
        "session_id",
        "corrected_utc",
        "standard_target",
        "target_valid",
        "feature_decisionable",
        "future_valid_fraction",
        "future_max_gap_seconds",
    ]
    for row in train[digest_columns].to_dict(orient="records"):
        timestamp = cast(pd.Timestamp, row["corrected_utc"])
        value = {
            "person_key": str(row["person_key"]),
            "session_id": str(row["session_id"]),
            "corrected_utc": timestamp.isoformat().replace("+00:00", "Z"),
            "split_role": "train",
            "standard_target_hex": float(cast(float, row["standard_target"])).hex(),
            "target_valid": int(cast(int, row["target_valid"])),
            "feature_decisionable": int(cast(int, row["feature_decisionable"])),
            "future_valid_fraction_hex": float(
                cast(float, row["future_valid_fraction"])
            ).hex(),
            "future_max_gap_seconds": int(
                cast(int, row["future_max_gap_seconds"])
            ),
        }
        digest.update(_canonical_json(value))
        digest.update(b"\n")
    return digest.hexdigest()


def _training_source_slice(source: pd.DataFrame, train_split: pd.DataFrame) -> pd.DataFrame:
    keys = ["person_key", "session_id"]
    source_copy = source.copy()
    source_copy["person_key"] = source_copy["person_key"].astype(str)
    source_copy["session_id"] = source_copy["session_id"].astype(str)
    source_copy["corrected_utc"] = pd.to_datetime(
        source_copy["corrected_utc"], utc=True, errors="raise"
    )
    bounds = (
        train_split.assign(
            person_key=train_split["person_key"].astype(str),
            session_id=train_split["session_id"].astype(str),
            corrected_utc=pd.to_datetime(
                train_split["corrected_utc"], utc=True, errors="raise"
            ),
        )
        .groupby(keys, sort=True)["corrected_utc"]
        .max()
        .rename("training_max_utc")
        .reset_index()
    )
    selected = source_copy.merge(bounds, on=keys, how="inner", validate="many_to_one")
    selected = selected.loc[selected["corrected_utc"] <= selected["training_max_utc"]]
    return selected.drop(columns=["training_max_utc"])


def _training_reference_evidence(
    source_path: Path,
    candidate_bundle: Path,
) -> dict[str, object]:
    source = source_path.resolve()
    root = candidate_bundle.resolve()
    manifest_path = root / "manifest.json"
    split_path = root / "split_assignments.parquet"
    schema_path = root / "feature_schema.json"
    for required in (source, manifest_path, split_path, schema_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    source_sha256 = _sha256_file(source)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_file_sha256") != source_sha256:
        raise ValueError("training source SHA-256 differs from candidate manifest")
    if manifest.get("target_version") != TARGET_VERSION:
        raise ValueError("candidate target version differs from canonical target")
    if manifest.get("feature_schema_hash") != FEATURE_SCHEMA_SHA256:
        raise ValueError("candidate feature schema hash differs from canonical schema")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if schema != canonical_feature_schema():
        raise ValueError("candidate feature schema differs from canonical schema")

    split = pd.read_parquet(split_path)
    train_split = split.loc[split["split_role"].astype(str).eq("train")].copy()
    if train_split.empty:
        raise ValueError("candidate split has no training rows")
    if not train_split["target_valid"].astype(bool).all():
        raise ValueError("training split contains invalid targets")
    if not train_split["feature_decisionable"].astype(bool).all():
        raise ValueError("training split contains non-decisionable features")
    observed = pd.read_parquet(source)
    training_source = _training_source_slice(observed, train_split)
    built = build_observational_frame(training_source)
    keys = ["person_key", "session_id", "corrected_utc"]
    built["person_key"] = built["person_key"].astype(str)
    built["session_id"] = built["session_id"].astype(str)
    train_split["person_key"] = train_split["person_key"].astype(str)
    train_split["session_id"] = train_split["session_id"].astype(str)
    train_split["corrected_utc"] = pd.to_datetime(
        train_split["corrected_utc"], utc=True, errors="raise"
    )
    joined = train_split.merge(
        built[[*keys, "watch_load_raw", "watch_load_median_300"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if len(joined) != len(train_split):
        raise ValueError("training feature join changed row count")
    target = joined["standard_target"].to_numpy(dtype="float64")
    metrics: dict[str, float] = {}
    for method in REFERENCE_METHODS:
        prediction = joined[REFERENCE_FEATURE[method]].to_numpy(dtype="float64")
        valid = np.isfinite(target) & np.isfinite(prediction)
        if int(valid.sum()) != len(joined):
            raise ValueError(f"training reference contains non-finite values: {method}")
        metrics[method] = float(np.mean(np.abs(prediction - target)))
    selected = min(
        REFERENCE_METHODS,
        key=lambda method: (metrics[method], REFERENCE_METHODS.index(method)),
    )
    time_values = pd.to_datetime(train_split["corrected_utc"], utc=True)
    source_files = [Path(__file__).resolve(), Path(build_observational_frame.__code__.co_filename)]
    calculation_files = {
        path.name: _sha256_file(path)
        for path in sorted(set(source_files), key=lambda item: item.name)
    }
    calculation_digest = hashlib.sha256(_canonical_json(calculation_files)).hexdigest()
    return {
        "target_version": TARGET_VERSION,
        "feature_schema_hash": FEATURE_SCHEMA_SHA256,
        "training_source_sha256": source_sha256,
        "training_split_digest": _training_split_digest(split),
        "training_time_bounds": {
            "min_corrected_utc": time_values.min()
            .isoformat()
            .replace("+00:00", "Z"),
            "max_corrected_utc": time_values.max()
            .isoformat()
            .replace("+00:00", "Z"),
        },
        "training_rows": len(joined),
        "training_persons": int(train_split["person_key"].nunique()),
        "training_sessions": int(train_split["session_id"].nunique()),
        "persistence_training_mae": metrics["persistence"],
        "rolling_median_300_training_mae": metrics["rolling_median_300"],
        "reference_method": selected,
        "selection_lineage": {
            "roles_read": ["train"],
            "validation_read": False,
            "locked_test_read": False,
        },
        "calculation_source_digest": calculation_digest,
        "calculation_files_sha256": calculation_files,
        "source_paths": {
            "training_source": str(source),
            "candidate_bundle": str(root),
        },
        "input_evidence_files": {
            "training_source.parquet": source_sha256,
            "split_assignments.parquet": _sha256_file(split_path),
            "candidate_manifest.json": _sha256_file(manifest_path),
            "feature_schema.json": _sha256_file(schema_path),
        },
    }


def _reference_onnx_bytes(method: str) -> bytes:
    import onnx
    from onnx import TensorProto, helper

    if method not in REFERENCE_METHODS:
        raise ValueError(f"unsupported reference method: {method}")
    feature_index = FEATURE_NAMES.index(REFERENCE_FEATURE[method])
    input_info = helper.make_tensor_value_info(
        "features", TensorProto.FLOAT, ["N", len(FEATURE_NAMES)]
    )
    output_info = helper.make_tensor_value_info(
        "prediction", TensorProto.FLOAT, ["N", 1]
    )
    index = helper.make_tensor("feature_index", TensorProto.INT64, [1], [feature_index])
    node = helper.make_node(
        "Gather", inputs=["features", "feature_index"], outputs=["prediction"], axis=1
    )
    graph = helper.make_graph(
        [node],
        "observational_training_reference_v1",
        [input_info],
        [output_info],
        [index],
    )
    model = helper.make_model(
        graph,
        producer_name="multisensor-ml",
        producer_version=CALCULATION_CODE_VERSION,
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx.checker.check_model(model)
    return cast(bytes, model.SerializeToString())


def reference_prediction(
    vector: np.ndarray,
    *,
    feature_decisionable: bool,
    ood_status: str,
) -> dict[str, object]:
    """Apply the serving gate; the ONNX model is never called for invalid input."""

    values = np.asarray(vector, dtype="float32")
    valid = (
        feature_decisionable
        and ood_status == IN_DISTRIBUTION
        and values.shape == (len(FEATURE_NAMES),)
        and bool(np.isfinite(values).all())
    )
    if not valid:
        return {"status": "NOT_DECISIONABLE", "prediction_value": None}
    return {"status": "PREDICTED", "prediction_value": float(values[0])}


def _run_golden(model_path: Path, method: str) -> dict[str, object]:
    import onnxruntime as ort  # type: ignore[import-untyped]

    vector = np.arange(len(FEATURE_NAMES), dtype="float32").reshape(1, -1)
    expected = float(vector[0, FEATURE_NAMES.index(REFERENCE_FEATURE[method])])
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"], sess_options=options
    )
    prediction = float(session.run(None, {"features": vector})[0][0, 0])
    if not math.isclose(prediction, expected, abs_tol=1e-7):
        raise ValueError("reference ONNX golden prediction mismatch")
    return {
        "schema_version": "observational-training-reference-golden/v1",
        "ordered_feature_names": list(FEATURE_NAMES),
        "input_name": "features",
        "output_name": "prediction",
        "input_vector": vector.reshape(-1).tolist(),
        "expected_prediction": expected,
        "reference_method": method,
    }


def export_training_reference_bundle(
    source_path: Path,
    candidate_bundle: Path,
    output_root: Path,
    *,
    generated_at: str,
) -> dict[str, object]:
    """Export an immutable training-only reference and its evidence contract."""

    root = output_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"training reference bundle already exists: {root}")
    root.mkdir(parents=True, exist_ok=True)
    evidence = _training_reference_evidence(source_path, candidate_bundle)
    method = str(evidence["reference_method"])
    model_bytes = _reference_onnx_bytes(method)
    if model_bytes != _reference_onnx_bytes(method):
        raise ValueError("reference ONNX serialization is not deterministic")
    model_path = root / "reference.onnx"
    model_path.write_bytes(model_bytes)
    golden = _run_golden(model_path, method)
    _write_json(root / "golden_reference.json", golden)

    valid_vector = np.arange(len(FEATURE_NAMES), dtype="float32")
    missing_vector = valid_vector.copy()
    missing_vector[0] = np.nan
    if reference_prediction(
        valid_vector,
        feature_decisionable=True,
        ood_status=IN_DISTRIBUTION,
    )["status"] != "PREDICTED":
        raise ValueError("valid reference vector was rejected")
    if reference_prediction(
        missing_vector,
        feature_decisionable=True,
        ood_status=IN_DISTRIBUTION,
    )["status"] != "NOT_DECISIONABLE":
        raise ValueError("missing reference vector did not fail closed")
    if reference_prediction(
        valid_vector,
        feature_decisionable=True,
        ood_status="OOD_MONITOR",
    )["status"] != "NOT_DECISIONABLE":
        raise ValueError("OOD reference vector did not fail closed")

    evidence_files = dict(cast(dict[str, str], evidence.pop("input_evidence_files")))
    evidence_files.update(
        {
            "reference.onnx": _sha256_file(model_path),
            "golden_reference.json": _sha256_file(root / "golden_reference.json"),
        }
    )
    persistence_mae = float(cast(float, evidence["persistence_training_mae"]))
    contract: dict[str, object] = {
        "contract_version": REFERENCE_CONTRACT_VERSION,
        **evidence,
        "reference_release": REFERENCE_RELEASE,
        "reference_artifact_sha256": _sha256_file(model_path),
        "reference_feature": REFERENCE_FEATURE[method],
        "reference_selection": {
            "candidate_methods": list(REFERENCE_METHODS),
            "metric": "training_mae",
            "minimum_wins": True,
            "tie_break": "persistence",
            "selected_before_validation_or_locked_holdout": True,
        },
        "mase_denominator": {
            "method": "persistence_training_mae",
            "value": persistence_mae,
            "split_role": "train",
        },
        "calculation_code_version": CALCULATION_CODE_VERSION,
        "generated_at": _normalized_generated_at(generated_at),
        "golden_verified": True,
        "idempotency_verified": True,
        "missing_ood_verified": True,
        "thresholds": None,
        "stage": None,
        "real_data_status": "NOT VERIFIED",
        "evidence_files": evidence_files,
        "runtime_contract": {
            "provider": "CPUExecutionProvider",
            "input_name": "features",
            "input_dtype": "float32",
            "input_shape": [None, len(FEATURE_NAMES)],
            "output_name": "prediction",
            "output_dtype": "float32",
            "output_shape": [None, 1],
            "missing_or_ood": "NOT_DECISIONABLE",
        },
        "versions": {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "baseline_version": BASELINE_VERSION,
            "load_formula_version": LOAD_FORMULA_VERSION,
            "quality_formula_version": QUALITY_FORMULA_VERSION,
        },
    }
    if _canonical_json(contract) != _canonical_json(dict(contract)):
        raise ValueError("reference contract serialization is not deterministic")
    _write_json(root / "reference_contract.json", contract)
    checksums = {
        path.name: _sha256_file(path)
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "SHA256SUMS.json"
    }
    _write_json(root / "SHA256SUMS.json", checksums)
    return contract


def verify_training_reference_bundle(bundle: Path) -> dict[str, object]:
    """Verify immutable hashes, ONNX golden output, and fail-closed gates."""

    import onnx

    root = bundle.resolve()
    contract_path = root / "reference_contract.json"
    checksums_path = root / "SHA256SUMS.json"
    if not contract_path.is_file() or not checksums_path.is_file():
        raise FileNotFoundError(f"reference bundle is incomplete: {root}")
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    for name, digest in checksums.items():
        path = root / str(name)
        if not path.is_file() or _sha256_file(path) != str(digest):
            raise ValueError(f"reference checksum mismatch: {name}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("contract_version") != REFERENCE_CONTRACT_VERSION:
        raise ValueError("reference contract version mismatch")
    if contract.get("target_version") != TARGET_VERSION:
        raise ValueError("reference target version mismatch")
    if contract.get("feature_schema_hash") != FEATURE_SCHEMA_SHA256:
        raise ValueError("reference feature schema mismatch")
    if contract.get("thresholds") is not None or contract.get("stage") is not None:
        raise ValueError("continuous reference must not contain thresholds or stage")
    lineage = contract.get("selection_lineage")
    if lineage != {
        "roles_read": ["train"],
        "validation_read": False,
        "locked_test_read": False,
    }:
        raise ValueError("reference selection lineage is not training-only")
    model_path = root / "reference.onnx"
    if _sha256_file(model_path) != contract.get("reference_artifact_sha256"):
        raise ValueError("reference ONNX SHA-256 mismatch")
    onnx.checker.check_model(onnx.load(model_path))
    method = str(contract["reference_method"])
    golden = _run_golden(model_path, method)
    expected_golden = json.loads(
        (root / "golden_reference.json").read_text(encoding="utf-8")
    )
    if golden != expected_golden:
        raise ValueError("reference golden fixture mismatch")
    missing = np.arange(len(FEATURE_NAMES), dtype="float32")
    missing[0] = np.nan
    if reference_prediction(
        missing, feature_decisionable=True, ood_status=IN_DISTRIBUTION
    )["status"] != "NOT_DECISIONABLE":
        raise ValueError("reference missing-input gate mismatch")
    if reference_prediction(
        np.arange(len(FEATURE_NAMES), dtype="float32"),
        feature_decisionable=True,
        ood_status="OOD_MONITOR",
    )["status"] != "NOT_DECISIONABLE":
        raise ValueError("reference OOD gate mismatch")
    return {
        "status": "VERIFIED",
        "reference_release": contract["reference_release"],
        "reference_method": method,
        "reference_artifact_sha256": _sha256_file(model_path),
        "reference_contract_sha256": _sha256_file(contract_path),
        "training_split_digest": contract["training_split_digest"],
        "persistence_training_mae": contract["persistence_training_mae"],
        "golden_verified": True,
        "idempotency_verified": bool(contract["idempotency_verified"]),
        "missing_ood_verified": True,
    }


__all__ = [
    "CALCULATION_CODE_VERSION",
    "REFERENCE_CONTRACT_VERSION",
    "REFERENCE_RELEASE",
    "export_training_reference_bundle",
    "reference_prediction",
    "verify_training_reference_bundle",
]
