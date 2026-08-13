"""Train the actual-observation 30-minute standard candidate.

Synthetic oracle bundles remain separate.  The legacy 15-feature experiment
is retained only as provenance and is explicitly not valid for serving.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import shutil
from pathlib import Path
from typing import Final, Literal, cast

import numpy as np
import pandas as pd
import skops.io as sio
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from multisensor_ml.observational_contract import (
    BASELINE_MAX_SPAN_SEC,
    BASELINE_VERSION,
    BASELINE_WARMUP_SEC,
    DECISION_WARMUP_SEC,
    EMA_ALPHA_1800,
    EMA_ALPHA_21600,
    FEATURE_NAMES,
    FEATURE_SCHEMA_SHA256,
    FEATURE_SCHEMA_VERSION,
    FORECAST_HORIZON_SEC,
    LEGACY_15_FEATURE_STATUS,
    LOAD_FORMULA_VERSION,
    MAX_GAP_SEC,
    MIN_VALID_FRACTION,
    MODEL_RELEASE,
    PURGE_SEC,
    QUALITY_FORMULA_VERSION,
    QUALITY_GATE,
    SIGNAL_COLUMNS,
    TARGET_VERSION,
    ObservationalStandardConfig,
    bounded_ema_eligible,
    build_observational_frame,
    canonical_feature_schema,
    export_baseline_shadow_bundle,
    ordered_feature_matrix,
    verify_baseline_shadow_bundle,
)

OBSERVATIONAL_STANDARD_SCHEMA: Final[str] = "goal1.5/observational-standard-30m/v2"
MODEL_VERSION: Final[str] = "observational_standard_30m_watch_candidate_v2"
CANDIDATE_RELEASE: Final[str] = (
    "observational_standard_30m_watch_baseline_shadow_candidate_v1"
)
CANDIDATE_SELECTION_BLOCKER: Final[str] = (
    "NO_SEPARATE_CALIBRATION_TRAINING_EVIDENCE"
)
MIN_FUTURE_VALID_FRACTION: Final[float] = MIN_VALID_FRACTION
SOURCE_DOMAINS: Final[tuple[str, str]] = (
    "real_observed",
    "synthetic_truth_oracle",
)
LEARNED_CANDIDATES: Final[tuple[str, ...]] = (
    "ridge",
    "elasticnet",
    "hist_gradient_boosting",
)
ALL_CANDIDATES: Final[tuple[str, ...]] = (
    "persistence",
    "rolling_median_300",
    *LEARNED_CANDIDATES,
)
SAFE_SKOPS_UNKNOWN_TYPES: Final[frozenset[str]] = frozenset({"numpy.dtype"})
MIN_SPLIT_SUPPORT_ROWS: Final[int] = 100


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _person_group_roles(people: list[str]) -> dict[str, str]:
    ordered = sorted(
        people,
        key=lambda value: hashlib.sha256(value.encode("utf-8")).digest(),
    )
    n_people = len(ordered)
    train_count = max(1, math.floor(n_people * 0.60))
    validation_count = max(1, math.floor(n_people * 0.20))
    if train_count + validation_count >= n_people:
        validation_count = max(0, n_people - train_count - 1)
    roles: dict[str, str] = {}
    for index, person in enumerate(ordered):
        if index < train_count:
            roles[person] = "train"
        elif index < train_count + validation_count:
            roles[person] = "validation"
        else:
            roles[person] = "locked_test"
    return roles


def make_purged_splits(
    frame: pd.DataFrame,
    *,
    purge_sec: int = PURGE_SEC,
) -> pd.DataFrame:
    """Assign person-group or chronological roles without reading locked test."""

    if purge_sec < FORECAST_HORIZON_SEC:
        raise ValueError("purge_sec must cover the forecast horizon")
    required = {
        "person_key",
        "session_id",
        "corrected_utc",
        "feature_decisionable",
        "target_valid",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"split frame missing columns: {missing}")
    result = frame.copy()
    result["split_role"] = "purged"
    usable = result["feature_decisionable"].astype(bool) & result["target_valid"].astype(bool)
    people = sorted(result["person_key"].astype(str).unique().tolist())
    if len(people) > 1:
        roles = _person_group_roles(people)
        result["split_role"] = result["person_key"].astype(str).map(roles).fillna("purged")
        result.loc[~usable, "split_role"] = "purged"
        return result
    embargo = pd.to_timedelta(purge_sec, unit="s")
    for _, group in result.groupby(["person_key", "session_id"], sort=True):
        positions = group.index
        usable_times = (
            group.loc[usable.loc[positions], "corrected_utc"]
            .sort_values(kind="stable")
        )
        if len(usable_times) < 1:
            continue
        start = cast(pd.Timestamp, usable_times.iloc[0])
        end = cast(pd.Timestamp, usable_times.iloc[-1])
        duration = max((end - start).total_seconds(), 1.0)
        # The purge is applied on the *old* side of each boundary.  Thus a
        # train row cannot have a future target crossing into validation, and
        # a validation row cannot have a future target crossing into locked
        # test.  This preserves a usable validation slice for short smoke
        # sessions while still enforcing the full forecast embargo.
        first_offset = max(
            float(purge_sec + MIN_SPLIT_SUPPORT_ROWS), duration * 0.35
        )
        first_boundary = start + pd.to_timedelta(first_offset, unit="s")
        times = result.loc[positions, "corrected_utc"]
        train = times < first_boundary - embargo
        # If the post-warm-up portion cannot fit a second purge plus a test
        # slice, keep the locked test empty rather than inventing a boundary.
        # Every validation row is still target-valid and the train side keeps
        # the full 1,800-second future-target purge.
        test_room = first_offset + purge_sec + MIN_SPLIT_SUPPORT_ROWS
        if duration < test_room:
            validation = times >= first_boundary
            locked_test = pd.Series(False, index=times.index)
        else:
            second_offset = max(
                first_offset + purge_sec + MIN_SPLIT_SUPPORT_ROWS, duration * 0.75
            )
            second_offset = min(
                second_offset, max(duration - MIN_SPLIT_SUPPORT_ROWS, 1.0)
            )
            second_boundary = start + pd.to_timedelta(second_offset, unit="s")
            validation = (times >= first_boundary) & (
                times < second_boundary - embargo
            )
            locked_test = times >= second_boundary
        result.loc[positions[train], "split_role"] = "train"
        result.loc[positions[validation], "split_role"] = "validation"
        result.loc[positions[locked_test], "split_role"] = "locked_test"
    result.loc[~usable, "split_role"] = "purged"
    return result


def _numeric_features(frame: pd.DataFrame, rows: pd.Series) -> np.ndarray:
    selected = frame.loc[rows, list(FEATURE_NAMES)]
    return ordered_feature_matrix(selected)


def _regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(target) & np.isfinite(prediction)
    if not valid.any():
        return {
            "rows": 0,
            "mae": math.nan,
            "rmse": math.nan,
            "r2": math.nan,
            "correlation": math.nan,
        }
    truth = target[valid]
    predicted = prediction[valid]
    error = predicted - truth
    variance = float(np.sum((truth - truth.mean()) ** 2))
    residual = float(np.sum(error**2))
    correlation = (
        float(np.corrcoef(truth, predicted)[0, 1]) if len(truth) > 1 else math.nan
    )
    return {
        "rows": len(truth),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(1.0 - residual / variance) if variance > 0 else math.nan,
        "correlation": correlation,
    }


def _candidate_models(config: ObservationalStandardConfig) -> dict[str, Pipeline]:
    return {
        "ridge": Pipeline(
            [("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]
        ),
        "elasticnet": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=0.01, l1_ratio=0.20, max_iter=5_000)),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=config.hgb_max_iter,
                        max_depth=4,
                        learning_rate=0.05,
                        l2_regularization=0.1,
                        random_state=config.random_state,
                    ),
                )
            ]
        ),
    }


def _dump_onnx(model: Pipeline, path: Path) -> None:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    converted = convert_sklearn(
        model,
        initial_types=[("features", FloatTensorType([None, len(FEATURE_NAMES)]))],
        target_opset=17,
    )
    path.write_bytes(converted.SerializeToString())


def _dump_baseline_onnx(candidate: str, path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper

    feature_index = {
        "persistence": FEATURE_NAMES.index("watch_load_raw"),
        "rolling_median_300": FEATURE_NAMES.index("watch_load_median_300"),
    }.get(candidate)
    if feature_index is None:
        raise ValueError(f"baseline ONNX is not defined for candidate: {candidate}")
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
        [node], "observational_standard_candidate_v2", [input_info], [output_info], [index]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    path.write_bytes(model.SerializeToString())


def train_observational_standard(
    input_path: Path,
    output_root: Path,
    *,
    source_domain: Literal["real_observed", "synthetic_truth_oracle"],
    random_state: int = 20260804,
    hgb_max_iter: int = 120,
    candidate_name: str | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Train candidates under v2 without using locked-test rows for selection."""

    if source_domain not in SOURCE_DOMAINS:
        raise ValueError(f"unsupported source_domain: {source_domain}")
    if candidate_name is not None and candidate_name not in ALL_CANDIDATES:
        raise ValueError(f"unsupported candidate_name: {candidate_name}")
    root = output_root.resolve()
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FileExistsError(f"observational artifact already exists: {root}")
    root.mkdir(parents=True, exist_ok=True)
    source = input_path.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    config = ObservationalStandardConfig(
        random_state=random_state,
        hgb_max_iter=hgb_max_iter,
    )
    config.validate()
    observed = pd.read_parquet(source)
    built = build_observational_frame(observed, config=config)
    split = make_purged_splits(built, purge_sec=config.purge_sec)
    split_columns = [
        "person_key",
        "session_id",
        "corrected_utc",
        "split_role",
        "standard_target",
        "target_valid",
        "feature_decisionable",
        "future_valid_fraction",
        "future_max_gap_seconds",
    ]
    split[split_columns].to_parquet(root / "split_assignments.parquet", index=False)
    baseline_columns = [
        "person_key",
        "session_id",
        "baseline_status",
        "baseline_start_index",
        "baseline_freeze_index",
        *(column for column in built.columns if column.startswith("baseline_center__")),
        *(column for column in built.columns if column.startswith("baseline_scale__")),
    ]
    built.groupby(["person_key", "session_id"], sort=True).head(1)[
        baseline_columns
    ].to_parquet(root / "baseline.parquet", index=False)
    schema = canonical_feature_schema()
    _write_json(root / "feature_schema.json", schema)

    train_rows = (split["split_role"] == "train")
    validation_rows = (split["split_role"] == "validation")
    train_count = int(train_rows.sum())
    validation_count = int(validation_rows.sum())
    status = "TRAINED"
    blockers: list[str] = []
    if train_count < config.min_train_rows:
        blockers.append("INSUFFICIENT_TRAIN_SUPPORT")
    if validation_count < config.min_validation_rows:
        blockers.append("INSUFFICIENT_VALIDATION_SUPPORT")
    metrics_rows: list[dict[str, object]] = []
    predictions = split.loc[
        validation_rows,
        ["person_key", "session_id", "corrected_utc", "standard_target"],
    ].copy()
    model_objects: dict[str, Pipeline] = {}
    candidate_selection_mode = "VALIDATION_COMPLEXITY"
    if not blockers:
        x_train = _numeric_features(built, train_rows)
        y_train = built.loc[train_rows, "standard_target"].to_numpy(dtype="float64")
        x_validation = _numeric_features(built, validation_rows)
        y_validation = built.loc[validation_rows, "standard_target"].to_numpy(
            dtype="float64"
        )
        candidate_predictions: dict[str, np.ndarray] = {
            "persistence": built.loc[validation_rows, "watch_load_raw"].to_numpy(
                dtype="float64"
            ),
            "rolling_median_300": built.loc[
                validation_rows, "watch_load_median_300"
            ].to_numpy(dtype="float64"),
        }
        for name, model in _candidate_models(config).items():
            model.fit(x_train, y_train)
            model_objects[name] = model
            candidate_predictions[name] = np.asarray(
                model.predict(x_validation), dtype="float64"
            )
        for name in ALL_CANDIDATES:
            metrics = _regression_metrics(y_validation, candidate_predictions[name])
            metrics_rows.append({"candidate": name, "role": "validation", **metrics})
            predictions[f"prediction__{name}"] = candidate_predictions[name]
        metrics_by_name = {str(row["candidate"]): row for row in metrics_rows}
        persistence_rmse = float(cast(float, metrics_by_name["persistence"]["rmse"]))
        eligible_candidates = []
        for name in ("rolling_median_300", *LEARNED_CANDIDATES):
            rmse = float(cast(float, metrics_by_name[name]["rmse"]))
            if (
                math.isfinite(rmse)
                and math.isfinite(persistence_rmse)
                and rmse <= persistence_rmse * (1.0 - config.min_relative_improvement)
            ):
                eligible_candidates.append(name)
        if candidate_name is not None:
            if candidate_name in LEARNED_CANDIDATES and candidate_name not in model_objects:
                raise ValueError(f"candidate model was not fitted: {candidate_name}")
            selected_candidate = candidate_name
            candidate_selection_mode = "PINNED_CANDIDATE"
        elif eligible_candidates:
            selected_candidate = eligible_candidates[0]
        else:
            selected_candidate = "persistence"
            status = "NO_MODEL_BEATS_PERSISTENCE"
            blockers.append("NO_CANDIDATE_BEATS_PERSISTENCE")
    else:
        selected_candidate = "none"
        status = "INSUFFICIENT_SUPPORT"
        predictions["prediction__persistence"] = np.nan
        predictions["prediction__rolling_median_300"] = np.nan
    pd.DataFrame(metrics_rows).to_parquet(root / "metrics_validation.parquet", index=False)
    predictions.to_parquet(root / "predictions_validation.parquet", index=False)

    onnx_status = "NOT_APPLICABLE"
    unknown_types: list[str] = []
    unsafe_types: list[str] = []
    if selected_candidate in {"persistence", "rolling_median_300"}:
        _dump_baseline_onnx(selected_candidate, root / "model.onnx")
        onnx_status = "EXPORTED_BASELINE"
    elif selected_candidate in model_objects:
        model = model_objects[selected_candidate]
        sio.dump(model, root / "model.skops")
        unknown_types = sorted(sio.get_untrusted_types(file=root / "model.skops"))
        unsafe_types = [
            value for value in unknown_types if value not in SAFE_SKOPS_UNKNOWN_TYPES
        ]
        if unsafe_types:
            onnx_status = "BLOCKED_UNKNOWN_TYPES"
            blockers.append("SKOPS_UNKNOWN_TYPES_PRESENT")
        else:
            try:
                _dump_onnx(model, root / "model.onnx")
                onnx_status = "EXPORTED"
            except Exception as error:  # pragma: no cover
                onnx_status = f"EXPORT_FAILED:{type(error).__name__}"
                blockers.append("ONNX_EXPORT_FAILED")

    file_hashes = {
        path.name: _sha256_file(path)
        for path in sorted(root.iterdir())
        if path.name != "manifest.json" and path.is_file()
    }
    manifest: dict[str, object] = {
        "schema_version": OBSERVATIONAL_STANDARD_SCHEMA,
        "model_version": MODEL_VERSION,
        "status": status,
        "source_domain": source_domain,
        "data_scope": (
            "oracle/sanity" if source_domain == "synthetic_truth_oracle" else "observed/neon"
        ),
        "real_data_status": "NOT VERIFIED",
        "forecast_metric_status": "MEASURED" if metrics_rows else "NOT VERIFIED",
        "behavior_accuracy_status": "NOT VERIFIED",
        "stage": None,
        "locked_test_read": False,
        "source_file_sha256": _sha256_file(source),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_hash": FEATURE_SCHEMA_SHA256,
        "feature_names": list(FEATURE_NAMES),
        "legacy_15_feature_status": LEGACY_15_FEATURE_STATUS,
        "baseline_version": BASELINE_VERSION,
        "load_formula_version": LOAD_FORMULA_VERSION,
        "quality_formula_version": QUALITY_FORMULA_VERSION,
        "target_version": TARGET_VERSION,
        "target_contract": {
            "name": "future_mean_watch_load",
            "horizon_sec": config.forecast_horizon_sec,
            "definition": "mean watch_load_raw over t+1..t+1800 eligible seconds",
            "quality_gate": config.quality_gate,
            "min_future_valid_fraction": config.min_valid_fraction,
            "max_consecutive_gap_sec": config.max_gap_sec,
            "quality_multiplication": False,
            "separate_from_feature_decisionable": True,
        },
        "baseline_contract": {
            "eligible_seconds": config.baseline_warmup_sec,
            "max_wall_seconds": config.baseline_max_span_sec,
            "max_consecutive_gap_sec": config.max_gap_sec,
            "decision_total_eligible_seconds": config.decision_warmup_sec,
            "required_core": list(SIGNAL_COLUMNS),
            "clock": "corrected UTC only",
        },
        "split_contract": {
            "strategy": "person_group when multiple people; session chronological otherwise",
            "purge_sec": config.purge_sec,
            "target_crossing_rows_excluded": True,
            "roles": {
                role: int((split["split_role"] == role).sum())
                for role in ("train", "validation", "locked_test", "purged")
            },
        },
        "candidate_models": list(ALL_CANDIDATES),
        "selection_rule": {
            "baseline": "persistence",
            "minimum_relative_rmse_improvement": config.min_relative_improvement,
            "complexity_order": ["rolling_median_300", *LEARNED_CANDIDATES],
            "locked_test_used_for_selection": False,
        },
        "selected_candidate": selected_candidate,
        "candidate_selection_mode": candidate_selection_mode,
        "metrics_validation": metrics_rows,
        "unknown_skops_types": unknown_types,
        "allowed_skops_unknown_types": sorted(SAFE_SKOPS_UNKNOWN_TYPES),
        "unsafe_skops_unknown_types": unsafe_types,
        "onnx_status": onnx_status,
        "blockers": sorted(set(blockers)),
        "artifacts": file_hashes,
    }
    _write_json(root / "manifest.json", manifest)
    return manifest


def _candidate_runtime() -> dict[str, object]:
    """Capture the CPU-only runtime versions required for a shadow bundle."""

    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "numpy",
            "pandas",
            "scikit-learn",
            "onnx",
            "onnxruntime",
            "skl2onnx",
            "skops",
        )
    }
    return {
        "python": "3.12",
        "packages": versions,
        "onnx_providers": ["CPUExecutionProvider"],
        "onnx_intra_op_threads": 1,
        "onnx_inter_op_threads": 1,
    }


def export_candidate_shadow_bundle(
    input_path: Path,
    output_root: Path,
    *,
    active_bundle: Path | None = None,
    candidate_name: str = "ridge",
    source_domain: Literal["real_observed", "synthetic_truth_oracle"] = (
        "synthetic_truth_oracle"
    ),
    random_state: int = 20260806,
    hgb_max_iter: int = 120,
) -> dict[str, object]:
    """Create an immutable, CPU-only learned candidate beside the ACTIVE bundle.

    This function intentionally has no threshold fitting or locked-test read.  The
    copied golden input is the same canonical 16-vector fixture used by ACTIVE;
    only the candidate prediction is added to the candidate-side fixture.
    """

    if candidate_name not in LEARNED_CANDIDATES:
        raise ValueError(
            "candidate shadow must pin a learned CPU candidate: "
            + ", ".join(LEARNED_CANDIDATES)
        )
    root = output_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"candidate shadow bundle already exists: {root}")
    root.mkdir(parents=True, exist_ok=True)
    active_root = (
        active_bundle
        or root.parent / "observational-standard-30m-watch-baseline-shadow-v1"
    ).resolve()
    active_manifest_path = active_root / "manifest.json"
    active_model_path = active_root / "model.onnx"
    active_golden_path = active_root / "golden_fixture.json"
    if not active_manifest_path.is_file() or not active_model_path.is_file():
        raise FileNotFoundError(f"ACTIVE bundle is incomplete: {active_root}")
    if not active_golden_path.is_file():
        raise FileNotFoundError(f"ACTIVE golden fixture is missing: {active_golden_path}")
    active_verification = verify_baseline_shadow_bundle(active_root)
    active_manifest = json.loads(active_manifest_path.read_text(encoding="utf-8"))

    trained = train_observational_standard(
        input_path,
        root,
        source_domain=source_domain,
        random_state=random_state,
        hgb_max_iter=hgb_max_iter,
        candidate_name=candidate_name,
        overwrite=False,
    )
    if trained.get("selected_candidate") != candidate_name:
        raise ValueError(
            "candidate training did not pin the requested model: "
            f"{trained.get('selected_candidate')}"
        )
    model_path = root / "model.onnx"
    if not model_path.is_file():
        raise ValueError("candidate training did not produce model.onnx")

    # ACTIVE and CANDIDATE must consume byte-identical raw rows and ordered
    # features.  Keep the copied fixture immutable and record both outputs.
    shutil.copy2(active_golden_path, root / "golden_fixture.json")
    golden = json.loads((root / "golden_fixture.json").read_text(encoding="utf-8"))
    expected_vector = np.asarray(
        golden["expected_feature_vector"], dtype="float32"
    ).reshape(1, len(FEATURE_NAMES))
    import onnxruntime as ort  # type: ignore[import-untyped]

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    active_session = ort.InferenceSession(
        str(active_model_path),
        providers=["CPUExecutionProvider"],
        sess_options=options,
    )
    active_prediction = float(
        np.asarray(active_session.run(None, {"features": expected_vector})[0])
        .reshape(-1)[0]
    )
    if not math.isclose(
        active_prediction, float(golden["expected_prediction"]), abs_tol=1e-7
    ):
        raise ValueError("ACTIVE golden prediction is not reproducible")
    candidate_session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
        sess_options=options,
    )
    candidate_output_name = candidate_session.get_outputs()[0].name
    candidate_prediction = float(
        np.asarray(candidate_session.run(None, {"features": expected_vector})[0])
        .reshape(-1)[0]
    )
    golden["expected_candidate_prediction"] = candidate_prediction
    golden["candidate_output_name"] = candidate_output_name
    golden["candidate_algorithm"] = candidate_name
    _write_json(root / "golden_fixture.json", golden)
    _write_json(
        root / "candidate_golden_output.json",
        {
            "feature_schema_hash": FEATURE_SCHEMA_SHA256,
            "ordered_feature_names": list(FEATURE_NAMES),
            "input_name": "features",
            "output_name": candidate_output_name,
            "expected_prediction": candidate_prediction,
            "expected_stage": None,
        },
    )
    runtime = _candidate_runtime()
    _write_json(root / "runtime.json", runtime)
    package_versions = cast(dict[str, str], runtime["packages"])
    requirements = "\n".join(
        f"{name}=={package_versions[name]}"
        for name in (
            "numpy",
            "pandas",
            "scikit-learn",
            "onnx",
            "onnxruntime",
            "skl2onnx",
            "skops",
        )
    )
    (root / "runtime_requirements.txt").write_text(requirements + "\n", encoding="utf-8")
    _write_json(
        root / "training_evidence.json",
        {
            "source_file": str(input_path.resolve()),
            "source_file_sha256": _sha256_file(input_path.resolve()),
            "source_domain": source_domain,
            "data_scope": trained.get("data_scope"),
            "target_recomputed_from_observed_signals": True,
            "locked_test_read": False,
            "evaluation_label_status": "NOT_USED_FOR_THRESHOLD",
            "threshold_fit": False,
            "candidate_algorithm": candidate_name,
            "selection_mode": "PINNED_CANDIDATE",
            "validation_metrics": trained.get("metrics_validation", []),
        },
    )

    manifest = dict(trained)
    manifest.update(
        {
            "schema_version": "goal1.5/observational-standard-shadow-candidate/v1",
            "model_release": CANDIDATE_RELEASE,
            "release_status": "candidate",
            "model_role": "dual_shadow_continuous_standard_candidate",
            "candidate_algorithm": candidate_name,
            "input_name": "features",
            "input_dtype": "float32",
            "input_shape": [None, len(FEATURE_NAMES)],
            "output_name": candidate_output_name,
            "output_dtype": "float32",
            "output_shape": [None, 1],
            "model_artifact_sha256": _sha256_file(model_path),
            "real_data_status": "NOT VERIFIED",
            "stage": None,
            "pattern": None,
            "behavior": None,
            "thresholds": None,
            "calibration": {
                "t2": None,
                "t3": None,
                "t4": None,
                "t5": None,
                "threshold_state": "UNSET",
                "blocker": CANDIDATE_SELECTION_BLOCKER,
            },
            "continuous_output": {
                "name": "prediction_value",
                "dtype": "float32",
                "shape": [None, 1],
                "available_after_eligible_seconds": DECISION_WARMUP_SEC,
                "delivery": "shadow_ledger_only",
            },
            "delivery_eligible": False,
            "gate_status": "EXPLORATORY",
            "promotion_status": "NOT_RECOMMENDED",
            "active_reference": {
                "model_release": active_manifest.get("model_release"),
                "model_artifact_sha256": _sha256_file(active_model_path),
                "feature_schema_hash": FEATURE_SCHEMA_SHA256,
                "verification_status": active_verification.get("status"),
            },
            "locked_test_read": False,
            "threshold_fit_evaluation_labels": False,
        }
    )
    artifact_hashes = {
        path.name: _sha256_file(path)
        for path in sorted(root.iterdir())
        if path.is_file() and path.name not in {"manifest.json", "SHA256SUMS.json"}
    }
    manifest["artifacts"] = artifact_hashes
    _write_json(root / "manifest.json", manifest)
    checksums = {
        path.name: _sha256_file(path)
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "SHA256SUMS.json"
    }
    _write_json(root / "SHA256SUMS.json", checksums)
    return manifest


def verify_candidate_shadow_bundle(bundle: Path) -> dict[str, object]:
    """Verify candidate hashes, CPU ONNX shape, schema, and golden parity."""

    import onnx
    import onnxruntime as ort

    root = bundle.resolve()
    manifest_path = root / "manifest.json"
    checksums_path = root / "SHA256SUMS.json"
    if not manifest_path.is_file() or not checksums_path.is_file():
        raise FileNotFoundError(f"candidate bundle manifest/checksums missing: {root}")
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    for name, digest in checksums.items():
        path = root / str(name)
        if not path.is_file() or _sha256_file(path) != str(digest):
            raise ValueError(f"bundle checksum mismatch: {name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "model_release": CANDIDATE_RELEASE,
        "release_status": "candidate",
        "delivery_eligible": False,
        "real_data_status": "NOT VERIFIED",
        "stage": None,
        "thresholds": None,
        "locked_test_read": False,
        "feature_schema_hash": FEATURE_SCHEMA_SHA256,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"candidate manifest contract mismatch: {key}")
    calibration = manifest.get("calibration")
    if not isinstance(calibration, dict) or any(
        calibration.get(key) is not None for key in ("t2", "t3", "t4", "t5")
    ):
        raise ValueError("candidate thresholds must remain unset")
    if calibration.get("blocker") != CANDIDATE_SELECTION_BLOCKER:
        raise ValueError("candidate calibration blocker is missing")
    schema = json.loads((root / "feature_schema.json").read_text(encoding="utf-8"))
    if schema != canonical_feature_schema():
        raise ValueError("candidate feature schema differs from canonical schema")
    if hashlib.sha256(_canonical_json(schema)).hexdigest() != FEATURE_SCHEMA_SHA256:
        raise ValueError("candidate feature schema SHA-256 mismatch")
    for name, digest in manifest["artifacts"].items():
        if _sha256_file(root / str(name)) != str(digest):
            raise ValueError(f"artifact SHA-256 mismatch: {name}")
    model_path = root / "model.onnx"
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    input_tensor = model.graph.input[0]
    shape = input_tensor.type.tensor_type.shape
    if input_tensor.name != "features" or shape.dim[1].dim_value != len(FEATURE_NAMES):
        raise ValueError("candidate ONNX input is not float32[N,16] features")
    golden = json.loads((root / "golden_fixture.json").read_text(encoding="utf-8"))
    if golden.get("ordered_feature_names") != list(FEATURE_NAMES):
        raise ValueError("candidate golden feature order mismatch")
    expected_vector = np.asarray(
        golden["expected_feature_vector"], dtype="float32"
    ).reshape(1, len(FEATURE_NAMES))
    replayed = build_observational_frame(pd.DataFrame(golden["input_rows"]))
    if int(replayed.iloc[-1]["feature_decisionable"]) != 1:
        raise ValueError("candidate golden replay is not feature-decisionable")
    replayed_vector = ordered_feature_matrix(replayed.iloc[[-1]])
    if not np.allclose(replayed_vector, expected_vector, rtol=1e-6, atol=1e-7):
        raise ValueError("candidate golden feature vector mismatch")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"], sess_options=options
    )
    if session.get_inputs()[0].name != "features":
        raise ValueError("candidate ONNX input name mismatch")
    prediction = float(
        np.asarray(session.run(None, {"features": replayed_vector})[0]).reshape(-1)[0]
    )
    if not math.isclose(
        prediction, float(golden["expected_candidate_prediction"]), abs_tol=1e-6
    ):
        raise ValueError("candidate golden ONNX prediction mismatch")
    candidate_output = json.loads(
        (root / "candidate_golden_output.json").read_text(encoding="utf-8")
    )
    if not math.isclose(
        prediction, float(candidate_output["expected_prediction"]), abs_tol=1e-6
    ):
        raise ValueError("candidate output fixture mismatch")
    if not math.isfinite(prediction):
        raise ValueError("candidate prediction is not finite")
    skops_path = root / "model.skops"
    unknown_types = sorted(sio.get_untrusted_types(file=skops_path)) if skops_path.is_file() else []
    unsafe_types = [value for value in unknown_types if value not in SAFE_SKOPS_UNKNOWN_TYPES]
    if unsafe_types:
        raise ValueError(f"candidate skops contains unsafe unknown types: {unsafe_types}")
    return {
        "status": "VERIFIED",
        "model_release": manifest["model_release"],
        "candidate_algorithm": manifest["candidate_algorithm"],
        "model_artifact_sha256": _sha256_file(model_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "feature_schema_hash": FEATURE_SCHEMA_SHA256,
        "prediction_value": prediction,
        "threshold_state": calibration["threshold_state"],
        "delivery_eligible": manifest["delivery_eligible"],
    }


__all__ = [
    "BASELINE_MAX_SPAN_SEC",
    "BASELINE_VERSION",
    "BASELINE_WARMUP_SEC",
    "DECISION_WARMUP_SEC",
    "EMA_ALPHA_1800",
    "EMA_ALPHA_21600",
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_SHA256",
    "FEATURE_SCHEMA_VERSION",
    "FORECAST_HORIZON_SEC",
    "LEGACY_15_FEATURE_STATUS",
    "LOAD_FORMULA_VERSION",
    "MAX_GAP_SEC",
    "MIN_FUTURE_VALID_FRACTION",
    "MODEL_RELEASE",
    "MODEL_VERSION",
    "PURGE_SEC",
    "QUALITY_FORMULA_VERSION",
    "QUALITY_GATE",
    "TARGET_VERSION",
    "ObservationalStandardConfig",
    "bounded_ema_eligible",
    "build_observational_frame",
    "canonical_feature_schema",
    "export_baseline_shadow_bundle",
    "export_candidate_shadow_bundle",
    "make_purged_splits",
    "ordered_feature_matrix",
    "train_observational_standard",
    "verify_baseline_shadow_bundle",
    "verify_candidate_shadow_bundle",
]
