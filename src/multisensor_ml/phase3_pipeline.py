from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from multisensor_ml.contracts import (
    FORBIDDEN_ORACLE_EXACT,
    ORACLE_LATENT_FACTORS,
    assert_oracle_columns,
)
from multisensor_ml.cumulative_load import (
    build_cumulative_load,
    classify_load_state,
    fit_train_load_quantiles,
)
from multisensor_ml.features import build_causal_features
from multisensor_ml.models import select_training_rows
from multisensor_ml.phase3_baselines import (
    BaselineMode,
    build_causal_baseline,
    fit_train_global_context_baseline,
)
from multisensor_ml.phase3_contracts import Phase3Config
from multisensor_ml.phase3_evaluation import (
    evaluate_group_uplift,
    evaluate_pattern_predictions,
    select_smallest_stable_cap,
)


@dataclass(frozen=True)
class Phase3Outputs:
    root: Path
    candidate_metrics_parquet: Path
    group_uplift_parquet: Path
    person_predictions_parquet: Path
    baseline_diagnostics_parquet: Path
    cumulative_load_parquet: Path
    manifest_json: Path


def build_locked_identity_stubs(
    source: pd.DataFrame,
    locked_splits: pd.DataFrame,
) -> pd.DataFrame:
    """Create identity-only rows without reading locked-test sensor values."""

    identity_columns = (
        "dataset_id",
        "run_id",
        "person_id",
        "person_key",
        "split_role",
    )
    missing = sorted(set(identity_columns).difference(locked_splits.columns))
    if missing:
        raise ValueError(f"locked split identity missing columns: {missing}")
    stubs = source.iloc[:0].copy().reindex(range(len(locked_splits)))
    for column in identity_columns:
        stubs[column] = locked_splits[column].to_numpy()
    return stubs


def prepare_phase3_source(config: Phase3Config) -> Path:
    """Materialize the approved oracle whitelist and independent labels only."""

    series_registry = config.registry_root / config.series_id
    splits = pd.read_parquet(series_registry / "splits.parquet")
    allowed = splits.loc[
        splits["split_role"].isin(["train", "validation"])
    ].copy()
    if allowed["person_key"].nunique() != 30:
        raise ValueError("Phase 3 source requires exactly 30 unlocked people")
    outcomes = config.outcome_root / config.series_id
    stage_path = outcomes / "outcome_stages.parquet"
    event_path = outcomes / "outcome_events.parquet"
    run_frames: list[pd.DataFrame] = []
    latent_columns = [
        "run_id",
        "person_id",
        "timestamp_utc",
        "date_utc",
        "day_index",
        "context_state",
        "is_awake",
        *ORACLE_LATENT_FACTORS,
    ]
    for run_id, run_split in allowed.groupby("run_id", sort=True):
        people = run_split["person_id"].astype(str).tolist()
        latent_path = (
            config.raw_root
            / config.series_id
            / "runs"
            / str(run_id)
            / "truth"
            / "latent_timeline.parquet"
        )
        latent = pd.read_parquet(
            latent_path,
            columns=latent_columns,
            filters=[("person_id", "in", people)],
        )
        stages = pd.read_parquet(
            stage_path,
            columns=["run_id", "person_id", "timestamp_utc", "stage_code"],
            filters=[("run_id", "=", str(run_id)), ("person_id", "in", people)],
        )
        events = pd.read_parquet(
            event_path,
            columns=[
                "run_id",
                "person_id",
                "is_target",
                "start_time_ns",
                "end_time_ns",
            ],
            filters=[("run_id", "=", str(run_id)), ("person_id", "in", people)],
        )
        latent = latent.merge(
            stages,
            on=["run_id", "person_id", "timestamp_utc"],
            how="left",
            validate="one_to_one",
        )
        latent["event_binary"] = latent["stage_code"].fillna("NO_EVENT").ne("NO_EVENT")
        latent["hard_negative"] = False
        timestamp_ns = (
            pd.to_datetime(latent["timestamp_utc"], utc=True)
            .dt.as_unit("ns")
            .astype("int64")
        )
        hard_events = events.loc[~events["is_target"].astype(bool)]
        for event in hard_events.itertuples(index=False):
            mask = (
                latent["person_id"].eq(event.person_id)
                & timestamp_ns.ge(int(cast(int, event.start_time_ns)))
                & timestamp_ns.le(int(cast(int, event.end_time_ns)))
            )
            latent.loc[mask, "hard_negative"] = True
        latent = latent.merge(
            run_split.loc[
                :,
                [
                    "dataset_id",
                    "run_id",
                    "person_id",
                    "person_key",
                    "split_role",
                ],
            ],
            on=["run_id", "person_id"],
            how="left",
            validate="many_to_one",
        )
        latent["series_id"] = config.series_id
        latent["canonical_time"] = latent["timestamp_utc"]
        latent["day_key"] = latent["date_utc"].astype(str)
        latent["session_id"] = (
            latent["person_key"].astype(str)
            + ":day-"
            + latent["day_index"].astype(str)
        )
        latent = latent.rename(columns={"context_state": "context"})
        run_frames.append(
            latent.loc[
                :,
                [
                    "series_id",
                    "dataset_id",
                    "run_id",
                    "person_id",
                    "person_key",
                    "day_key",
                    "session_id",
                    "timestamp_utc",
                    "canonical_time",
                    "context",
                    "is_awake",
                    "event_binary",
                    "hard_negative",
                    "split_role",
                    *ORACLE_LATENT_FACTORS,
                ],
            ]
        )
    source = pd.concat(run_frames, ignore_index=True)
    source_root = config.artifact_root / "source"
    source_root.mkdir(parents=True, exist_ok=False)
    source_path = source_root / "phase3_source.parquet"
    source.to_parquet(source_path, index=False)
    manifest = {
        "schema_version": "goal1.5/phase3-source/v1",
        "series_id": config.series_id,
        "data_status": "oracle/sanity",
        "locked_test_read": False,
        "rows": len(source),
        "people": int(source["person_key"].nunique()),
        "split_person_counts": source.groupby("split_role")["person_key"]
        .nunique()
        .astype(int)
        .to_dict(),
        "columns": list(source.columns),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    }
    (source_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return source_path


def run_phase3_from_config(config: Phase3Config) -> Phase3Outputs:
    source_path = config.artifact_root / "source" / "phase3_source.parquet"
    if not source_path.exists():
        raise FileNotFoundError("run 'phase3 prepare' before train-validate")
    source = pd.read_parquet(source_path)
    locked_splits = pd.read_parquet(
        config.registry_root / config.series_id / "splits.parquet"
    ).loc[lambda value: value["split_role"].eq("locked_test")]
    locked_stub = build_locked_identity_stubs(source, locked_splits)
    combined = pd.concat([source, locked_stub], ignore_index=True)
    return run_phase3_experiment(
        combined,
        config.artifact_root / "result",
        warmup_sec=config.warmup_sec,
        lookback_sec=config.lookback_sec,
        refresh_sec=config.refresh_sec,
        horizons_sec=config.cumulative_horizons_sec,
        random_state=config.random_state,
    )


def _sha256_payload(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_phase3_frame(frame: pd.DataFrame) -> None:
    required = {
        "series_id",
        "dataset_id",
        "run_id",
        "person_id",
        "person_key",
        "day_key",
        "session_id",
        "timestamp_utc",
        "canonical_time",
        "context",
        "is_awake",
        "event_binary",
        "hard_negative",
        "split_role",
        *ORACLE_LATENT_FACTORS,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Phase 3 input missing columns: {missing}")
    forbidden = sorted(set(frame.columns).intersection(FORBIDDEN_ORACLE_EXACT))
    forbidden.extend(
        column
        for column in frame.columns
        if column.startswith("active_target_")
        or column.startswith("event_intensity")
    )
    if forbidden:
        raise ValueError(f"Phase 3 input contains truth leakage: {sorted(set(forbidden))}")
    assert_oracle_columns(list(frame.columns))
    if not set(frame["split_role"].unique()).issubset(
        {"train", "validation", "locked_test"}
    ):
        raise ValueError("Phase 3 input has an unknown split role")
    if frame["person_key"].isna().any():
        raise ValueError("Phase 3 person identity cannot be null")


def _sampling_hash(frame: pd.DataFrame, selected: NDArray[np.bool_]) -> str:
    identities = (
        frame.loc[selected, ["dataset_id", "canonical_time"]]
        .astype(str)
        .agg("|".join, axis=1)
        .sort_values(kind="mergesort")
        .tolist()
    )
    return _sha256_payload(identities)


def _build_features(
    frame: pd.DataFrame,
    global_baseline: pd.DataFrame,
    *,
    mode: BaselineMode,
    weight_cap: float,
    warmup_sec: int,
    lookback_sec: int,
    refresh_sec: int,
    horizons_sec: tuple[int, ...],
    windows: tuple[int, ...],
    lags: tuple[int, ...],
    include_load: bool,
    load_adjusted_baseline: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    baseline = build_causal_baseline(
        frame,
        global_baseline,
        mode=mode,
        weight_cap=weight_cap,
        warmup_sec=warmup_sec,
        lookback_sec=lookback_sec,
        refresh_sec=refresh_sec,
    )
    initial_features, _ = build_causal_features(
        frame,
        baseline,
        windows=windows,
        lags=lags,
    )
    robust_z = pd.DataFrame(
        {
            factor: initial_features[f"{factor}__robust_z"].astype("float64")
            for factor in ORACLE_LATENT_FACTORS
        }
    )
    load: pd.DataFrame | None = None
    if include_load or load_adjusted_baseline:
        load = build_cumulative_load(
            frame,
            robust_z,
            horizons_sec=horizons_sec,
        )
    if load_adjusted_baseline:
        if load is None:
            raise AssertionError("load-adjusted baseline requires load state")
        longest = max(horizons_sec)
        label = {
            1800: "30m",
            21600: "6h",
            86400: "24h",
            259200: "72h",
        }.get(longest, f"{longest}s")
        influence = np.tanh(load[f"cumulative_load_{label}"].to_numpy()) * 0.10
        adjusted = baseline.copy()
        for factor in ORACLE_LATENT_FACTORS:
            if factor in {"motor_activation", "social_context"}:
                continue
            direction = -1.0 if factor == "recovery_capacity" else 1.0
            adjusted[f"{factor}__center"] = (
                adjusted[f"{factor}__center"]
                + direction * adjusted[f"{factor}__mad"] * influence
            )
        baseline = adjusted
        initial_features, _ = build_causal_features(
            frame,
            baseline,
            windows=windows,
            lags=lags,
        )
    if include_load:
        if load is None:
            raise AssertionError("load features were not built")
        load_columns = [
            column
            for column in load.columns
            if column.startswith("cumulative_load_")
            and not column.endswith("_mature")
        ]
        if "load_slope_30m" in load:
            load_columns.append("load_slope_30m")
        for column in load_columns:
            initial_features[column] = load[column].astype("float32")
    return initial_features, baseline, load


def _new_model(random_state: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        class_weight="balanced",
        learning_rate=0.08,
        max_iter=80,
        max_leaf_nodes=31,
        min_samples_leaf=5,
        random_state=random_state,
    )


def _select_threshold(frame: pd.DataFrame) -> float:
    candidates = np.linspace(0.1, 0.9, 17)
    scores: list[tuple[float, float, float]] = []
    for threshold in candidates:
        metrics = evaluate_pattern_predictions(frame, threshold=float(threshold))
        scores.append(
            (
                metrics["event_f1"],
                -metrics["false_alerts_per_hour"],
                float(threshold),
            )
        )
    return max(scores)[2]


def _oof_candidate(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    selected: NDArray[np.bool_],
    *,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    train_frame = frame.loc[selected].reset_index(drop=True)
    train_features = features.loc[selected].reset_index(drop=True)
    groups = train_frame["person_key"].astype(str).to_numpy()
    probabilities = np.zeros(len(train_frame), dtype="float64")
    splitter = GroupKFold(n_splits=min(3, len(np.unique(groups))))
    target = train_frame["event_binary"].to_numpy(dtype=np.int8)
    for fold, (fit_index, predict_index) in enumerate(
        splitter.split(train_features, target, groups)
    ):
        model = _new_model(random_state + fold)
        model.fit(train_features.iloc[fit_index], target[fit_index])
        probabilities[predict_index] = model.predict_proba(
            train_features.iloc[predict_index]
        )[:, 1]
    predictions = train_frame.loc[
        :,
        [
            "dataset_id",
            "person_key",
            "day_key",
            "session_id",
            "canonical_time",
            "event_binary",
        ],
    ].copy()
    predictions["probability"] = probabilities
    threshold = _select_threshold(predictions)
    metrics = evaluate_pattern_predictions(predictions, threshold=threshold)
    metrics["threshold"] = threshold
    return predictions, metrics


def _fit_validation_candidate(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    selected: NDArray[np.bool_],
    *,
    threshold: float,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    train_mask = frame["split_role"].eq("train").to_numpy() & selected
    validation_mask = frame["split_role"].eq("validation").to_numpy()
    model = _new_model(random_state)
    model.fit(
        features.loc[train_mask],
        frame.loc[train_mask, "event_binary"].to_numpy(dtype=np.int8),
    )
    probabilities = model.predict_proba(features.loc[validation_mask])[:, 1]
    columns = [
        "dataset_id",
        "run_id",
        "person_key",
        "day_key",
        "session_id",
        "canonical_time",
        "context",
        "event_binary",
        "split_role",
    ]
    predictions = frame.loc[validation_mask, columns].reset_index(drop=True)
    predictions["probability"] = probabilities
    metrics = evaluate_pattern_predictions(predictions, threshold=threshold)
    metrics["threshold"] = threshold
    return predictions, metrics


def _candidate_spec(candidate_id: str) -> tuple[BaselineMode, float, bool, bool]:
    if candidate_id == "G0":
        return "global_context", 0.0, False, False
    if candidate_id.startswith("P1-"):
        return "personal_pooled", int(candidate_id[-3:]) / 100, False, False
    if candidate_id.startswith("P2-"):
        return "personal_context", int(candidate_id[-3:]) / 100, False, False
    if candidate_id == "P2+CL":
        return "personal_context", 0.5, True, False
    if candidate_id == "P2+CL-B":
        return "personal_context", 0.5, True, True
    raise ValueError(f"unknown Phase 3 candidate: {candidate_id}")


def run_phase3_experiment(
    frame: pd.DataFrame,
    output_root: Path,
    *,
    warmup_sec: int = 1800,
    lookback_sec: int = 21600,
    refresh_sec: int = 60,
    horizons_sec: tuple[int, ...] = (1800, 21600, 86400, 259200),
    random_state: int = 20260730,
    windows: tuple[int, ...] = (5, 15, 30, 60, 180, 300),
    lags: tuple[int, ...] = (1, 5, 15, 30, 60),
) -> Phase3Outputs:
    """Run a bounded Phase 3 ablation without opening locked-test rows."""

    _validate_phase3_frame(frame)
    split_people = (
        frame.loc[:, ["person_key", "split_role"]]
        .drop_duplicates()
        .groupby("split_role")["person_key"]
        .nunique()
        .to_dict()
    )
    if split_people != {"locked_test": 6, "train": 24, "validation": 6}:
        raise ValueError(f"Phase 3 requires exact 24/6/6 people: {split_people}")
    work = frame.loc[frame["split_role"].isin(["train", "validation"])].copy()
    work = work.sort_values(
        ["person_key", "canonical_time"],
        kind="mergesort",
    ).reset_index(drop=True)
    work["timestamp_utc"] = pd.to_datetime(work["timestamp_utc"], utc=True)
    work["canonical_time"] = pd.to_datetime(work["canonical_time"], utc=True)
    global_baseline = fit_train_global_context_baseline(
        work.loc[work["split_role"].eq("train")]
    )
    train_work = work.loc[work["split_role"].eq("train")].reset_index(drop=True)
    train_mask = work["split_role"].eq("train").to_numpy()
    train_selection = select_training_rows(
        train_work,
        target="event_binary",
    )
    selected = np.zeros(len(work), dtype=bool)
    selected[np.flatnonzero(train_mask)] = train_selection
    sampling_hash = _sampling_hash(work, selected)

    oof_rows: list[dict[str, object]] = []
    for family, mode in (("P1", "personal_pooled"), ("P2", "personal_context")):
        for cap in (0.25, 0.5, 1.0):
            candidate_id = f"{family}-{int(cap * 100):03d}"
            candidate = _build_features(
                train_work,
                global_baseline,
                mode=cast(BaselineMode, mode),
                weight_cap=cap,
                warmup_sec=warmup_sec,
                lookback_sec=lookback_sec,
                refresh_sec=refresh_sec,
                horizons_sec=horizons_sec,
                windows=windows,
                lags=lags,
                include_load=False,
                load_adjusted_baseline=False,
            )
            _, metrics = _oof_candidate(
                train_work,
                candidate[0],
                train_selection,
                random_state=random_state,
            )
            oof_rows.append(
                {
                    "candidate_id": candidate_id,
                    "weight_cap": cap,
                    **metrics,
                }
            )
    oof_metrics = pd.DataFrame(oof_rows)
    selected_p1 = select_smallest_stable_cap(
        oof_metrics.loc[oof_metrics["candidate_id"].str.startswith("P1")]
    )
    selected_p2 = select_smallest_stable_cap(
        oof_metrics.loc[oof_metrics["candidate_id"].str.startswith("P2")]
    )

    candidate_ids = ["G0", selected_p1, selected_p2, "P2+CL", "P2+CL-B"]
    prediction_frames: dict[str, pd.DataFrame] = {}
    metric_rows: list[dict[str, object]] = []
    baseline_rows: list[pd.DataFrame] = []
    common_load: pd.DataFrame | None = None
    p2_cap = _candidate_spec(selected_p2)[1]
    for candidate_id in candidate_ids:
        mode, cap, include_load, load_adjusted = _candidate_spec(candidate_id)
        if candidate_id.startswith("P2+"):
            cap = p2_cap
        features, baseline, load = _build_features(
            work,
            global_baseline,
            mode=mode,
            weight_cap=cap,
            warmup_sec=warmup_sec,
            lookback_sec=lookback_sec,
            refresh_sec=refresh_sec,
            horizons_sec=horizons_sec,
            windows=windows,
            lags=lags,
            include_load=include_load,
            load_adjusted_baseline=load_adjusted,
        )
        _, oof = _oof_candidate(
            work,
            features,
            selected,
            random_state=random_state,
        )
        threshold = oof["threshold"]
        predictions, metrics = _fit_validation_candidate(
            work,
            features,
            selected,
            threshold=threshold,
            random_state=random_state,
        )
        predictions["candidate_id"] = candidate_id
        prediction_frames[candidate_id] = predictions
        metric_rows.append(
            {
                "candidate_id": candidate_id,
                "sampling_hash": sampling_hash,
                "oof_aucpr": oof["aucpr"],
                "oof_event_recall": oof["event_recall"],
                **metrics,
            }
        )
        diagnostics = (
            baseline.assign(person_key=work["person_key"])
            .groupby("person_key", as_index=False)
            .agg(
                personal_weight_mean=("personal_weight", "mean"),
                personal_weight_max=("personal_weight", "max"),
                n_eff_max=("n_eff", "max"),
            )
        )
        diagnostics["candidate_id"] = candidate_id
        baseline_rows.append(diagnostics)
        if candidate_id == "P2+CL":
            common_load = load

    if common_load is None:
        raise AssertionError("P2+CL did not produce cumulative load")
    quantile_input = common_load.copy()
    quantile_input["split_role"] = work["split_role"].to_numpy()
    load_state_column = next(
        column
        for column in common_load.columns
        if column.startswith("cumulative_load_") and not column.endswith("_mature")
    )
    quantiles = fit_train_load_quantiles(
        quantile_input,
        column=load_state_column,
    )
    load_state = classify_load_state(
        common_load,
        quantiles,
        column=load_state_column,
    )
    for predictions in prediction_frames.values():
        validation_mask = work["split_role"].eq("validation").to_numpy()
        predictions["load_state"] = load_state.loc[validation_mask].reset_index(drop=True)

    g0 = prediction_frames["G0"]
    threshold_by_candidate = {
        str(row["candidate_id"]): float(cast(float, row["threshold"]))
        for row in metric_rows
    }
    uplift_frames: list[pd.DataFrame] = []
    for candidate_id in candidate_ids[1:]:
        uplift = evaluate_group_uplift(
            g0,
            prediction_frames[candidate_id],
            group_columns=(
                "split_role",
                "run_id",
                "dataset_id",
                "person_key",
                "context",
                "load_state",
            ),
            threshold=threshold_by_candidate[candidate_id],
        )
        uplift["candidate_id"] = candidate_id
        uplift_frames.append(uplift)

    output_root.mkdir(parents=True, exist_ok=False)
    outputs = Phase3Outputs(
        root=output_root,
        candidate_metrics_parquet=output_root / "phase3_candidate_metrics.parquet",
        group_uplift_parquet=output_root / "phase3_group_uplift.parquet",
        person_predictions_parquet=output_root / "phase3_person_predictions.parquet",
        baseline_diagnostics_parquet=output_root
        / "phase3_baseline_diagnostics.parquet",
        cumulative_load_parquet=output_root / "phase3_cumulative_load.parquet",
        manifest_json=output_root / "phase3_manifest.json",
    )
    pd.DataFrame(metric_rows).to_parquet(outputs.candidate_metrics_parquet, index=False)
    pd.concat(uplift_frames, ignore_index=True).to_parquet(
        outputs.group_uplift_parquet,
        index=False,
    )
    pd.concat(prediction_frames.values(), ignore_index=True).to_parquet(
        outputs.person_predictions_parquet,
        index=False,
    )
    pd.concat(baseline_rows, ignore_index=True).to_parquet(
        outputs.baseline_diagnostics_parquet,
        index=False,
    )
    load_export = work.loc[
        :,
        ["dataset_id", "run_id", "person_key", "canonical_time", "split_role"],
    ].copy()
    for raw_column in common_load:
        column = str(raw_column)
        if column.startswith("cumulative_load_") or column == "load_slope_30m":
            load_export[column] = common_load[column]
    load_export["load_state"] = load_state
    load_export.to_parquet(outputs.cumulative_load_parquet, index=False)
    manifest = {
        "schema_version": "goal1.5/phase3-result/v1",
        "series_id": str(work["series_id"].iloc[0]),
        "data_status": "oracle/sanity",
        "real_accuracy_status": "NOT VERIFIED",
        "device_synchronization_status": "NOT_AVAILABLE_TRUTH_ONLY",
        "locked_test_read": False,
        "split_person_counts": {"train": 24, "validation": 6},
        "candidate_ids": candidate_ids,
        "selected_p1": selected_p1,
        "selected_p2": selected_p2,
        "sampling_hash": sampling_hash,
        "source_hash": _sha256_payload(
            work.loc[:, ["dataset_id", "canonical_time", "split_role"]].astype(str).values.tolist()
        ),
        "load_quantiles_train_only": quantiles,
        "baseline_load_influence_cap": 0.10,
    }
    outputs.manifest_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return outputs
