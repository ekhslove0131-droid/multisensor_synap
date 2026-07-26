from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import precision_recall_curve

from multisensor_ml.baseline import fit_global_baseline, personalize_baseline
from multisensor_ml.bundle import ModelKey, load_model_bundle, write_model_bundle
from multisensor_ml.features import build_causal_features
from multisensor_ml.metrics import (
    evaluate_probabilities,
    forecast_lead_times,
    select_event_threshold,
)
from multisensor_ml.models import (
    ProbabilityClassifier,
    fit_candidate_models,
    select_training_rows,
)
from multisensor_ml.oracle import read_oracle_person
from multisensor_ml.registry import sha256_file
from multisensor_ml.stress import StressScenario, apply_stress

TARGETS: tuple[str, ...] = ("event_binary", "forecast_60s")


@dataclass(frozen=True, slots=True)
class PreparedSeries:
    root: Path
    manifest_json: Path
    global_baseline_json: Path
    personal_baseline_parquet: Path


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False),
        path,
        compression="zstd",
    )


def _load_registry_runs(registry_dir: Path) -> dict[str, Path]:
    manifest = json.loads((registry_dir / "manifest.json").read_text(encoding="utf-8"))
    return {str(row["run_id"]): Path(str(row["source_path"])) for row in manifest["runs"]}


def prepare_series(registry_dir: Path, output_root: Path) -> PreparedSeries:
    """Prepare per-person causal features while preserving person split isolation."""

    root = output_root.resolve()
    if root.exists():
        raise FileExistsError(f"prepared series already exists: {root}")
    root.mkdir(parents=True)
    split = pq.read_table(registry_dir / "splits.parquet").to_pandas()
    runs = _load_registry_runs(registry_dir)

    global_samples: list[pd.DataFrame] = []
    for row in split.loc[split["split_role"] == "train"].itertuples(index=False):
        oracle = read_oracle_person(runs[str(row.run_id)], str(row.person_id))
        global_samples.append(
            oracle.frame.loc[
                ::60,
                ["context", *cast(tuple[str, ...], tuple(oracle.feature_source_columns[3:]))],
            ]
        )
    global_baseline = fit_global_baseline(pd.concat(global_samples, ignore_index=True))
    global_path = root / "global_baseline.json"
    global_path.write_text(
        json.dumps(global_baseline.to_dict(orient="records"), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    entries: list[dict[str, object]] = []
    personal_refreshes: list[pd.DataFrame] = []
    feature_names: list[str] | None = None
    for row in split.sort_values(["run_id", "person_id"]).itertuples(index=False):
        run_id = str(row.run_id)
        person_id = str(row.person_id)
        person_key = str(row.person_key)
        oracle = read_oracle_person(runs[run_id], person_id)
        baseline = personalize_baseline(oracle.frame, global_baseline)
        features, names = build_causal_features(oracle.frame, baseline)
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("feature schema changed between people")
        prepared = pd.concat(
            [
                pd.DataFrame(
                    {
                        "person_key": person_key,
                        "run_id": run_id,
                        "person_id": person_id,
                        "timestamp_utc": oracle.frame["timestamp_utc"],
                        "context": oracle.frame["context"],
                        "phase": oracle.frame["phase"],
                        "event_binary": oracle.frame["event_binary"].astype("int8"),
                        "forecast_60s": oracle.frame["forecast_60s"].astype("int8"),
                        "hard_negative": oracle.frame["hard_negative"].astype("int8"),
                    }
                ),
                features,
            ],
            axis=1,
        )
        relative = Path("people") / f"{row.dataset_id}.parquet"
        _write_parquet(prepared, root / relative)
        refresh = baseline.loc[baseline["is_refresh"]].copy()
        refresh.insert(0, "person_key", person_key)
        personal_refreshes.append(refresh)
        entries.append(
            {
                "dataset_id": str(row.dataset_id),
                "run_id": run_id,
                "person_id": person_id,
                "person_key": person_key,
                "split_role": str(row.split_role),
                "logical_hash": str(row.logical_hash),
                "path": str(relative),
                "row_count": len(prepared),
            }
        )

    personal_path = root / "personal_baseline.parquet"
    _write_parquet(pd.concat(personal_refreshes, ignore_index=True), personal_path)
    if feature_names is None:
        raise ValueError("series contained no people")
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "prepared_schema": "goal1.5/prepared/v1",
                "status": "oracle/sanity",
                "real_data_status": "NOT VERIFIED",
                "source_registry": str(registry_dir.resolve()),
                "source_split_sha256": sha256_file(registry_dir / "splits.parquet"),
                "feature_names": feature_names,
                "people": entries,
                "global_baseline_sha256": sha256_file(global_path),
                "personal_baseline_sha256": sha256_file(personal_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return PreparedSeries(
        root=root,
        manifest_json=manifest_path,
        global_baseline_json=global_path,
        personal_baseline_parquet=personal_path,
    )


def _read_people(
    root: Path,
    entries: list[dict[str, object]],
    role: str,
) -> list[tuple[dict[str, object], pd.DataFrame]]:
    return [
        (entry, pq.read_table(root / str(entry["path"])).to_pandas())
        for entry in entries
        if entry["split_role"] == role
    ]


def _append_scalar_metrics(
    destination: list[dict[str, object]],
    metrics: dict[str, float | int | list[list[int]]],
    *,
    target: str,
    model_name: str,
    role: str,
    threshold: float,
    scenario: str = "clean",
    degradation: float | None = None,
) -> None:
    matrix = cast(list[list[int]], metrics["confusion_matrix"])
    for metric, value in metrics.items():
        if metric == "confusion_matrix":
            continue
        destination.append(
            {
                "target": target,
                "model_name": model_name,
                "role": role,
                "scenario": scenario,
                "metric": metric,
                "value": float(cast(float | int, value)),
                "threshold": threshold,
                "degradation": degradation,
            }
        )
    for metric, matrix_value in zip(
        ("tn", "fp", "fn", "tp"),
        np.asarray(matrix).ravel(),
        strict=True,
    ):
        destination.append(
            {
                "target": target,
                "model_name": model_name,
                "role": role,
                "scenario": scenario,
                "metric": metric,
                "value": float(cast(int, matrix_value)),
                "threshold": threshold,
                "degradation": degradation,
            }
        )


def _evaluate_people(
    people: list[tuple[dict[str, object], pd.DataFrame]],
    *,
    model: ProbabilityClassifier,
    feature_names: list[str],
    target: str,
    threshold: float,
    model_name: str,
    role: str,
) -> tuple[dict[str, float | int | list[list[int]]], list[pd.DataFrame]]:
    truths: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    retained: list[pd.DataFrame] = []
    person_recall: list[float] = []
    person_f1: list[float] = []
    lead_times: list[int] = []
    for entry, frame in people:
        feature_values = frame[feature_names].to_numpy(dtype=np.float32)
        truth = frame[target].to_numpy(dtype=np.int8)
        probability = model.predict_proba(feature_values)[:, 1]
        truths.append(truth)
        probabilities.append(probability)
        person_metrics = evaluate_probabilities(
            truth,
            probability,
            threshold=threshold,
            duration_hours=len(frame) / 3600,
        )
        person_recall.append(float(cast(float, person_metrics["row_recall"])))
        person_f1.append(float(cast(float, person_metrics["row_f1"])))
        if target == "forecast_60s":
            lead_times.extend(
                forecast_lead_times(
                    truth,
                    probability.astype(np.float64),
                    threshold=threshold,
                )
            )
        keep = (
            (np.arange(len(frame)) % 60 == 0)
            | truth.astype(bool)
            | (probability >= threshold)
        )
        retained.append(
            pd.DataFrame(
                {
                    "person_key": entry["person_key"],
                    "timestamp_utc": frame.loc[keep, "timestamp_utc"].to_numpy(),
                    "phase": frame.loc[keep, "phase"].to_numpy(),
                    "target": target,
                    "model_name": model_name,
                    "role": role,
                    "truth": truth[keep],
                    "probability": probability[keep],
                    "threshold": threshold,
                }
            )
        )
    combined_truth = np.concatenate(truths).astype(np.int8)
    combined_probability = np.concatenate(probabilities).astype(np.float64)
    metrics = evaluate_probabilities(
        combined_truth,
        combined_probability,
        threshold=threshold,
        duration_hours=len(combined_truth) / 3600,
    )
    metrics["person_macro_recall"] = float(np.mean(person_recall))
    metrics["person_macro_f1"] = float(np.mean(person_f1))
    if target == "forecast_60s":
        metrics["forecast_mean_lead_time_sec"] = (
            float(np.mean(lead_times)) if lead_times else 0.0
        )
        metrics["forecast_median_lead_time_sec"] = (
            float(np.median(lead_times)) if lead_times else 0.0
        )
        metrics["forecast_detected_events"] = len(lead_times)
    return metrics, retained


def _stress_scenarios(seed: int) -> list[StressScenario]:
    return [
        *(StressScenario("gaussian", level, seed) for level in (0.05, 0.10, 0.20)),
        *(StressScenario("block_missing", seconds, seed) for seconds in (5, 30, 120)),
        *(StressScenario("time_shift", seconds, seed) for seconds in (-5, -1, 1, 5)),
        *(StressScenario("latent_dropout", fraction, seed) for fraction in (0.25, 0.5, 1.0)),
    ]


def _stress_metrics(
    people: list[tuple[dict[str, object], pd.DataFrame]],
    *,
    model: ProbabilityClassifier,
    feature_names: list[str],
    target: str,
    threshold: float,
    clean_aucpr: float,
    seed: int,
) -> list[tuple[str, dict[str, float | int | list[list[int]]], float]]:
    output: list[tuple[str, dict[str, float | int | list[list[int]]], float]] = []
    for scenario in _stress_scenarios(seed):
        truths: list[np.ndarray] = []
        probabilities: list[np.ndarray] = []
        for _, frame in people:
            stressed = apply_stress(frame[feature_names], scenario)
            truths.append(frame[target].to_numpy(dtype=np.int8))
            probabilities.append(
                model.predict_proba(stressed.to_numpy(dtype=np.float32))[:, 1]
            )
        truth = np.concatenate(truths).astype(np.int8)
        probability = np.concatenate(probabilities).astype(np.float64)
        metrics = evaluate_probabilities(
            truth,
            probability,
            threshold=threshold,
            duration_hours=len(truth) / 3600,
        )
        stressed_aucpr = float(cast(float, metrics["aucpr"]))
        degradation = (clean_aucpr - stressed_aucpr) / clean_aucpr if clean_aucpr else 0.0
        output.append(
            (
                f"{scenario.kind}:{scenario.magnitude:g}",
                metrics,
                degradation,
            )
        )
    return output


def train_prepared(
    prepared_root: Path,
    bundle_root: Path,
    *,
    random_state: int,
    locked_test_audit_reason: str,
    uv_lock: Path,
) -> Path:
    """Train fixed candidates, tune on validation, and audit one locked-test evaluation."""

    if len(locked_test_audit_reason.strip()) < 8:
        raise ValueError("locked_test audit_reason is required")
    manifest = json.loads(
        (prepared_root / "manifest.json").read_text(encoding="utf-8")
    )
    feature_names = cast(list[str], manifest["feature_names"])
    entries = cast(list[dict[str, object]], manifest["people"])
    validation_people = _read_people(prepared_root, entries, "validation")
    test_people = _read_people(prepared_root, entries, "locked_test")
    train_entries = [entry for entry in entries if entry["split_role"] == "train"]
    if not train_entries or not validation_people or not test_people:
        raise ValueError("train, validation, and locked_test people are all required")

    models: dict[ModelKey, ProbabilityClassifier] = {}
    thresholds: dict[ModelKey, float] = {}
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    pr_frames: list[pd.DataFrame] = []
    for target in TARGETS:
        training_frames: list[pd.DataFrame] = []
        for entry in train_entries:
            frame = pq.read_table(prepared_root / str(entry["path"])).to_pandas()
            selected = select_training_rows(frame, target=target)
            training_frames.append(frame.loc[selected])
        training = pd.concat(training_frames, ignore_index=True)
        target_models = fit_candidate_models(
            training[feature_names].to_numpy(dtype=np.float32),
            training[target].to_numpy(dtype=np.int8),
            random_state=random_state,
        )
        validation_truth = np.concatenate(
            [frame[target].to_numpy(dtype=np.int8) for _, frame in validation_people]
        ).astype(np.int8)
        for model_name, model in target_models.items():
            validation_probability = np.concatenate(
                [
                    model.predict_proba(
                        frame[feature_names].to_numpy(dtype=np.float32)
                    )[:, 1]
                    for _, frame in validation_people
                ]
            ).astype(np.float64)
            threshold = select_event_threshold(
                validation_truth,
                validation_probability,
                duration_hours=len(validation_truth) / 3600,
            )
            key = (target, model_name)
            models[key] = model
            thresholds[key] = threshold
            precision, recall, pr_thresholds = precision_recall_curve(
                validation_truth, validation_probability
            )
            pr_frame = pd.DataFrame(
                {
                    "target": target,
                    "model_name": model_name,
                    "precision": precision,
                    "recall": recall,
                    "threshold": np.append(pr_thresholds, np.nan),
                }
            )
            if len(pr_frame) > 500:
                pr_frame = pr_frame.iloc[
                    np.linspace(0, len(pr_frame) - 1, 500, dtype=np.int64)
                ]
            pr_frames.append(pr_frame)
            validation_metrics, retained = _evaluate_people(
                validation_people,
                model=model,
                feature_names=feature_names,
                target=target,
                threshold=threshold,
                model_name=model_name,
                role="validation",
            )
            prediction_frames.extend(retained)
            _append_scalar_metrics(
                metric_rows,
                validation_metrics,
                target=target,
                model_name=model_name,
                role="validation",
                threshold=threshold,
            )
            test_metrics, retained = _evaluate_people(
                test_people,
                model=model,
                feature_names=feature_names,
                target=target,
                threshold=threshold,
                model_name=model_name,
                role="locked_test",
            )
            prediction_frames.extend(retained)
            _append_scalar_metrics(
                metric_rows,
                test_metrics,
                target=target,
                model_name=model_name,
                role="locked_test",
                threshold=threshold,
            )
            clean_aucpr = float(cast(float, validation_metrics["aucpr"]))
            for scenario, stress_metrics, degradation in _stress_metrics(
                validation_people,
                model=model,
                feature_names=feature_names,
                target=target,
                threshold=threshold,
                clean_aucpr=clean_aucpr,
                seed=random_state,
            ):
                _append_scalar_metrics(
                    metric_rows,
                    stress_metrics,
                    target=target,
                    model_name=model_name,
                    role="stress",
                    threshold=threshold,
                    scenario=scenario,
                    degradation=degradation,
                )

    global_baseline = pd.DataFrame(
        json.loads((prepared_root / "global_baseline.json").read_text(encoding="utf-8"))
    )
    personal_baseline = pq.read_table(
        prepared_root / "personal_baseline.parquet"
    ).to_pandas()
    lineage = {
        "prepared_manifest_sha256": sha256_file(prepared_root / "manifest.json"),
        "prepared_root": str(prepared_root.resolve()),
        "prepared_series_id": prepared_root.name,
        "split_hash": manifest["source_split_sha256"],
        "dataset_ids": sorted(str(entry["dataset_id"]) for entry in entries),
        "locked_test_audit_reason": locked_test_audit_reason,
        "locked_test_threshold_policy": "validation_only",
        "model_scope": "oracle/sanity",
    }
    bundle = write_model_bundle(
        bundle_root,
        models=models,
        feature_names=feature_names,
        thresholds=thresholds,
        lineage=lineage,
        global_baseline=global_baseline,
        personal_baseline=personal_baseline,
        metrics=pd.DataFrame(metric_rows),
        predictions=pd.concat(prediction_frames, ignore_index=True),
        uv_lock=uv_lock,
        pr_curve=pd.concat(pr_frames, ignore_index=True),
    )
    _record_locked_test_audit(
        bundle,
        prepared_root,
        audit_reason=locked_test_audit_reason,
    )
    return bundle


def logical_model_hash(bundle_root: Path) -> str:
    manifest = json.loads((bundle_root / "manifest.json").read_text(encoding="utf-8"))
    files = cast(dict[str, str], manifest["files"])
    material = "".join(
        f"{path}:{digest}\n"
        for path, digest in sorted(files.items())
        if path.endswith(".skops")
    )
    return hashlib.sha256(material.encode()).hexdigest()


def _audit_ledger(bundle_root: Path) -> Path:
    return bundle_root.parent / "audit" / "locked_test_ledger.jsonl"


def _audit_key(bundle_root: Path, prepared_root: Path) -> tuple[str, str]:
    return logical_model_hash(bundle_root), sha256_file(prepared_root / "manifest.json")


def _record_locked_test_audit(
    bundle_root: Path,
    prepared_root: Path,
    *,
    audit_reason: str,
) -> None:
    model_hash, dataset_hash = _audit_key(bundle_root, prepared_root)
    ledger = _audit_ledger(bundle_root)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "model_hash": model_hash,
        "dataset_hash": dataset_hash,
        "audit_reason": audit_reason,
        "role": "locked_test",
        "result": "completed",
        "scope": "oracle/sanity",
        "real_data_status": "NOT VERIFIED",
    }
    with ledger.open("a", encoding="utf-8") as destination:
        destination.write(json.dumps(record, sort_keys=True) + "\n")


def evaluate_prepared_bundle(
    bundle_root: Path,
    prepared_root: Path,
    *,
    role: str,
    audit_reason: str | None = None,
) -> Path:
    """Evaluate a verified bundle and fail closed on duplicate locked-test access."""

    if role not in {"validation", "locked_test"}:
        raise ValueError("role must be validation or locked_test")
    model_hash, dataset_hash = _audit_key(bundle_root, prepared_root)
    ledger = _audit_ledger(bundle_root)
    if role == "locked_test":
        if not audit_reason or len(audit_reason.strip()) < 8:
            raise ValueError("locked_test audit_reason is required")
        if ledger.exists():
            for line in ledger.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                if (
                    record["model_hash"] == model_hash
                    and record["dataset_hash"] == dataset_hash
                ):
                    raise ValueError("duplicate locked_test model hash x dataset hash")

    prepared_manifest = json.loads(
        (prepared_root / "manifest.json").read_text(encoding="utf-8")
    )
    feature_names = cast(list[str], prepared_manifest["feature_names"])
    entries = cast(list[dict[str, object]], prepared_manifest["people"])
    people = _read_people(prepared_root, entries, role)
    loaded = load_model_bundle(bundle_root)
    model_entries = cast(list[dict[str, object]], loaded.manifest["models"])
    thresholds = {
        (str(entry["target"]), str(entry["model_name"])): float(
            cast(float | int, entry["threshold"])
        )
        for entry in model_entries
    }
    metric_rows: list[dict[str, object]] = []
    for (target, model_name), model in loaded.models.items():
        threshold = thresholds[(target, model_name)]
        metrics, _ = _evaluate_people(
            people,
            model=model,
            feature_names=feature_names,
            target=target,
            threshold=threshold,
            model_name=model_name,
            role=role,
        )
        _append_scalar_metrics(
            metric_rows,
            metrics,
            target=target,
            model_name=model_name,
            role=role,
            threshold=threshold,
        )
    report_root = bundle_root.parent / "evaluations"
    report_root.mkdir(parents=True, exist_ok=True)
    report = report_root / f"{bundle_root.name}__{role}.parquet"
    _write_parquet(pd.DataFrame(metric_rows), report)
    if role == "locked_test":
        _record_locked_test_audit(
            bundle_root,
            prepared_root,
            audit_reason=cast(str, audit_reason),
        )
    return report


def upgrade_bundle_forecast_lead_metrics(bundle_root: Path) -> int:
    """Add forecast lead-time rows to a pre-upgrade bundle and refresh its file hash."""

    metrics_path = bundle_root / "metrics.parquet"
    predictions_path = bundle_root / "predictions.parquet"
    metrics = pq.read_table(metrics_path).to_pandas()
    predictions = pq.read_table(predictions_path).to_pandas()
    existing = set(metrics["metric"].astype(str))
    if "forecast_mean_lead_time_sec" in existing:
        return 0

    additions: list[dict[str, object]] = []
    forecast = predictions.loc[predictions["target"] == "forecast_60s"].copy()
    for (model_name, role), group in forecast.groupby(["model_name", "role"]):
        leads: list[int] = []
        for _, person in group.groupby("person_key"):
            person = person.sort_values("timestamp_utc")
            leads.extend(
                forecast_lead_times(
                    person["truth"].to_numpy(dtype=np.int8),
                    person["probability"].to_numpy(dtype=np.float64),
                    threshold=float(person["threshold"].iloc[0]),
                )
            )
        threshold = float(group["threshold"].iloc[0])
        values = {
            "forecast_mean_lead_time_sec": float(np.mean(leads)) if leads else 0.0,
            "forecast_median_lead_time_sec": float(np.median(leads)) if leads else 0.0,
            "forecast_detected_events": float(len(leads)),
        }
        for metric, value in values.items():
            additions.append(
                {
                    "target": "forecast_60s",
                    "model_name": str(model_name),
                    "role": str(role),
                    "scenario": "clean",
                    "metric": metric,
                    "value": value,
                    "threshold": threshold,
                    "degradation": None,
                }
            )
    _write_parquet(pd.concat([metrics, pd.DataFrame(additions)]), metrics_path)
    manifest_path = bundle_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["metrics.parquet"] = sha256_file(metrics_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return len(additions)
