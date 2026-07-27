from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import skops.io as sio
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

OODStatus = Literal[
    "IN_DISTRIBUTION",
    "ADAPTATION_CAPPED",
    "OOD_MONITOR",
    "RETRAIN_CANDIDATE",
    "NOT_DECISIONABLE",
]


@dataclass(frozen=True, slots=True)
class StandardTypeResult:
    selected_k: int
    memberships: pd.DataFrame
    diagnostics: pd.DataFrame
    centroids: pd.DataFrame
    model: GaussianMixture
    feature_names: tuple[str, ...]
    median: np.ndarray
    scale: np.ndarray
    component_order: np.ndarray


@dataclass(frozen=True, slots=True)
class StandardTypeArtifacts:
    root: Path
    manifest_json: Path
    selected_k: int


def _robust_matrix(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    feature_names = [
        column
        for column in frame.columns
        if column != "person_key" and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not feature_names:
        raise ValueError("standard type discovery requires numeric baseline features")
    values = frame[feature_names].to_numpy(dtype=np.float64)
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    scale = np.where(mad > 1e-8, 1.4826 * mad, 1.0)
    return (values - median) / scale, feature_names, median, scale


def _type_names(count: int) -> list[str]:
    return [f"STD-{chr(ord('A') + index)}" for index in range(count)]


def discover_standard_types(
    frame: pd.DataFrame,
    *,
    random_state: int,
    max_k: int = 6,
    stability_repeats: int = 10,
) -> StandardTypeResult:
    """Discover neutral soft standard types from train-person baseline summaries."""

    if frame["person_key"].duplicated().any():
        raise ValueError("standard type discovery requires unique person_key rows")
    matrix, feature_names, median, scale = _robust_matrix(frame)
    people = len(frame)
    if people < 3:
        raise ValueError("standard type discovery requires at least three people")
    unique_rows = int(np.unique(matrix, axis=0).shape[0])
    candidate_max = min(max_k, people // 3, unique_rows)
    diagnostics: list[dict[str, object]] = []
    fitted: dict[int, GaussianMixture] = {}
    constant = bool(np.all(np.std(matrix, axis=0) < 1e-8))

    for k in range(1, max(candidate_max, 1) + 1):
        if k == 1 or constant:
            bic = float(np.sum(matrix**2)) if k == 1 else float("inf")
            stability = 1.0 if k == 1 else 0.0
            min_mass = float(people) if k == 1 else 0.0
            confidence = 1.0 if k == 1 else 0.0
            kmeans_agreement = 1.0 if k == 1 else 0.0
            feasible = k == 1
            if k == 1:
                model = GaussianMixture(
                    n_components=1,
                    covariance_type="diag",
                    random_state=random_state,
                    reg_covar=1e-5,
                ).fit(matrix)
                fitted[k] = model
        else:
            model = GaussianMixture(
                n_components=k,
                covariance_type="diag",
                random_state=random_state,
                n_init=5,
                reg_covar=1e-5,
            ).fit(matrix)
            fitted[k] = model
            probabilities = model.predict_proba(matrix)
            base_labels = probabilities.argmax(axis=1)
            scores: list[float] = []
            sample_size = max(k * 3, int(np.ceil(people * 0.8)))
            for repeat in range(stability_repeats):
                rng = np.random.default_rng(random_state + k * 1000 + repeat)
                sample = np.sort(rng.choice(people, size=sample_size, replace=False))
                subsample_model = GaussianMixture(
                    n_components=k,
                    covariance_type="diag",
                    random_state=random_state + repeat + 1,
                    n_init=3,
                    reg_covar=1e-5,
                ).fit(matrix[sample])
                scores.append(
                    adjusted_rand_score(
                        base_labels[sample],
                        subsample_model.predict(matrix[sample]),
                    )
                )
            kmeans = KMeans(
                n_clusters=k,
                n_init=20,
                random_state=random_state,
            ).fit(matrix)
            bic = float(model.bic(matrix))
            stability = float(np.median(scores))
            min_mass = float(probabilities.sum(axis=0).min())
            confidence = float(np.median(probabilities.max(axis=1)))
            kmeans_agreement = float(
                adjusted_rand_score(base_labels, kmeans.labels_)
            )
            feasible = (
                stability >= 0.75
                and min_mass >= 3.0
                and confidence >= 0.60
                and kmeans_agreement >= 0.60
            )
        diagnostics.append(
            {
                "k": k,
                "bic": bic,
                "stability_median_ari": stability,
                "minimum_effective_members": min_mass,
                "median_max_membership": confidence,
                "kmeans_agreement_ari": kmeans_agreement,
                "feasible": feasible,
            }
        )

    diagnostics_frame = pd.DataFrame(diagnostics)
    feasible_rows = diagnostics_frame.loc[diagnostics_frame["feasible"]]
    selected_k = int(feasible_rows.sort_values(["bic", "k"]).iloc[0]["k"])
    selected = fitted[selected_k]
    probabilities = selected.predict_proba(matrix)
    component_order = np.lexsort(
        tuple(
            selected.means_[:, index]
            for index in reversed(range(selected.means_.shape[1]))
        )
    )
    probabilities = probabilities[:, component_order]
    ordered_centers = selected.means_[component_order]
    names = _type_names(selected_k)
    memberships = pd.DataFrame(probabilities, columns=names)
    memberships.insert(0, "person_key", frame["person_key"].astype(str).to_numpy())
    original_centers = ordered_centers * scale + median
    centroids = pd.DataFrame(original_centers, columns=feature_names)
    centroids.insert(0, "standard_type", names)
    return StandardTypeResult(
        selected_k=selected_k,
        memberships=memberships,
        diagnostics=diagnostics_frame,
        centroids=centroids,
        model=selected,
        feature_names=tuple(feature_names),
        median=median,
        scale=scale,
        component_order=component_order,
    )


def select_adaptation_cap(metrics: pd.DataFrame) -> float:
    """Choose the smallest near-best cap that does not degrade the global baseline."""

    required = {
        "cap",
        "aucpr",
        "event_recall",
        "event_f1",
        "false_alerts_per_hour",
        "ece",
        "lead_time_sec",
    }
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"adaptation metrics missing columns: {sorted(missing)}")
    baseline_rows = metrics.loc[np.isclose(metrics["cap"].astype(float), 0.0)]
    if len(baseline_rows) != 1:
        raise ValueError("adaptation metrics require exactly one cap=0 baseline")
    baseline = baseline_rows.iloc[0]
    false_alert_limit = float(baseline["false_alerts_per_hour"]) * 1.10
    feasible = metrics.loc[
        (metrics["aucpr"] >= float(baseline["aucpr"]) * 0.98)
        & (metrics["event_recall"] >= float(baseline["event_recall"]) - 0.02)
        & (metrics["false_alerts_per_hour"] <= false_alert_limit)
        & (metrics["ece"] <= float(baseline["ece"]) + 0.02)
        & (metrics["lead_time_sec"] >= float(baseline["lead_time_sec"]) - 5.0)
    ]
    if feasible.empty:
        return 0.0
    best_f1 = float(feasible["event_f1"].max())
    near_best = feasible.loc[feasible["event_f1"] >= best_f1 - 0.01]
    return float(near_best["cap"].min())


def classify_ood(
    *,
    distance: float,
    p95: float,
    p99: float,
    quality_valid: bool,
    adaptation_capped: bool,
    persistent_sessions: int,
    distinct_days: int,
    valid_seconds: int,
) -> OODStatus:
    if not quality_valid:
        return "NOT_DECISIONABLE"
    if distance <= p95:
        return "ADAPTATION_CAPPED" if adaptation_capped else "IN_DISTRIBUTION"
    if (
        distance > p99
        and persistent_sessions >= 3
        and distinct_days >= 2
        and valid_seconds >= 1800
    ):
        return "RETRAIN_CANDIDATE"
    return "OOD_MONITOR"


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False),
        path,
        compression="zstd",
    )


def fit_standard_type_artifacts(
    prepared_root: Path,
    output_root: Path,
    *,
    random_state: int,
) -> StandardTypeArtifacts:
    """Fit standard types on train-person baselines and score every registered person."""

    prepared = prepared_root.resolve()
    root = output_root.resolve()
    if root.exists():
        raise FileExistsError(f"standard type artifact root already exists: {root}")
    root.mkdir(parents=True)
    manifest = json.loads(
        (prepared / "manifest.json").read_text(encoding="utf-8")
    )
    people = pd.DataFrame(manifest["people"])[["person_key", "split_role"]]
    baseline = pq.read_table(prepared / "personal_baseline.parquet").to_pandas()
    summary_features = [
        column
        for column in baseline.columns
        if column.endswith("__center") or column.endswith("__mad")
    ]
    if not summary_features:
        raise ValueError("personal baseline contains no center/MAD summary features")
    summary = (
        baseline.groupby("person_key", as_index=False)[summary_features]
        .median()
        .merge(people, on="person_key", how="inner", validate="one_to_one")
    )
    train = summary.loc[summary["split_role"] == "train", ["person_key", *summary_features]]
    discovered = discover_standard_types(train, random_state=random_state)
    all_matrix = (
        summary[list(discovered.feature_names)].to_numpy(dtype=np.float64)
        - discovered.median
    ) / discovered.scale
    probabilities = discovered.model.predict_proba(all_matrix)[
        :, discovered.component_order
    ]
    names = _type_names(discovered.selected_k)
    memberships = pd.DataFrame(probabilities, columns=names)
    memberships.insert(0, "person_key", summary["person_key"].astype(str).to_numpy())
    memberships.insert(1, "split_role", summary["split_role"].astype(str).to_numpy())
    _write_parquet(memberships, root / "person_standard_types.parquet")
    _write_parquet(
        discovered.diagnostics,
        root / "standard_type_diagnostics.parquet",
    )
    _write_parquet(discovered.centroids, root / "standard_type_centroids.parquet")

    train_matrix = (
        train[list(discovered.feature_names)].to_numpy(dtype=np.float64)
        - discovered.median
    ) / discovered.scale
    train_distances = -discovered.model.score_samples(train_matrix)
    p95 = float(np.quantile(train_distances, 0.95))
    p99 = float(np.quantile(train_distances, 0.99))
    distances = -discovered.model.score_samples(all_matrix)
    ood = pd.DataFrame(
        {
            "person_key": summary["person_key"].astype(str),
            "split_role": summary["split_role"].astype(str),
            "distance": distances,
            "p95": p95,
            "p99": p99,
            "status": [
                classify_ood(
                    distance=float(distance),
                    p95=p95,
                    p99=p99,
                    quality_valid=True,
                    adaptation_capped=False,
                    persistent_sessions=0,
                    distinct_days=0,
                    valid_seconds=0,
                )
                for distance in distances
            ],
        }
    )
    _write_parquet(ood, root / "ood_status.parquet")
    sio.dump(discovered.model, root / "standard_type_model.skops")
    preprocessing = {
        "feature_names": list(discovered.feature_names),
        "median": discovered.median.tolist(),
        "scale": discovered.scale.tolist(),
        "component_order": discovered.component_order.tolist(),
        "p95": p95,
        "p99": p99,
    }
    (root / "standard_type_preprocessing.json").write_text(
        json.dumps(preprocessing, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "goal1.5/standard-types/v1",
                "status": "oracle/sanity",
                "selected_k": discovered.selected_k,
                "train_person_count": len(train),
                "scored_person_count": len(summary),
                "type_names": names,
                "real_data_status": "NOT VERIFIED",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return StandardTypeArtifacts(
        root=root,
        manifest_json=manifest_path,
        selected_k=discovered.selected_k,
    )
