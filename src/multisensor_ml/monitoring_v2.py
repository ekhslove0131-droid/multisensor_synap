"""Synthetic observed views and v2 standard-plus-pattern model bundles.

This module is deliberately separate from the Goal 1.5 oracle pipeline.  It
turns the read-only truth runs into a deterministic *oracle/sanity* observed
view, derives only causal features, trains one standard 30-minute forecast and
one multi-head virtual-pattern bundle for each supported sensor view, and
exports the models as ONNX files for the Cloud Run adapter.

The generated observation columns are proxies, not device SDK signals.  The
manifest therefore keeps ``real_data_status=NOT VERIFIED`` and records that
truth columns never enter the feature matrix.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml

V2_SCHEMA = "goal1.5/monitoring-v2/v1"
V2_BUNDLE_SCHEMA = "goal1.5/monitoring-onnx-bundle/v1"
V2_VARIANTS: tuple[str, str] = ("galaxy_watch", "galaxy_watch_h10")
PATTERN_CODES: tuple[str, ...] = (
    "meltdown_like",
    "tantrum_like",
    "sensory_seeking_like",
    "stereotypy_like",
    "shutdown_like",
)
STANDARD_HORIZON_SEC = 1800
WARMUP_SEC = 1800
QUALITY_COLUMNS = ("quality_watch", "quality_h10")

# These are observed/derived names only.  Hidden truth columns, event ids,
# intensity, archetype and phase are never in this list.
WATCH_RAW = ("watch_eda", "watch_hr", "watch_motion", "watch_temperature")
H10_RAW = ("h10_hr", "h10_hrv_rmssd", "h10_motion")
BASE_FEATURES: tuple[str, ...] = (
    "watch_eda_z",
    "watch_hr_z",
    "watch_motion_z",
    "watch_load",
    "watch_load_cum_1800",
    "watch_load_cum_21600",
    "watch_eda_mean_30",
    "watch_hr_mean_30",
    "watch_motion_mean_30",
    "watch_eda_slope_60",
    "watch_hr_slope_60",
    "watch_motion_slope_60",
    "watch_load_std_60",
    "context_index",
    "is_awake",
    "quality_watch",
)
H10_FEATURES: tuple[str, ...] = (
    "h10_hr_z",
    "h10_hrv_rmssd_z",
    "h10_motion_z",
    "h10_load",
    "h10_load_cum_1800",
    "h10_load_cum_21600",
    "h10_hr_mean_30",
    "h10_hrv_rmssd_mean_30",
    "h10_motion_mean_30",
    "h10_hr_slope_60",
    "h10_hrv_rmssd_slope_60",
    "h10_motion_slope_60",
    "h10_load_std_60",
    "quality_h10",
)
STANDARD_AUX_FEATURES: tuple[str, ...] = (
    "standard_forecast",
    "standard_residual",
)


@dataclass(frozen=True, slots=True)
class MonitoringV2Config:
    """Filesystem and deterministic-run settings for the v2 pipeline."""

    project_root: Path
    series_id: str
    data_root: Path
    artifact_root: Path
    output_root: Path
    seed: int = 20260803
    max_train_rows: int = 0
    max_validation_rows: int = 0
    locked_test_read: bool = False

    def __post_init__(self) -> None:
        if self.locked_test_read:
            raise ValueError("locked_test_read must remain false")
        if self.max_train_rows < 0 or self.max_validation_rows < 0:
            raise ValueError("row caps cannot be negative")


@dataclass(frozen=True, slots=True)
class MonitoringV2Result:
    dataset_root: Path
    artifact_root: Path
    bundles: tuple[Path, ...]
    report_json: Path


def load_monitoring_v2_config(path: Path) -> MonitoringV2Config:
    """Load a small YAML config while resolving paths relative to that file."""

    source = path.resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("monitoring-v2 config must be a mapping")
    base = source.parent
    return MonitoringV2Config(
        project_root=base,
        series_id=str(payload["series_id"]),
        data_root=(base / str(payload.get("data_root", "../data"))).resolve(),
        artifact_root=(base / str(payload.get("artifact_root", "../artifacts"))).resolve(),
        output_root=(
            base
            / str(payload.get("output_root", "../services/onnx_api/models/goal15-monitor-v2"))
        ).resolve(),
        seed=int(payload.get("seed", 20260803)),
        max_train_rows=int(payload.get("max_train_rows", 0)),
        max_validation_rows=int(payload.get("max_validation_rows", 0)),
        locked_test_read=bool(payload.get("locked_test_read", False)),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_int(*parts: object) -> int:
    material = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**32 - 1)


def _safe_person_path(person_key: str) -> str:
    return person_key.replace("::", "__").replace("/", "_") + ".parquet"


def _load_split_table(data_root: Path, series_id: str) -> pd.DataFrame:
    path = data_root / "registry" / series_id / "splits.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"split registry is missing: {path}")
    frame = pd.read_parquet(path)
    required = {"run_id", "person_id", "person_key", "split_role", "logical_hash"}
    if not required.issubset(frame.columns):
        raise ValueError(f"split registry is missing {sorted(required - set(frame.columns))}")
    if frame["person_key"].duplicated().any():
        raise ValueError("person_key is duplicated in the split registry")
    counts = frame.groupby("split_role")["person_key"].nunique().to_dict()
    if counts != {"train": 24, "validation": 6, "locked_test": 6}:
        raise ValueError(f"expected 24/6/6 split, got {counts}")
    return frame.sort_values("person_key").reset_index(drop=True)


def _run_root(data_root: Path, series_id: str, run_id: str) -> Path:
    root = data_root / "raw" / series_id / "runs" / run_id
    if not root.is_dir():
        raise FileNotFoundError(f"truth run is missing: {root}")
    return root


def _forward_normal_mean(values: np.ndarray, event: np.ndarray, horizon: int) -> np.ndarray:
    """Mean of the next horizon normal rows; current row is excluded."""

    valid = (~event) & np.isfinite(values)
    forward_values = np.where(valid, values, 0.0)[::-1]
    forward_count = valid.astype(np.int64)[::-1]
    cumulative_values = np.cumsum(forward_values)
    cumulative_count = np.cumsum(forward_count)
    n = len(values)
    end_values = cumulative_values[horizon:]
    end_count = cumulative_count[horizon:]
    start_values = np.concatenate(([0.0], cumulative_values[:-horizon]))
    start_count = np.concatenate(([0], cumulative_count[:-horizon]))
    # This is the future interval [i+1, i+horizon].  Reverse-prefix indexes
    # are converted by slicing the reversed arrays above.
    out = np.full(n, np.nan, dtype="float64")
    future_sum = end_values - start_values
    future_n = end_count - start_count
    usable = future_n >= max(1, horizon // 2)
    out[: n - horizon] = np.divide(
        future_sum,
        future_n,
        out=np.full(n - horizon, np.nan, dtype="float64"),
        where=usable,
    )
    return out


def _future_normal_mean(values: np.ndarray, event: np.ndarray, horizon: int) -> np.ndarray:
    """Forward rolling mean implemented with ordinary (non-reversed) prefixes."""

    n = len(values)
    valid = (~event) & np.isfinite(values)
    prefix_sum = np.concatenate(([0.0], np.cumsum(np.where(valid, values, 0.0))))
    prefix_n = np.concatenate(([0], np.cumsum(valid.astype(np.int64))))
    out = np.full(n, np.nan, dtype="float64")
    if n <= horizon:
        return out
    future_sum = prefix_sum[horizon + 1 : n + 1] - prefix_sum[1 : n - horizon + 1]
    future_n = prefix_n[horizon + 1 : n + 1] - prefix_n[1 : n - horizon + 1]
    usable = future_n >= max(1, horizon // 2)
    out[: n - horizon] = np.divide(
        future_sum,
        future_n,
        out=np.full(n - horizon, np.nan, dtype="float64"),
        where=usable,
    )
    return out


def _causal_rolling(frame: pd.DataFrame, column: str, window: int, op: str) -> pd.Series:
    grouped = frame.groupby("person_key", sort=False)[column]

    def transform(values: pd.Series) -> pd.Series:
        shifted = values.shift(1)
        rolling = shifted.rolling(window, min_periods=1)
        if op == "mean":
            return rolling.mean()
        if op == "std":
            return rolling.std(ddof=0).fillna(0.0)
        raise ValueError(f"unsupported rolling op: {op}")

    return grouped.transform(transform)


def _cumulative_load(frame: pd.DataFrame, column: str, half_life: float) -> pd.Series:
    decay = math.exp(-math.log(2.0) / half_life)
    result = np.zeros(len(frame), dtype="float64")
    values = frame[column].to_numpy(dtype="float64")
    for _, index in frame.groupby("person_key", sort=False).groups.items():
        previous = 0.0
        for position in np.asarray(index, dtype="int64"):
            previous = decay * previous + values[position]
            result[position] = previous
    return pd.Series(result, index=frame.index)


def _event_arrays(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    person_id: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    timestamps = frame["timestamp_utc"].astype("int64").to_numpy()
    event_binary = np.zeros(len(frame), dtype="int8")
    pattern_labels = {
        code: np.zeros(len(frame), dtype="int8") for code in PATTERN_CODES
    }
    hard_negative = np.zeros(len(frame), dtype="int8")
    person_events = events.loc[events["person_id"].astype(str) == person_id]
    archetype_weights: dict[str, dict[str, float]] = {
        "autonomic_first": {"meltdown_like": 0.82, "shutdown_like": 0.20},
        "motor_dominant": {"sensory_seeking_like": 0.74, "stereotypy_like": 0.76},
        "motor_first": {"sensory_seeking_like": 0.62, "stereotypy_like": 0.72},
        "quiet_internal": {"shutdown_like": 0.82, "meltdown_like": 0.36},
        "eeg_first": {"sensory_seeking_like": 0.52, "meltdown_like": 0.46},
        "partial_response": {"meltdown_like": 0.61, "shutdown_like": 0.48},
        "non_responder": {"shutdown_like": 0.32},
    }
    for event in person_events.to_dict(orient="records"):
        start = int(event["start_time_ns"])
        end = int(event["end_time_ns"])
        mask = (timestamps >= start) & (timestamps <= end)
        if not bool(event["is_target"]):
            hard_negative[mask] = 1
            continue
        event_binary[mask] = 1
        archetype = str(event.get("archetype", ""))
        intensity = float(event.get("intensity_truth", 0.5))
        event_id = str(event.get("event_id", ""))
        for code in PATTERN_CODES:
            weight = archetype_weights.get(archetype, {}).get(code, 0.16)
            # A deterministic draw makes labels multi-label and avoids a
            # one-to-one mapping between event type and behavior code.
            draw = _stable_int(person_id, event_id, code) / float(2**32 - 1)
            if code == "tantrum_like":
                # This head requires context/observer evidence in real data;
                # the synthetic prior uses a reproducible demand proxy only.
                weight = 0.28 + 0.30 * ((int(event["start_time_ns"]) // 60_000_000_000) % 3 == 0)
            active = draw < min(0.92, weight + 0.18 * intensity)
            if active:
                pattern_labels[code][mask] = 1
    return event_binary, pattern_labels, hard_negative


def _materialize_person(
    truth_root: Path,
    person_id: str,
    person_key: str,
    split_role: str,
    seed: int,
) -> pd.DataFrame:
    truth_tables = truth_root / "truth"
    latent_path = truth_tables / "latent_timeline.parquet"
    events_path = truth_tables / "events.parquet"
    participant_path = truth_tables / "participants.parquet"
    latent = pd.read_parquet(latent_path)
    latent = latent.loc[latent["person_id"].astype(str) == person_id].copy()
    if latent.empty:
        raise ValueError(f"no latent rows for {person_key}")
    events = pd.read_parquet(events_path)
    participants = pd.read_parquet(participant_path)
    participant = participants.loc[participants["person_id"].astype(str) == person_id]
    if participant.empty:
        raise ValueError(f"participant metadata is missing for {person_key}")
    row = participant.iloc[0]
    rng = np.random.default_rng(_stable_int(seed, person_key))
    latent = latent.sort_values("timestamp_utc").reset_index(drop=True)
    n = len(latent)
    autonomic = latent["autonomic_arousal"].to_numpy(dtype="float64")
    motor = latent["motor_activation"].to_numpy(dtype="float64")
    cognitive = latent["cognitive_load"].to_numpy(dtype="float64")
    sensory = latent["sensory_context"].to_numpy(dtype="float64")
    recovery = latent["recovery_capacity"].to_numpy(dtype="float64")
    resting_hr = float(row["resting_hr_bpm_truth"])
    rmssd = float(row["rmssd_ms_truth"])
    eda_baseline = float(row["eda_tonic_level_us_truth"])
    # The following are deterministic observation proxies with distinct
    # modality noise. They are not claimed to be SDK-level physical signals.
    watch_eda = eda_baseline + 1.5 * autonomic + 0.75 * sensory + 0.30 * cognitive
    watch_eda += rng.normal(0.0, max(0.03, eda_baseline * 0.025), n)
    watch_hr = resting_hr + 19.0 * autonomic + 5.0 * motor + rng.normal(0.0, 1.8, n)
    watch_motion = np.maximum(
        0.0,
        0.35 * motor + 0.10 * sensory + rng.normal(0.0, 0.035, n),
    )
    watch_temperature = 33.0 + 0.15 * recovery - 0.05 * autonomic + rng.normal(0.0, 0.025, n)
    h10_hr = resting_hr + 18.0 * autonomic + 4.0 * motor + rng.normal(0.0, 0.9, n)
    h10_hrv = rmssd - 11.0 * autonomic - 3.5 * motor + 2.0 * recovery
    h10_hrv += rng.normal(0.0, max(0.5, rmssd * 0.035), n)
    h10_motion = np.maximum(0.0, 0.28 * motor + rng.normal(0.0, 0.025, n))
    quality_watch = np.clip(0.985 - np.abs(rng.normal(0.0, 0.015, n)), 0.55, 1.0)
    quality_h10 = np.clip(0.99 - np.abs(rng.normal(0.0, 0.010, n)), 0.65, 1.0)
    frame = pd.DataFrame(
        {
            "person_key": person_key,
            "person_id": person_id,
            "split_role": split_role,
            "timestamp_utc": pd.to_datetime(latent["timestamp_utc"], utc=True),
            "context_state": latent["context_state"].astype(str),
            "is_awake": latent["is_awake"].astype("int8"),
            "watch_eda": watch_eda,
            "watch_hr": watch_hr,
            "watch_motion": watch_motion,
            "watch_temperature": watch_temperature,
            "h10_hr": h10_hr,
            "h10_hrv_rmssd": h10_hrv,
            "h10_motion": h10_motion,
            "quality_watch": quality_watch,
            "quality_h10": quality_h10,
        }
    )
    event_binary, pattern_labels, hard_negative = _event_arrays(frame, events, person_id)
    frame["event_binary"] = event_binary
    frame["hard_negative"] = hard_negative
    for code, values in pattern_labels.items():
        frame[f"label_{code}"] = values
    frame["timestamp_ns"] = frame["timestamp_utc"].astype("int64")
    # Context is intentionally an observed categorical encoding, not a hidden
    # event or archetype field.
    context_codes = pd.Categorical(frame["context_state"]).codes.astype("float64")
    frame["context_index"] = context_codes
    return frame


def _add_causal_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["person_key", "timestamp_utc"]).reset_index(drop=True)
    # Person-specific warm-up statistics are fixed using the first 900 seconds;
    # all scored rows begin after the 1,800-second warm-up.
    warm = frame.groupby("person_key", sort=False).cumcount() < 900
    for column in (*WATCH_RAW, *H10_RAW):
        center = frame.loc[warm].groupby("person_key")[column].median()
        scale = frame.loc[warm].groupby("person_key")[column].apply(
            lambda values: float(1.4826 * np.median(np.abs(values - np.median(values))) + 1e-3)
        )
        centers = frame["person_key"].map(center).astype("float64")
        scales = frame["person_key"].map(scale).astype("float64").clip(lower=1e-3)
        frame[f"{column}_z"] = ((frame[column] - centers) / scales).clip(-20.0, 20.0)
    frame["watch_load"] = (
        frame[["watch_eda_z", "watch_hr_z", "watch_motion_z"]].clip(lower=0.0).mean(axis=1)
        * frame["quality_watch"]
    )
    frame["h10_load"] = (
        frame[["h10_hr_z", "h10_hrv_rmssd_z", "h10_motion_z"]].clip(lower=0.0).mean(axis=1)
        * frame["quality_h10"]
    )
    for prefix, raw_columns, load_column in (
        ("watch", WATCH_RAW, "watch_load"),
        ("h10", H10_RAW, "h10_load"),
    ):
        for column in raw_columns:
            z_column = f"{column}_z"
            frame[f"{column}_mean_30"] = _causal_rolling(frame, z_column, 30, "mean")
            mean_60 = _causal_rolling(frame, z_column, 60, "mean")
            mean_120 = _causal_rolling(frame, z_column, 120, "mean")
            frame[f"{column}_slope_60"] = (mean_60 - mean_120) / 60.0
        frame[f"{prefix}_load_cum_1800"] = _cumulative_load(frame, load_column, 1800.0)
        frame[f"{prefix}_load_cum_21600"] = _cumulative_load(frame, load_column, 21600.0)
        frame[f"{prefix}_load_std_60"] = _causal_rolling(frame, load_column, 60, "std")
    frame["watch_load_mean_300"] = _causal_rolling(frame, "watch_load", 300, "mean")
    frame["watch_load_std_300"] = _causal_rolling(frame, "watch_load", 300, "std")
    frame["standard_residual_proxy"] = frame["watch_load"] - frame["watch_load_mean_300"]
    frame["warmup_complete"] = (
        ~frame.groupby("person_key").cumcount().lt(WARMUP_SEC)
    ).astype("int8")
    frame["quality_ok"] = (
        (frame["quality_watch"] >= 0.70) & (frame["quality_h10"] >= 0.70)
    ).astype("int8")
    frame["standard_target"] = np.nan
    targets: list[np.ndarray] = []
    for _, group in frame.groupby("person_key", sort=False):
        indices = group.index.to_numpy()
        target = _future_normal_mean(
            group["watch_load"].to_numpy(dtype="float64"),
            group["event_binary"].to_numpy(dtype=bool),
            STANDARD_HORIZON_SEC,
        )
        frame.loc[indices, "standard_target"] = target
        targets.append(target)
    frame["standard_valid"] = (
        frame["standard_target"].notna()
        & (frame["warmup_complete"] == 1)
        & (frame["quality_ok"] == 1)
    ).astype("int8")
    return frame


def materialize_monitoring_dataset(config: MonitoringV2Config) -> Path:
    """Create deterministic observed/model-ready person tables and manifest."""

    data_root = config.data_root.resolve()
    split = _load_split_table(data_root, config.series_id)
    dataset_root = data_root / "model_ready" / config.series_id
    people_root = dataset_root / "people"
    people_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    for record in split.to_dict(orient="records"):
        person_key = str(record["person_key"])
        output = people_root / _safe_person_path(person_key)
        if output.exists():
            frame = pd.read_parquet(output)
        else:
            truth_root = _run_root(data_root, config.series_id, str(record["run_id"]))
            frame = _materialize_person(
                truth_root,
                str(record["person_id"]),
                person_key,
                str(record["split_role"]),
                config.seed,
            )
            frame = _add_causal_features(frame)
            frame.to_parquet(output, index=False)
        entries.append(
            {
                "person_key": person_key,
                "run_id": str(record["run_id"]),
                "person_id": str(record["person_id"]),
                "split_role": str(record["split_role"]),
                "path": str(output.relative_to(dataset_root)),
                "rows": len(frame),
                "logical_hash": str(record["logical_hash"]),
                "file_sha256": _sha256(output),
            }
        )
    feature_names = list(BASE_FEATURES + H10_FEATURES + ("standard_residual_proxy",))
    manifest: dict[str, object] = {
        "schema_version": V2_SCHEMA,
        "dataset_id": f"{config.series_id}-model-ready-v2",
        "series_id": config.series_id,
        "source_domain": "oracle/sanity",
        "status": "oracle/sanity",
        "real_data_status": "NOT VERIFIED",
        "truth_columns_used_only_for_label_generation": True,
        "feature_names": feature_names,
        "standard_target": "standard_target",
        "standard_horizon_sec": STANDARD_HORIZON_SEC,
        "pattern_codes": list(PATTERN_CODES),
        "label_source": "synthetic_prior_v1",
        "split_contract": {"train": 24, "validation": 6, "locked_test": 6},
        "locked_test_read": False,
        "people": entries,
    }
    (dataset_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return dataset_root


def _load_people(dataset_root: Path, role: str) -> pd.DataFrame:
    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("locked_test_read", False) is not False:
        raise ValueError("model-ready manifest has locked_test_read=true")
    entries = [entry for entry in manifest["people"] if entry["split_role"] == role]
    frames = [pd.read_parquet(dataset_root / str(entry["path"])) for entry in entries]
    if not frames:
        raise ValueError(f"no people for split role {role}")
    return pd.concat(frames, ignore_index=True)


def _variant_base_features(variant: str) -> tuple[str, ...]:
    if variant not in V2_VARIANTS:
        raise ValueError(f"unsupported v2 variant: {variant}")
    return BASE_FEATURES if variant == "galaxy_watch" else BASE_FEATURES + H10_FEATURES


def _lightgbm_regressor(seed: int) -> Any:
    try:
        from lightgbm import LGBMRegressor
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("install the tree-benchmark extra") from error
    return LGBMRegressor(
        objective="regression",
        n_estimators=180,
        num_leaves=31,
        learning_rate=0.05,
        min_child_samples=40,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        random_state=seed,
        bagging_seed=seed,
        feature_fraction_seed=seed,
        data_random_seed=seed,
        deterministic=True,
        n_jobs=1,
        verbosity=-1,
        device_type="cpu",
        force_col_wise=True,
        subsample_freq=1,
    )


def _lightgbm_classifier(seed: int, positive: int, negative: int) -> Any:
    try:
        from lightgbm import LGBMClassifier
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("install the tree-benchmark extra") from error
    ratio = max(1.0, negative / max(1, positive))
    return LGBMClassifier(
        objective="binary",
        n_estimators=160,
        num_leaves=31,
        learning_rate=0.05,
        min_child_samples=35,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        scale_pos_weight=ratio,
        random_state=seed,
        bagging_seed=seed,
        feature_fraction_seed=seed,
        data_random_seed=seed,
        deterministic=True,
        n_jobs=1,
        verbosity=-1,
        device_type="cpu",
        force_col_wise=True,
        subsample_freq=1,
    )


def _export_lightgbm(model: Any, feature_count: int, path: Path) -> None:
    try:
        import onnx
        import onnxmltools  # type: ignore[import-untyped]
        from onnxmltools.convert.common.data_types import (  # type: ignore[import-untyped]
            FloatTensorType,
        )
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("install the onnx-serving extra") from error
    graph = onnxmltools.convert_lightgbm(
        model,
        initial_types=[("features", FloatTensorType([None, feature_count]))],
        target_opset=15,
        zipmap=False,
    )
    # onnxmltools otherwise assigns a random UUID graph name, which makes an
    # identical seed produce a different SHA-256 on every export.
    graph.graph.name = f"monitoring_v2_{path.stem}"
    onnx.checker.check_model(graph)
    path.write_bytes(graph.SerializeToString())


def _metric_dict(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, recall_score

    predicted = probability >= threshold
    positives = int(y_true.sum())
    hours = max(1e-9, len(y_true) / 3600.0)
    return {
        "aucpr": float(average_precision_score(y_true, probability)) if positives else 0.0,
        "event_recall": float(recall_score(y_true, predicted, zero_division=0)),
        "event_f1": float(f1_score(y_true, predicted, zero_division=0)),
        "false_alerts_per_hour": float(((predicted == 1) & (y_true == 0)).sum() / hours),
        "brier": float(brier_score_loss(y_true, probability)),
        "positive_rows": float(positives),
    }


def _best_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    candidates = np.unique(np.quantile(probability, np.linspace(0.02, 0.98, 49)))
    if len(candidates) == 0:
        return 0.5
    scores = [f1_score(y_true, probability >= value, zero_division=0) for value in candidates]
    return float(candidates[int(np.argmax(scores))])


def _predict_probability(
    model: Any, frame: pd.DataFrame, feature_names: tuple[str, ...]
) -> np.ndarray:
    return np.asarray(model.predict_proba(frame[list(feature_names)])[:, 1], dtype="float64")


def _standard_oof_predictions(
    train: pd.DataFrame,
    feature_names: tuple[str, ...],
    seed: int,
) -> tuple[np.ndarray, Any, float]:
    valid = train["standard_valid"].astype(bool).to_numpy()
    predictions = np.full(len(train), np.nan, dtype="float64")
    people = np.array(sorted(train.loc[valid, "person_key"].astype(str).unique()))
    for fold in range(3):
        held_out = set(people[fold::3])
        fit_mask = valid & ~train["person_key"].astype(str).isin(held_out).to_numpy()
        hold_mask = valid & train["person_key"].astype(str).isin(held_out).to_numpy()
        model = _lightgbm_regressor(seed + fold)
        model.fit(train.loc[fit_mask, list(feature_names)], train.loc[fit_mask, "standard_target"])
        predictions[hold_mask] = model.predict(train.loc[hold_mask, list(feature_names)])
    final_model = _lightgbm_regressor(seed)
    final_model.fit(train.loc[valid, list(feature_names)], train.loc[valid, "standard_target"])
    train_predictions = final_model.predict(train.loc[valid, list(feature_names)])
    baseline = train.loc[valid, "watch_load_mean_300"].to_numpy(dtype="float64")
    target = train.loc[valid, "standard_target"].to_numpy(dtype="float64")
    model_mae = float(np.mean(np.abs(train_predictions - target)))
    baseline_mae = float(np.mean(np.abs(baseline - target)))
    return predictions, final_model, model_mae / max(1e-9, baseline_mae)


def train_monitoring_v2(config: MonitoringV2Config) -> MonitoringV2Result:
    """Train the two sensor-view bundles and write reproducible ONNX artifacts."""

    dataset_root = materialize_monitoring_dataset(config)
    train = _load_people(dataset_root, "train")
    validation = _load_people(dataset_root, "validation")
    if config.max_train_rows and len(train) > config.max_train_rows:
        train = (
            train.sort_values(["person_key", "timestamp_utc"])
            .groupby("person_key", sort=False)
            .head(max(1, config.max_train_rows // 24))
        )
    if config.max_validation_rows and len(validation) > config.max_validation_rows:
        validation = (
            validation.sort_values(["person_key", "timestamp_utc"])
            .groupby("person_key", sort=False)
            .head(max(1, config.max_validation_rows // 6))
        )
    artifact_root = config.artifact_root.resolve() / "monitoring-v2" / config.series_id
    artifact_root.mkdir(parents=True, exist_ok=True)
    output_root = config.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    bundle_paths: list[Path] = []
    report: dict[str, object] = {
        "schema_version": V2_BUNDLE_SCHEMA,
        "status": "oracle/sanity",
        "real_data_status": "NOT VERIFIED",
        "locked_test_read": False,
        "dataset_id": f"{config.series_id}-model-ready-v2",
        "standard_horizon_sec": STANDARD_HORIZON_SEC,
        "variants": {},
    }
    for variant in V2_VARIANTS:
        base_features = _variant_base_features(variant)
        valid_train = train["standard_valid"].astype(bool)
        valid_validation = validation["standard_valid"].astype(bool)
        standard_model = _lightgbm_regressor(config.seed + len(variant))
        standard_model.fit(
            train.loc[valid_train, list(base_features)],
            train.loc[valid_train, "standard_target"],
        )
        validation_standard = standard_model.predict(
            validation.loc[valid_validation, list(base_features)]
        )
        target_validation = validation.loc[valid_validation, "standard_target"].to_numpy(
            dtype="float64"
        )
        persistence_validation = validation.loc[
            valid_validation, "watch_load_mean_300"
        ].to_numpy(dtype="float64")
        standard_metrics = {
            "mae": float(np.mean(np.abs(validation_standard - target_validation))),
            "rmse": float(np.sqrt(np.mean((validation_standard - target_validation) ** 2))),
            "persistence_mae": float(np.mean(np.abs(persistence_validation - target_validation))),
            "normal_rows": float(len(target_validation)),
        }
        train_oof, _, _ = _standard_oof_predictions(train, base_features, config.seed + 11)
        train = train.copy()
        validation = validation.copy()
        train["standard_forecast"] = train_oof
        validation["standard_forecast"] = np.nan
        validation.loc[valid_validation, "standard_forecast"] = validation_standard
        train["standard_forecast"] = train["standard_forecast"].fillna(
            train["watch_load_mean_300"]
        )
        validation["standard_forecast"] = validation["standard_forecast"].fillna(
            validation["watch_load_mean_300"]
        )
        train["standard_residual"] = train["watch_load"] - train["standard_forecast"]
        validation["standard_residual"] = validation["watch_load"] - validation["standard_forecast"]
        pattern_features = base_features + STANDARD_AUX_FEATURES
        bundle_root = output_root / variant
        bundle_root.mkdir(parents=True, exist_ok=True)
        standard_path = bundle_root / "standard_30m.onnx"
        _export_lightgbm(standard_model, len(base_features), standard_path)
        pattern_paths: dict[str, str] = {}
        pattern_metrics: dict[str, object] = {}
        for offset, code in enumerate(PATTERN_CODES):
            target_column = f"label_{code}"
            positive = int(train[target_column].sum())
            negative = int(len(train) - positive)
            if positive < 20:
                pattern_metrics[code] = {
                    "status": "INSUFFICIENT_LABEL_SUPPORT",
                    "positive_rows": positive,
                }
                continue
            model = _lightgbm_classifier(config.seed + offset + len(variant), positive, negative)
            model.fit(train[list(pattern_features)], train[target_column].astype("int8"))
            probability = _predict_probability(model, validation, pattern_features)
            y_validation = validation[target_column].to_numpy(dtype="int8")
            threshold = _best_threshold(y_validation, probability)
            pattern_metrics[code] = {
                "status": "CANDIDATE",
                "threshold": threshold,
                **_metric_dict(y_validation, probability, threshold),
            }
            path = bundle_root / f"pattern_{code}.onnx"
            _export_lightgbm(model, len(pattern_features), path)
            pattern_paths[code] = path.name
        thresholds: dict[str, float] = {}
        for code in pattern_paths:
            metric = pattern_metrics.get(code)
            if isinstance(metric, dict) and isinstance(metric.get("threshold"), (int, float)):
                thresholds[code] = float(metric["threshold"])
        manifest: dict[str, object] = {
            "schema_version": V2_BUNDLE_SCHEMA,
            "bundle_kind": "standard_30m_plus_virtual_patterns",
            "bundle_id": f"monitoring-v2-{config.series_id}-{variant}",
            "model_version": "v2.0.0-oracle-sanity",
            "variant": variant,
            "dataset_id": f"{config.series_id}-model-ready-v2",
            "feature_names": list(base_features),
            "pattern_feature_names": list(pattern_features),
            "standard_target": "next_30m_normal_watch_load_mean",
            "standard_horizon_sec": STANDARD_HORIZON_SEC,
            "standard_model_path": standard_path.name,
            "pattern_model_paths": pattern_paths,
            "pattern_codes": list(PATTERN_CODES),
            "thresholds": thresholds,
            "model_scope": "oracle/sanity",
            "real_data_status": "NOT VERIFIED",
            "locked_test_read": False,
            "split_contract": {"train_people": 24, "validation_people": 6, "locked_test_people": 6},
            "standard_metrics": standard_metrics,
            "pattern_metrics": pattern_metrics,
            "model_sha256": {
                "standard_30m.onnx": _sha256(standard_path),
                **{name: _sha256(bundle_root / name) for name in pattern_paths.values()},
            },
        }
        (bundle_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        bundle_paths.append(bundle_root)
        cast(dict[str, object], report["variants"])[variant] = manifest
    report_json = artifact_root / "result.json"
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (artifact_root / "dataset_manifest.json").write_text(
        (dataset_root / "manifest.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    return MonitoringV2Result(dataset_root, artifact_root, tuple(bundle_paths), report_json)


def run_monitoring_v2(config: MonitoringV2Config) -> MonitoringV2Result:
    return train_monitoring_v2(config)
