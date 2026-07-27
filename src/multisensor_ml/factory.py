from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from multisensor_ml.materialize import materialize_synthetic
from multisensor_ml.outcomes import (
    BEHAVIOR_CODES,
    STAGE_CODES,
    build_behavior_labels,
    build_stage_labels,
    load_behavior_ontology,
)
from multisensor_ml.receipts import StageReceipt, create_stage_receipt
from multisensor_ml.registry import sha256_file
from multisensor_ml.settings import FactoryConfig, Goal15Config

OUTCOME_SCHEMA_VERSION = "goal1.5/behavior-outcomes/v1"


@dataclass(frozen=True, slots=True)
class FactoryArtifacts:
    root: Path
    manifest_json: Path
    status: Literal["SUCCESS", "REUSED"]


def run_synthetic_factory(
    config: FactoryConfig,
    output_receipt: Path,
) -> StageReceipt:
    """Materialize registered truth, generate outcomes, and emit a KNIME receipt."""

    load_behavior_ontology(config.behavior_ontology)
    legacy_config = Goal15Config(
        schema_version="goal1.5/config/v1",
        series_id=config.series_id,
        experiment_id=config.series_id,
        generator_config=config.generator_config,
        seeds=config.seeds,
        generator_overrides=config.generator_overrides,
        data_root=config.data_root,
        artifact_root=config.outcome_root.parent / "artifacts",
        random_state=config.random_state,
        locked_test_audit_reason="Synthetic factory lineage registration",
    )
    registered = materialize_synthetic(legacy_config)
    artifacts = build_outcome_artifacts(
        registered.registry_dir,
        config.outcome_root / config.series_id,
        random_state=config.random_state,
        ontology_sha256=sha256_file(config.behavior_ontology),
    )
    manifest = cast(
        dict[str, object],
        json.loads(artifacts.manifest_json.read_text(encoding="utf-8")),
    )
    label_version = str(manifest["factory_input_sha256"])[:24]
    return create_stage_receipt(
        pipeline_run_id=f"factory-{config.series_id}-{label_version[:12]}",
        stage_id="labels",
        artifact_uri=artifacts.manifest_json,
        output_path=output_receipt,
        status=artifacts.status,
        versions={
            "dataset_version": config.series_id,
            "label_version": label_version,
        },
        message_ko=(
            "기존 합성 라벨 재사용"
            if artifacts.status == "REUSED"
            else "합성 라벨 생성 완료"
        ),
    )


def _input_hash(
    registry_manifest: Path,
    random_state: int,
    ontology_sha256: str | None,
) -> str:
    material = {
        "registry_sha256": sha256_file(registry_manifest),
        "random_state": random_state,
        "outcome_schema": OUTCOME_SCHEMA_VERSION,
        "behavior_codes": sorted(BEHAVIOR_CODES),
        "stage_codes": list(STAGE_CODES),
        "ontology_sha256": ontology_sha256 or "built-in-v1",
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _reuse_if_valid(root: Path, expected_input_hash: str) -> FactoryArtifacts | None:
    if not root.exists():
        return None
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"existing outcome root has no manifest: {root}")
    manifest = cast(
        dict[str, object], json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    if manifest.get("factory_input_sha256") != expected_input_hash:
        raise ValueError(f"existing outcome root belongs to a different input: {root}")
    files = cast(dict[str, str], manifest.get("files", {}))
    for name, expected_hash in files.items():
        artifact = root / name
        if not artifact.is_file() or sha256_file(artifact) != expected_hash:
            raise ValueError(f"immutable outcome artifact hash mismatch: {artifact}")
    return FactoryArtifacts(root=root, manifest_json=manifest_path, status="REUSED")


def _event_contexts(
    timeline: pd.DataFrame,
    events: pd.DataFrame,
) -> dict[str, str]:
    timestamps = (
        pd.to_datetime(timeline["timestamp_utc"], utc=True)
        .dt.as_unit("ns")
        .astype("int64")
        .to_numpy()
    )
    contexts = timeline["context_state"].astype(str).to_numpy()
    result: dict[str, str] = {}
    for event in events.to_dict(orient="records"):
        start_ns = int(event["start_time_ns"])
        index = int(np.searchsorted(timestamps, start_ns, side="left"))
        if index >= len(contexts):
            index = len(contexts) - 1
        result[str(event["event_id"])] = str(contexts[index])
    return result


def _safe_event_frame(events: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "run_id",
        "person_id",
        "event_id",
        "event_type",
        "is_target",
        "hard_negative_kind",
        "start_time_ns",
        "pre_late_start_time_ns",
        "onset_time_ns",
        "peak_start_time_ns",
        "peak_end_time_ns",
        "recovery_early_end_time_ns",
        "recovery_late_end_time_ns",
        "end_time_ns",
    ]
    safe = events.loc[:, [column for column in columns if column in events]].copy()
    safe["label_source"] = "synthetic_rule_v1"
    safe["label_confidence"] = np.float32(1.0)
    return safe


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False),
        path,
        compression="zstd",
    )


def build_outcome_artifacts(
    registry_dir: Path,
    output_root: Path,
    *,
    random_state: int,
    ontology_sha256: str | None = None,
) -> FactoryArtifacts:
    """Build an immutable synthetic outcome extension from registered truth runs."""

    registry_manifest = registry_dir.resolve() / "manifest.json"
    expected_input_hash = _input_hash(
        registry_manifest,
        random_state,
        ontology_sha256,
    )
    root = output_root.resolve()
    reused = _reuse_if_valid(root, expected_input_hash)
    if reused is not None:
        return reused

    root.mkdir(parents=True, exist_ok=False)
    registry = cast(
        dict[str, object], json.loads(registry_manifest.read_text(encoding="utf-8"))
    )
    runs = cast(list[dict[str, object]], registry["runs"])
    all_events: list[pd.DataFrame] = []
    hidden_events: list[pd.DataFrame] = []
    stage_writer: pq.ParquetWriter | None = None
    stage_path = root / "outcome_stages.parquet"
    stage_rows = 0
    try:
        for run in runs:
            run_dir = Path(str(run["source_path"]))
            truth_dir = run_dir / "truth"
            events = pq.read_table(truth_dir / "events.parquet").to_pandas()
            all_events.append(_safe_event_frame(events))
            enriched_people: list[pd.DataFrame] = []
            for person_id in sorted(events["person_id"].astype(str).unique()):
                person_events = events.loc[
                    events["person_id"].astype(str) == person_id
                ].copy()
                timeline = pq.read_table(
                    truth_dir / "latent_timeline.parquet",
                    columns=[
                        "run_id",
                        "person_id",
                        "timestamp_utc",
                        "context_state",
                    ],
                    filters=[("person_id", "=", person_id)],
                ).to_pandas()
                timeline = timeline.sort_values("timestamp_utc").reset_index(drop=True)
                contexts = _event_contexts(timeline, person_events)
                person_events["context"] = person_events["event_id"].astype(str).map(
                    contexts
                )
                enriched_people.append(person_events)
                stages = build_stage_labels(timeline, person_events)
                stage_table = pa.Table.from_pandas(stages, preserve_index=False)
                if stage_writer is None:
                    stage_writer = pq.ParquetWriter(
                        stage_path, stage_table.schema, compression="zstd"
                    )
                stage_writer.write_table(stage_table)
                stage_rows += len(stages)
            hidden_events.extend(enriched_people)
    finally:
        if stage_writer is not None:
            stage_writer.close()

    if not hidden_events or not stage_path.exists():
        raise ValueError("registered series contains no outcome events or timeline rows")
    safe_events = pd.concat(all_events, ignore_index=True)
    behavior_source = pd.concat(hidden_events, ignore_index=True)
    behaviors, review_tags = build_behavior_labels(
        behavior_source, random_state=random_state
    )
    _write_frame(safe_events, root / "outcome_events.parquet")
    _write_frame(behaviors, root / "outcome_behaviors.parquet")
    _write_frame(review_tags, root / "outcome_review_tags.parquet")

    artifact_names = (
        "outcome_behaviors.parquet",
        "outcome_events.parquet",
        "outcome_review_tags.parquet",
        "outcome_stages.parquet",
    )
    file_hashes = {name: sha256_file(root / name) for name in artifact_names}
    manifest_path = root / "manifest.json"
    manifest = {
        "schema_version": OUTCOME_SCHEMA_VERSION,
        "series_id": registry["series_id"],
        "status": "oracle/sanity",
        "real_data_status": "NOT VERIFIED",
        "synchronization_status": "NOT_AVAILABLE_TRUTH_ONLY",
        "factory_input_sha256": expected_input_hash,
        "random_state": random_state,
        "ontology_sha256": ontology_sha256 or "built-in-v1",
        "stage_mapping": {
            "NO_EVENT": ["baseline"],
            "LOW": ["pre_early"],
            "MEDIUM": ["pre_late", "onset"],
            "HIGH": ["peak"],
            "DECREASING": ["recovery_early"],
            "RECOVERY": ["recovery_late", "post"],
        },
        "behavior_codes": sorted(BEHAVIOR_CODES),
        "event_count": len(safe_events),
        "stage_row_count": stage_rows,
        "behavior_positive_count": len(behaviors),
        "review_tag_count": len(review_tags),
        "files": file_hashes,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return FactoryArtifacts(root=root, manifest_json=manifest_path, status="SUCCESS")
