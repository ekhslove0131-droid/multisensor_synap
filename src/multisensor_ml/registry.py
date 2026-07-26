from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq

from multisensor_ml.contracts import DatasetRecord, assign_person_splits


@dataclass(frozen=True, slots=True)
class RegisteredSeries:
    series_id: str
    registry_dir: Path
    records_jsonl: Path
    split_parquet: Path
    manifest_json: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _people_in_timeline(path: Path) -> set[tuple[str, str]]:
    people: set[tuple[str, str]] = set()
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["run_id", "person_id"], batch_size=65_536):
        run_ids = batch.column(0).to_pylist()
        person_ids = batch.column(1).to_pylist()
        people.update(zip(run_ids, person_ids, strict=True))
    return people


def _table_hash(manifest: dict[str, object], relative: str, path: Path) -> str:
    tables = cast(dict[str, object], manifest.get("tables", {}))
    summary = cast(dict[str, object], tables.get(relative, {}))
    logical = summary.get("logical_sha256")
    return str(logical) if logical else sha256_file(path)


def register_series(
    series_id: str,
    run_dirs: list[Path],
    registry_root: Path,
) -> RegisteredSeries:
    """Register immutable Goal 1 truth references without copying generator output."""

    people: list[tuple[str, str]] = []
    run_metadata: dict[str, tuple[Path, dict[str, object], str]] = {}
    seen: set[tuple[str, str]] = set()
    for raw_run_dir in run_dirs:
        run_dir = raw_run_dir.resolve()
        manifest_path = run_dir / "manifest.json"
        timeline_path = run_dir / "truth" / "latent_timeline.parquet"
        events_path = run_dir / "truth" / "events.parquet"
        manifest = cast(dict[str, object], json.loads(manifest_path.read_text(encoding="utf-8")))
        run_id = str(manifest["run_id"])
        timeline_hash = _table_hash(
            manifest, "truth/latent_timeline.parquet", timeline_path
        )
        events_hash = _table_hash(manifest, "truth/events.parquet", events_path)
        run_metadata[run_id] = (run_dir, manifest, f"{timeline_hash}:{events_hash}")
        for person in sorted(_people_in_timeline(timeline_path)):
            if person in seen:
                raise ValueError(f"duplicate run/person pair: {person[0]}/{person[1]}")
            if person[0] != run_id:
                raise ValueError(f"manifest run_id does not match timeline: {person[0]}")
            seen.add(person)
            people.append(person)

    roles = assign_person_splits(people)
    records: list[DatasetRecord] = []
    for run_id, person_id in sorted(people):
        run_dir, manifest, source_hashes = run_metadata[run_id]
        logical_hash = hashlib.sha256(
            f"{source_hashes}:{run_id}:{person_id}".encode()
        ).hexdigest()
        dataset_id = hashlib.sha256(
            f"{series_id}:{logical_hash}".encode()
        ).hexdigest()[:24]
        records.append(
            DatasetRecord(
                dataset_id=dataset_id,
                source_domain="synthetic_truth_oracle",
                generator_commit=str(manifest["git_commit"]),
                config_sha256=str(manifest["config_sha256"]),
                run_id=run_id,
                person_key=f"{run_id}/{person_id}",
                time_column="timestamp_utc",
                schema_version=f"goal1.5/{manifest['schema_version']}",
                logical_hash=logical_hash,
                split_role=roles[(run_id, person_id)],
            )
        )

    registry_dir = registry_root / series_id
    registry_dir.mkdir(parents=True, exist_ok=False)
    records_path = registry_dir / "datasets.jsonl"
    records_text = "".join(
        json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n"
        for record in records
    )
    records_path.write_text(records_text, encoding="utf-8")
    split_path = registry_dir / "splits.parquet"
    split_rows = [
        {
            "series_id": series_id,
            "dataset_id": record.dataset_id,
            "run_id": record.run_id,
            "person_id": record.person_key.split("/", 1)[1],
            "person_key": record.person_key,
            "split_role": record.split_role,
            "logical_hash": record.logical_hash,
        }
        for record in records
    ]
    pq.write_table(
        pa.Table.from_pylist(split_rows),
        split_path,
        compression="zstd",
    )
    manifest_path = registry_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "series_id": series_id,
                "status": "oracle/sanity",
                "real_data_status": "NOT VERIFIED",
                "record_count": len(records),
                "records_sha256": hashlib.sha256(records_text.encode()).hexdigest(),
                "runs": [
                    {
                        "run_id": run_id,
                        "source_path": str(metadata[0]),
                        "config_sha256": metadata[1]["config_sha256"],
                        "generator_commit": metadata[1]["git_commit"],
                    }
                    for run_id, metadata in sorted(run_metadata.items())
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return RegisteredSeries(
        series_id=series_id,
        registry_dir=registry_dir,
        records_jsonl=records_path,
        split_parquet=split_path,
        manifest_json=manifest_path,
    )
