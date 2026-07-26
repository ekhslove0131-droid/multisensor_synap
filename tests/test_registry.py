from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from multisensor_ml.registry import register_series


def _write_fake_run(root: Path, run_id: str, config_byte: str) -> Path:
    truth = root / run_id / "truth"
    truth.mkdir(parents=True)
    people = [f"P{index:03d}" for index in range(1, 13)]
    timeline = pd.DataFrame(
        {
            "run_id": [run_id] * 12,
            "person_id": people,
            "timestamp_utc": pd.date_range("2026-01-01", periods=12, tz="UTC"),
        }
    )
    pq.write_table(
        pa.Table.from_pandas(timeline, preserve_index=False),
        truth / "latent_timeline.parquet",
    )
    pq.write_table(
        pa.Table.from_pandas(pd.DataFrame({"run_id": [run_id], "event_id": ["E1"]})),
        truth / "events.parquet",
    )
    (root / run_id / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "config_sha256": config_byte * 64,
                "git_commit": "a" * 40,
                "schema_version": "1.0",
                "tables": {
                    "truth/latent_timeline.parquet": {
                        "logical_sha256": "b" * 64,
                    },
                    "truth/events.parquet": {"logical_sha256": "c" * 64},
                },
            }
        ),
        encoding="utf-8",
    )
    return root / run_id


def test_register_series_writes_immutable_person_lineage_and_split_table(tmp_path: Path) -> None:
    runs = [
        _write_fake_run(tmp_path / "runs", "mvp-one", "1"),
        _write_fake_run(tmp_path / "runs", "mvp-two", "2"),
        _write_fake_run(tmp_path / "runs", "mvp-three", "3"),
    ]

    result = register_series("mvp3-test", runs, tmp_path / "registry")

    records = [json.loads(line) for line in result.records_jsonl.read_text().splitlines()]
    split = pq.read_table(result.split_parquet).to_pandas()
    assert len(records) == 36
    assert split.groupby("split_role").size().to_dict() == {
        "locked_test": 6,
        "train": 24,
        "validation": 6,
    }
    assert split["person_key"].is_unique
    assert set(records[0]) == {
        "config_sha256",
        "dataset_id",
        "generator_commit",
        "logical_hash",
        "person_key",
        "run_id",
        "schema_version",
        "source_domain",
        "split_role",
        "time_column",
    }
    assert result.manifest_json.exists()


def test_register_series_refuses_duplicate_run_person_pairs(tmp_path: Path) -> None:
    run = _write_fake_run(tmp_path / "runs", "mvp-one", "1")

    try:
        register_series("duplicate", [run, run], tmp_path / "registry")
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate run/person registration was accepted")
