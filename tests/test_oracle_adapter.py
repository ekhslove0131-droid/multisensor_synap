from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from multisensor_ml.contracts import ORACLE_LATENT_FACTORS
from multisensor_ml.oracle import read_oracle_person


def test_oracle_reader_projects_whitelist_before_building_independent_labels(
    tmp_path: Path,
) -> None:
    truth = tmp_path / "run-1" / "truth"
    truth.mkdir(parents=True)
    timeline: dict[str, object] = {
        "run_id": ["run-1"] * 100,
        "person_id": ["P001"] * 100,
        "timestamp_utc": pd.date_range("2026-01-01", periods=100, freq="s", tz="UTC"),
        "context_state": ["focused_task"] * 100,
        "is_awake": [True] * 100,
        "active_target_event_id": ["leak"] * 100,
        "active_target_event_phase": ["peak"] * 100,
        "active_hard_negative_event_ids": [["leak"]] * 100,
        "active_hard_negative_types": [["leak"]] * 100,
        "artifact_only_schedule_ids": [["leak"]] * 100,
    }
    for factor in ORACLE_LATENT_FACTORS:
        timeline[factor] = np.zeros(100, dtype=np.float32)
    pq.write_table(
        pa.Table.from_pandas(pd.DataFrame(timeline), preserve_index=False),
        truth / "latent_timeline.parquet",
    )
    events = pd.DataFrame(
        {
            "run_id": ["run-1", "run-1"],
            "person_id": ["P001", "P001"],
            "event_id": ["T1", "H1"],
            "is_target": [True, False],
            "hard_negative_kind": [None, "quiet_cognitive_load"],
            "start_time_ns": [20_000_000_000, 70_000_000_000],
            "onset_time_ns": [30_000_000_000, None],
            "end_time_ns": [40_000_000_000, 80_000_000_000],
        }
    )
    base_ns = pd.Timestamp("2026-01-01", tz="UTC").value
    events[["start_time_ns", "onset_time_ns", "end_time_ns"]] += base_ns
    pq.write_table(
        pa.Table.from_pandas(events, preserve_index=False),
        truth / "events.parquet",
    )

    result = read_oracle_person(tmp_path / "run-1", "P001")

    assert not any(column.startswith("active_") for column in result.frame)
    assert "artifact_only_schedule_ids" not in result.frame
    assert result.frame.loc[30:40, "event_binary"].eq(1).all()
    assert result.frame.loc[70:80, "hard_negative"].eq(1).all()
    assert result.feature_source_columns == (
        "timestamp_utc",
        "context",
        "is_awake",
        *ORACLE_LATENT_FACTORS,
    )
