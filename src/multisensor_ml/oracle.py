from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from multisensor_ml.contracts import (
    ORACLE_LATENT_FACTORS,
    assert_oracle_columns,
    build_labels,
)

_PARQUET_COLUMNS: tuple[str, ...] = (
    "run_id",
    "person_id",
    "timestamp_utc",
    "context_state",
    "is_awake",
    *ORACLE_LATENT_FACTORS,
)
ORACLE_FEATURE_SOURCE_COLUMNS: tuple[str, ...] = (
    "timestamp_utc",
    "context",
    "is_awake",
    *ORACLE_LATENT_FACTORS,
)


@dataclass(frozen=True, slots=True)
class OraclePerson:
    run_id: str
    person_id: str
    frame: pd.DataFrame
    events: pd.DataFrame
    feature_source_columns: tuple[str, ...] = ORACLE_FEATURE_SOURCE_COLUMNS


def _hard_negative_mask(timestamps: pd.Series, events: pd.DataFrame) -> np.ndarray:
    time_ns = (
        pd.to_datetime(timestamps, utc=True).dt.as_unit("ns").astype("int64").to_numpy()
    )
    mask = np.zeros(len(timestamps), dtype=np.int8)
    if "is_target" not in events:
        return mask
    hard_negatives = events.loc[~events["is_target"].astype(bool)]
    if "hard_negative_kind" in hard_negatives:
        hard_negatives = hard_negatives.loc[hard_negatives["hard_negative_kind"].notna()]
    starts = hard_negatives["start_time_ns"].astype("int64").to_numpy()
    ends = hard_negatives["end_time_ns"].astype("int64").to_numpy()
    for start, end in zip(starts, ends, strict=True):
        mask[(time_ns >= start) & (time_ns <= end)] = 1
    return mask


def read_oracle_person(run_dir: Path, person_id: str) -> OraclePerson:
    """Read only the approved oracle projection, then attach independently built labels."""

    truth_dir = run_dir.resolve() / "truth"
    timeline = pq.read_table(
        truth_dir / "latent_timeline.parquet",
        columns=list(_PARQUET_COLUMNS),
        filters=[("person_id", "=", person_id)],
    ).to_pandas()
    if timeline.empty:
        raise ValueError(f"person not found in timeline: {person_id}")
    timeline = timeline.rename(columns={"context_state": "context"})
    timeline = timeline.sort_values("timestamp_utc").reset_index(drop=True)
    assert_oracle_columns(list(ORACLE_FEATURE_SOURCE_COLUMNS))

    events = pq.read_table(
        truth_dir / "events.parquet",
        filters=[("person_id", "=", person_id)],
    ).to_pandas()
    labels = build_labels(timeline, events)
    timeline["event_binary"] = labels["event_binary"]
    timeline["forecast_60s"] = labels["forecast_60s"]
    timeline["phase"] = labels["phase"]
    timeline["hard_negative"] = _hard_negative_mask(timeline["timestamp_utc"], events)
    return OraclePerson(
        run_id=str(timeline["run_id"].iloc[0]),
        person_id=person_id,
        frame=timeline,
        events=events,
    )
