from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from multisensor_synth.domain.types import (
    DailyContextRecord,
    EventRecord,
    ParticipantRecord,
)
from multisensor_synth.latent.state_process import LatentTimeline

SAFETY_METADATA = {
    b"synthetic": b"true",
    b"non_diagnostic": b"true",
    b"not_for_clinical_use": b"true",
    b"implemented_layer": b"truth",
}


PARTICIPANT_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("person_id", pa.string(), nullable=False),
        pa.field("age_years", pa.float32(), nullable=False),
        pa.field("biological_sex", pa.string(), nullable=False),
        pa.field("response_archetype", pa.string(), nullable=False),
        pa.field("resting_hr_bpm_truth", pa.float32(), nullable=False),
        pa.field("rmssd_ms_truth", pa.float32(), nullable=False),
        pa.field("eeg_aperiodic_exponent_truth", pa.float32(), nullable=False),
        pa.field("eeg_relative_alpha_truth", pa.float32(), nullable=False),
        pa.field("eeg_relative_gamma_truth", pa.float32(), nullable=False),
        pa.field("eda_tonic_level_us_truth", pa.float32(), nullable=False),
        pa.field("recovery_capacity_truth", pa.float32(), nullable=False),
        pa.field("opposite_response_tendency_truth", pa.float32(), nullable=False),
        pa.field("no_response_tendency_truth", pa.float32(), nullable=False),
        pa.field("seed", pa.uint64(), nullable=False),
    ],
    metadata=SAFETY_METADATA,
)

DAILY_CONTEXT_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("person_id", pa.string(), nullable=False),
        pa.field("day_index", pa.int8(), nullable=False),
        pa.field("date_utc", pa.date32(), nullable=False),
        pa.field("sleep_start_time_ns", pa.int64()),
        pa.field("sleep_end_time_ns", pa.int64()),
        pa.field("sleep_duration_sec", pa.int32(), nullable=False),
        pa.field("fatigue_drift_truth", pa.float32(), nullable=False),
        pa.field("activity_drift_truth", pa.float32(), nullable=False),
        pa.field("baseline_drift_truth", pa.float32(), nullable=False),
        pa.field("sensor_tolerance_truth", pa.float32(), nullable=False),
        pa.field("seed", pa.uint64(), nullable=False),
    ],
    metadata=SAFETY_METADATA,
)

EVENT_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("person_id", pa.string(), nullable=False),
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("event_type", pa.string(), nullable=False),
        pa.field("is_target", pa.bool_(), nullable=False),
        pa.field("hard_negative_kind", pa.string()),
        pa.field("start_time_ns", pa.int64(), nullable=False),
        pa.field("pre_late_start_time_ns", pa.int64()),
        pa.field("onset_time_ns", pa.int64()),
        pa.field("peak_start_time_ns", pa.int64()),
        pa.field("peak_end_time_ns", pa.int64()),
        pa.field("recovery_early_end_time_ns", pa.int64()),
        pa.field("recovery_late_end_time_ns", pa.int64()),
        pa.field("end_time_ns", pa.int64(), nullable=False),
        pa.field("intensity_truth", pa.float32(), nullable=False),
        pa.field("archetype", pa.string(), nullable=False),
        pa.field("is_artifact_only_schedule", pa.bool_(), nullable=False),
        pa.field("seed", pa.uint64(), nullable=False),
    ],
    metadata=SAFETY_METADATA,
)

LATENT_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("person_id", pa.string(), nullable=False),
        pa.field("truth_time_ns", pa.int64(), nullable=False),
        pa.field("timestamp_utc", pa.timestamp("ns", tz="UTC"), nullable=False),
        pa.field("date_utc", pa.date32(), nullable=False),
        pa.field("day_index", pa.int8(), nullable=False),
        pa.field("seconds_from_start", pa.int64(), nullable=False),
        pa.field("context_state", pa.string(), nullable=False),
        pa.field("is_awake", pa.bool_(), nullable=False),
        pa.field("autonomic_arousal", pa.float32(), nullable=False),
        pa.field("motor_activation", pa.float32(), nullable=False),
        pa.field("cognitive_load", pa.float32(), nullable=False),
        pa.field("sleep_pressure", pa.float32(), nullable=False),
        pa.field("sensory_context", pa.float32(), nullable=False),
        pa.field("recovery_capacity", pa.float32(), nullable=False),
        pa.field("social_context", pa.float32(), nullable=False),
        pa.field("active_target_event_id", pa.string()),
        pa.field("active_target_event_phase", pa.string()),
        pa.field("active_hard_negative_event_ids", pa.list_(pa.string()), nullable=False),
        pa.field("active_hard_negative_types", pa.list_(pa.string()), nullable=False),
        pa.field("artifact_only_schedule_ids", pa.list_(pa.string()), nullable=False),
    ],
    metadata=SAFETY_METADATA,
)


def write_records(
    path: Path,
    rows: list[ParticipantRecord] | list[DailyContextRecord] | list[EventRecord],
    schema: pa.Schema,
    compression: str | None,
) -> None:
    table = pa.Table.from_pylist([asdict(row) for row in rows], schema=schema)
    pq.write_table(
        table,
        path,
        compression=compression,
        version="2.6",
        write_statistics=True,
    )


def _timeline_table(
    timeline: LatentTimeline, start: int, end: int
) -> pa.Table:
    return pa.Table.from_arrays(
        [
            pa.array(timeline.run_id[start:end], type=pa.string()),
            pa.array(timeline.person_id[start:end], type=pa.string()),
            pa.array(timeline.truth_time_ns[start:end], type=pa.int64()),
            pa.array(timeline.timestamp_utc[start:end], type=pa.timestamp("ns", tz="UTC")),
            pa.array(timeline.date_utc[start:end], type=pa.date32()),
            pa.array(timeline.day_index[start:end], type=pa.int8()),
            pa.array(timeline.seconds_from_start[start:end], type=pa.int64()),
            pa.array(timeline.context_state[start:end], type=pa.string()),
            pa.array(timeline.is_awake[start:end], type=pa.bool_()),
            pa.array(timeline.autonomic_arousal[start:end], type=pa.float32()),
            pa.array(timeline.motor_activation[start:end], type=pa.float32()),
            pa.array(timeline.cognitive_load[start:end], type=pa.float32()),
            pa.array(timeline.sleep_pressure[start:end], type=pa.float32()),
            pa.array(timeline.sensory_context[start:end], type=pa.float32()),
            pa.array(timeline.recovery_capacity[start:end], type=pa.float32()),
            pa.array(timeline.social_context[start:end], type=pa.float32()),
            pa.array(timeline.active_target_event_id[start:end], type=pa.string()),
            pa.array(timeline.active_target_event_phase[start:end], type=pa.string()),
            pa.array(
                timeline.active_hard_negative_event_ids[start:end],
                type=pa.list_(pa.string()),
            ),
            pa.array(
                timeline.active_hard_negative_types[start:end],
                type=pa.list_(pa.string()),
            ),
            pa.array(
                timeline.artifact_only_schedule_ids[start:end],
                type=pa.list_(pa.string()),
            ),
        ],
        schema=LATENT_SCHEMA,
    )


class LatentParquetWriter:
    def __init__(self, path: Path, compression: str | None) -> None:
        self._writer = pq.ParquetWriter(
            path,
            LATENT_SCHEMA,
            compression=compression,
            version="2.6",
            write_statistics=True,
        )

    def write(self, timeline: LatentTimeline, chunk_size: int) -> None:
        for start in range(0, len(timeline), chunk_size):
            self._writer.write_table(
                _timeline_table(timeline, start, min(start + chunk_size, len(timeline)))
            )

    def close(self) -> None:
        self._writer.close()

    def __enter__(self) -> LatentParquetWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def logical_table_hash(table: pa.Table) -> str:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table.combine_chunks())
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def table_summary(path: Path) -> dict[str, object]:
    table = pq.read_table(path)
    return {
        "row_count": table.num_rows,
        "columns": table.schema.names,
        "logical_sha256": logical_table_hash(table),
    }
