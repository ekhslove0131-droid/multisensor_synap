from __future__ import annotations

import math
from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from multisensor_synth.config.models import ProjectConfig
from multisensor_synth.domain.seeds import SeedTree
from multisensor_synth.domain.types import DailyContextRecord, ParticipantRecord

DAY_SECONDS = 86_400
NANOSECONDS = 1_000_000_000


def sample_sleep_hours(
    config: ProjectConfig, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    bedtime = float(
        np.clip(
            rng.normal(
                config.context.sleep.bedtime_local_hour.mean,
                config.context.sleep.bedtime_local_hour.std,
            ),
            18,
            27,
        )
    )
    duration = float(
        np.clip(
            rng.normal(
                config.context.sleep.duration_hours.mean,
                config.context.sleep.duration_hours.std,
            ),
            5,
            12,
        )
    )
    return bedtime % 24, duration


def _local_datetime(local_date: datetime, hour_value: float, zone: ZoneInfo) -> datetime:
    hour = int(hour_value)
    minute = round((hour_value - hour) * 60) % 60
    return datetime.combine(local_date.date(), time(hour=hour, minute=minute), tzinfo=zone)


def generate_daily_contexts(
    config: ProjectConfig,
    participants: list[ParticipantRecord],
    seeds: SeedTree,
    run_id: str,
) -> list[DailyContextRecord]:
    run_start = config.run.start_time_utc
    run_start_ns = int(run_start.timestamp() * NANOSECONDS)
    run_end_ns = run_start_ns + config.run.duration_sec * NANOSECONDS
    day_count = math.ceil(config.run.duration_sec / DAY_SECONDS)
    zone = ZoneInfo(config.run.timezone_context)
    rows: list[DailyContextRecord] = []

    for participant in participants:
        for day_index in range(day_count):
            namespace = f"person/{participant.person_id}/day/{day_index}"
            seed = seeds.child_seed(namespace)
            bedtime_hour, duration_hour = sample_sleep_hours(config, seed)
            rng = np.random.default_rng(seed)
            rng.normal()
            rng.normal()
            utc_day_start = run_start + timedelta(days=day_index)
            local_reference = utc_day_start.astimezone(zone)
            sleep_start = _local_datetime(local_reference, bedtime_hour, zone)
            sleep_end = sleep_start + timedelta(hours=duration_hour)
            sleep_start_ns = int(sleep_start.astimezone(UTC).timestamp() * NANOSECONDS)
            sleep_end_ns = int(sleep_end.astimezone(UTC).timestamp() * NANOSECONDS)

            rows.append(
                DailyContextRecord(
                    run_id=run_id,
                    person_id=participant.person_id,
                    day_index=day_index,
                    date_utc=utc_day_start.date(),
                    sleep_start_time_ns=(
                        sleep_start_ns if run_start_ns <= sleep_start_ns < run_end_ns else None
                    ),
                    sleep_end_time_ns=(
                        sleep_end_ns if run_start_ns <= sleep_end_ns < run_end_ns else None
                    ),
                    sleep_duration_sec=round(duration_hour * 3600),
                    fatigue_drift_truth=float(np.float32(np.clip(rng.normal(0, 0.08), -0.3, 0.3))),
                    activity_drift_truth=float(
                        np.float32(np.clip(rng.normal(0, 0.08), -0.3, 0.3))
                    ),
                    baseline_drift_truth=float(
                        np.float32(np.clip(rng.normal(0, 0.05), -0.2, 0.2))
                    ),
                    sensor_tolerance_truth=float(
                        np.float32(np.clip(rng.beta(5, 2), 0, 1))
                    ),
                    seed=seed,
                )
            )
    return rows
