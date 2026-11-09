"""Hidden truth label writer kept outside runtime packages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast


def labels_at(events: Sequence[Mapping[str, object]], subject_uuid: str,
              decision_ms: int) -> tuple[int, int | None]:
    subject_events = [event for event in events if event["subject_uuid"] == subject_uuid]
    current = int(any(cast(int, event["onset_ms"]) <= decision_ms
                      < cast(int, event["offset_ms"])
                      for event in subject_events))
    if current:
        return current, None
    onset60 = int(any(decision_ms < cast(int, event["onset_ms"]) <= decision_ms + 60_000
                      for event in subject_events))
    return current, onset60


def temporal_stage_at(events: Sequence[Mapping[str, object]], subject_uuid: str,
                      decision_ms: int) -> int:
    for event in events:
        if event["subject_uuid"] != subject_uuid:
            continue
        boundaries = (
            (1, cast(int, event["pre_early_start_ms"]),
             cast(int, event["pre_late_start_ms"])),
            (2, cast(int, event["pre_late_start_ms"]), cast(int, event["onset_ms"])),
            (3, cast(int, event["onset_ms"]), cast(int, event["offset_ms"])),
            (4, cast(int, event["recovery_early_start_ms"]),
             cast(int, event["recovery_late_start_ms"])),
            (5, cast(int, event["recovery_late_start_ms"]),
             cast(int, event["phase_end_ms"])),
        )
        for stage, start, end in boundaries:
            if start <= decision_ms < end:
                return stage
    return 0
