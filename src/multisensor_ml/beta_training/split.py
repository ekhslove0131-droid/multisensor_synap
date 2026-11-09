"""Subject-disjoint split validation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PreparedSplit:
    role: str
    subject_uuids: tuple[str, ...]


def validate_splits(rows: list[dict[str, str]]) -> tuple[PreparedSplit, ...]:
    allowed = ("TRAIN", "VALIDATION", "LOCKED_TEST")
    grouped: dict[str, list[str]] = {role: [] for role in allowed}
    seen: set[str] = set()
    for row in rows:
        if set(row) != {"subject_uuid", "split"} or row["split"] not in grouped:
            raise ValueError("split_row_invalid")
        if row["subject_uuid"] in seen:
            raise ValueError("split_subject_overlap")
        seen.add(row["subject_uuid"])
        grouped[row["split"]].append(row["subject_uuid"])
    if any(not grouped[role] for role in allowed):
        raise ValueError("split_role_empty")
    return tuple(PreparedSplit(role, tuple(grouped[role])) for role in allowed)
