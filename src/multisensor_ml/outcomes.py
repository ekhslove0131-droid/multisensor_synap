from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

STAGE_CODES: tuple[str, ...] = (
    "NO_EVENT",
    "LOW",
    "MEDIUM",
    "HIGH",
    "DECREASING",
    "RECOVERY",
)
BEHAVIOR_CODES: frozenset[str] = frozenset(
    {
        "ear_covering",
        "head_turn_away",
        "withdrawal_movement",
        "exit_attempt",
        "repetitive_hand_movement",
        "repetitive_body_movement",
        "movement_reduction",
        "motion_freeze",
        "repetitive_object_contact",
        "sustained_pressure_or_contact",
    }
)
REVIEW_TAGS: frozenset[str] = frozenset(
    {
        "reported_meltdown_like",
        "reported_sensory_seeking_like",
    }
)

_TARGET_BASE_PROBABILITY: dict[str, float] = {
    "ear_covering": 0.28,
    "head_turn_away": 0.35,
    "withdrawal_movement": 0.32,
    "exit_attempt": 0.16,
    "repetitive_hand_movement": 0.38,
    "repetitive_body_movement": 0.34,
    "movement_reduction": 0.22,
    "motion_freeze": 0.18,
    "repetitive_object_contact": 0.24,
    "sustained_pressure_or_contact": 0.20,
}


class BehaviorOntology(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["goal1.5/behavior-ontology/v1"]
    behavior_codes: tuple[str, ...] = Field(min_length=1)
    review_tags: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def separate_training_codes_from_review_terms(self) -> BehaviorOntology:
        forbidden = set(self.behavior_codes) & {
            "meltdown",
            "sensory_seeking",
            "reported_meltdown_like",
            "reported_sensory_seeking_like",
        }
        if forbidden:
            raise ValueError(
                "review terms are forbidden behavior codes: "
                + ", ".join(sorted(forbidden))
            )
        if len(set(self.behavior_codes)) != len(self.behavior_codes):
            raise ValueError("behavior codes must be unique")
        return self


def load_behavior_ontology(path: Path) -> BehaviorOntology:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("behavior ontology root must be a mapping")
    ontology = BehaviorOntology.model_validate(payload)
    if set(ontology.behavior_codes) != BEHAVIOR_CODES:
        raise ValueError("behavior ontology codes do not match the v1 runtime contract")
    if set(ontology.review_tags) != REVIEW_TAGS:
        raise ValueError("behavior ontology review tags do not match the v1 contract")
    return ontology


def _stable_unit_interval(*parts: object) -> float:
    material = "\0".join(str(part) for part in parts).encode()
    value = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return value / float(2**64)


def _behavior_probability(event: dict[str, object], code: str) -> float:
    is_target = bool(event.get("is_target", False))
    raw_intensity = event.get("intensity_truth", 0.0)
    intensity = (
        float(raw_intensity)
        if isinstance(raw_intensity, (int, float, np.integer, np.floating))
        else 0.0
    )
    archetype = str(event.get("archetype", ""))
    context = str(event.get("context", ""))
    if is_target:
        probability = _TARGET_BASE_PROBABILITY[code] + 0.18 * intensity
    else:
        kind = str(event.get("hard_negative_kind", ""))
        probability = 0.08
        if kind == "ordinary_physical_activity" and code in {
            "repetitive_body_movement",
            "withdrawal_movement",
        }:
            probability = 0.34
        elif kind == "quiet_cognitive_load" and code in {
            "movement_reduction",
            "motion_freeze",
        }:
            probability = 0.30
        elif kind == "recovery_without_peak" and code in {
            "head_turn_away",
            "movement_reduction",
        }:
            probability = 0.25
    if archetype in {"motor_first", "motor_dominant"} and code in {
        "repetitive_hand_movement",
        "repetitive_body_movement",
        "exit_attempt",
    }:
        probability += 0.16
    if archetype == "quiet_internal" and code in {
        "movement_reduction",
        "motion_freeze",
    }:
        probability += 0.16
    if context in {"light_activity", "moderate_activity"} and code in {
        "repetitive_body_movement",
        "withdrawal_movement",
    }:
        probability += 0.08
    return float(np.clip(probability, 0.03, 0.82))


def build_behavior_labels(
    events: pd.DataFrame,
    *,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate deterministic neutral behavior labels and separate review tags."""

    behavior_rows: list[dict[str, object]] = []
    tag_rows: list[dict[str, object]] = []
    records = events.sort_values(["run_id", "person_id", "event_id"]).to_dict(
        orient="records"
    )
    for raw_event in records:
        event: dict[str, object] = {
            str(key): value for key, value in raw_event.items()
        }
        event_id = str(event["event_id"])
        selected: set[str] = set()
        for code in sorted(BEHAVIOR_CODES):
            probability = _behavior_probability(event, code)
            if _stable_unit_interval(random_state, event_id, code) < probability:
                selected.add(code)
                behavior_rows.append(
                    {
                        "run_id": str(event["run_id"]),
                        "person_id": str(event["person_id"]),
                        "event_id": event_id,
                        "behavior_code": code,
                        "label_value": 1,
                        "label_source": "synthetic_rule_v1",
                        "label_confidence": np.float32(1.0),
                    }
                )

        meltdown_evidence = len(
            selected
            & {
                "ear_covering",
                "withdrawal_movement",
                "exit_attempt",
                "movement_reduction",
                "motion_freeze",
            }
        )
        sensory_evidence = len(
            selected
            & {
                "repetitive_hand_movement",
                "repetitive_body_movement",
                "repetitive_object_contact",
                "sustained_pressure_or_contact",
            }
        )
        tag_probabilities = {
            "reported_meltdown_like": min(0.75, 0.08 + 0.18 * meltdown_evidence),
            "reported_sensory_seeking_like": min(
                0.75, 0.08 + 0.18 * sensory_evidence
            ),
        }
        for tag, probability in tag_probabilities.items():
            if _stable_unit_interval(random_state, event_id, tag) < probability:
                tag_rows.append(
                    {
                        "run_id": str(event["run_id"]),
                        "person_id": str(event["person_id"]),
                        "event_id": event_id,
                        "review_tag": tag,
                        "label_value": 1,
                        "label_source": "synthetic_observer_v1",
                        "label_confidence": np.float32(
                            0.55 + 0.4 * min(1.0, probability)
                        ),
                    }
                )
    behavior_columns = [
        "run_id",
        "person_id",
        "event_id",
        "behavior_code",
        "label_value",
        "label_source",
        "label_confidence",
    ]
    tag_columns = [
        "run_id",
        "person_id",
        "event_id",
        "review_tag",
        "label_value",
        "label_source",
        "label_confidence",
    ]
    return (
        pd.DataFrame(behavior_rows, columns=behavior_columns),
        pd.DataFrame(tag_rows, columns=tag_columns),
    )


def build_stage_labels(timeline: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Map independent target-event boundaries onto a canonical one-second timeline."""

    required_timeline = {"run_id", "person_id", "timestamp_utc"}
    missing_timeline = required_timeline.difference(timeline.columns)
    if missing_timeline:
        raise ValueError(f"timeline missing columns: {sorted(missing_timeline)}")
    time_ns = (
        pd.to_datetime(timeline["timestamp_utc"], utc=True)
        .dt.as_unit("ns")
        .astype("int64")
        .to_numpy()
    )
    stage = np.full(len(timeline), "NO_EVENT", dtype=object)
    event_ids = np.full(len(timeline), None, dtype=object)
    targets = events.loc[events["is_target"].astype(bool)] if "is_target" in events else events
    phase_ranges = (
        ("LOW", "start_time_ns", "pre_late_start_time_ns"),
        ("MEDIUM", "pre_late_start_time_ns", "peak_start_time_ns"),
        ("HIGH", "peak_start_time_ns", "peak_end_time_ns"),
        ("DECREASING", "peak_end_time_ns", "recovery_early_end_time_ns"),
        ("RECOVERY", "recovery_early_end_time_ns", "end_time_ns"),
    )
    for row in targets.to_dict(orient="records"):
        event_id = str(row["event_id"])
        for code, start_column, end_column in phase_ranges:
            start = row.get(start_column)
            end = row.get(end_column)
            if pd.isna(start) or pd.isna(end):
                continue
            mask = (time_ns >= int(start)) & (time_ns < int(end))
            stage[mask] = code
            event_ids[mask] = event_id
    return pd.DataFrame(
        {
            "run_id": timeline["run_id"].to_numpy(),
            "person_id": timeline["person_id"].to_numpy(),
            "timestamp_utc": pd.to_datetime(timeline["timestamp_utc"], utc=True),
            "event_id": pd.Series(event_ids.tolist(), dtype=object, index=timeline.index),
            "stage_code": stage,
        },
        index=timeline.index,
    )
