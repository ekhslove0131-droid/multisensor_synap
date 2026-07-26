from __future__ import annotations

CONTEXT_STATES = (
    "sleep",
    "wake_rest",
    "sedentary_activity",
    "light_activity",
    "moderate_activity",
    "focused_task",
    "meal_context",
    "transition",
)

TARGET_EVENT_TYPE = "multimodal_arousal_episode"

HARD_NEGATIVE_TYPES = (
    "ordinary_physical_activity",
    "quiet_cognitive_load",
    "sensor_artifact_episode",
    "recovery_without_peak",
    "false_alarm_like_episode",
)

FORBIDDEN_MEDICAL_EVENT_TERMS = (
    "meltdown",
    "aggression",
    "sensory_overload",
    "autism",
    "asd",
)
