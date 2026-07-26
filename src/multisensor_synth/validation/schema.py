from __future__ import annotations

REQUIRED_TRUTH_FILES = (
    "truth/participants.parquet",
    "truth/daily_context.parquet",
    "truth/latent_timeline.parquet",
    "truth/events.parquet",
)

FORBIDDEN_GOAL_ONE_DIRECTORIES = ("observed", "model_ready", "clips")
