from __future__ import annotations

ARCHETYPE_MODALITY_ORDER: dict[str, tuple[str, ...]] = {
    "eeg_first": ("eeg", "autonomic", "motor"),
    "autonomic_first": ("autonomic", "eeg", "motor"),
    "motor_first": ("motor", "autonomic", "eeg"),
    "quiet_internal": ("eeg", "autonomic"),
    "motor_dominant": ("motor",),
    "partial_response": ("autonomic", "eeg"),
    "non_responder": (),
}

ARCHETYPE_MODALITY_STRENGTH: dict[str, tuple[float, float, float]] = {
    "eeg_first": (0.8, 0.45, 0.35),
    "autonomic_first": (1.0, 0.5, 0.5),
    "motor_first": (0.65, 1.0, 0.35),
    "quiet_internal": (0.75, 0.08, 0.8),
    "motor_dominant": (0.35, 1.0, 0.2),
    "partial_response": (0.5, 0.15, 0.5),
    "non_responder": (0.05, 0.05, 0.05),
}
