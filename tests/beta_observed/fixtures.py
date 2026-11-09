from __future__ import annotations

from multisensor_synth.beta_observed.config import BetaConfig, BetaSignalState


def beta_config(*, duration_seconds: int = 30, subject_count: int = 2) -> BetaConfig:
    return BetaConfig(
        seed=4242,
        subject_count=subject_count,
        duration_seconds=duration_seconds,
        prefill_seconds=10,
        calibration_seconds=10,
        scored_seconds=duration_seconds - 20,
        train_subjects=1,
        validation_subjects=1 if subject_count > 1 else 0,
        test_subjects=subject_count - 2 if subject_count > 2 else 0,
        events_per_subject=1,
    )


def beta_state(*, seed: int = 4242) -> BetaSignalState:
    return BetaSignalState.fixture(seed=seed)
