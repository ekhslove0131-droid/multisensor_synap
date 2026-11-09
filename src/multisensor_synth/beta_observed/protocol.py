"""Frozen generation assumptions; research priors are not clinical effect sizes."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .codec import canonical_bytes


def _generator_source_sha256() -> str:
    root = Path(__file__).resolve().parent
    names = ("cardiac.py", "eda.py", "eeg.py", "generator.py", "exporter.py")
    material = b"".join(name.encode() + b"\0" + (root / name).read_bytes()
                        for name in names)
    return hashlib.sha256(material).hexdigest()


PROTOCOL = {
    "schema_version": "synthetic_beta_generation_protocol_v2",
    "realm": "SYNTHETIC_BETA",
    "research_prior": {
        "doi": "10.1145/3240925.3240980",
        "use": "preceding-physiology timing inspiration only",
        "effect_size_imported": False,
    },
    "morphology": {
        "ecg_template": "NeuroKit2 ecg_simulate ecgsyn",
        "neurokit_version": "0.2.13",
        "template_seed": 20261006,
        "template_normalization": "median_center_then_max_abs",
        "ecg_unit_conversion": "normalized_template_x_950_uV",
        "eda_unit_conversion": "NeuroKit_template_normalized_x_0.05_uS",
        "ppg": {"baseline_au": 0.2, "pulse_sd_cycles": 0.08,
                "ir_gain": 0.85, "red_gain": 0.72},
    },
    "response_gain_bpm": {"RESPONSIVE": 18.0, "WEAK": 7.0,
                          "INVERSE": -9.0, "NO_RESPONSE": 0.0},
    "lags_ms": {"ecg": 0, "ppg": 200, "sample_availability": 50,
                "interval_availability": 80},
    "noise": {"ecg_uV_sd": 12.0, "ppg_au_sd": 0.015, "eeg_uV_sd": 2.5,
              "label_dependent": False, "negative_positive_same_process": True},
    "physiology": {
        "eda": {"baseline_uS": 4.0, "response_gain_uS": 1.2},
        "temperature": {"baseline_degC": 33.0, "response_gain_degC": -0.2,
                        "drift_amplitude_degC": 0.05, "drift_period_radians_s": 120},
        "eeg": {"alpha_base_uV": 13.0, "alpha_response_gain_uV": -3.0,
                "beta_base_uV": 5.0, "beta_response_gain_uV": 4.0,
                "theta_uV": 7.0},
    },
    "population": {
        "baseline_hr_bpm": "66_plus_subject_index_mod_9",
        "response_assignment": "subject_index_mod_4",
        "response_order": ["RESPONSIVE", "WEAK", "INVERSE", "NO_RESPONSE"],
    },
    "event_schedule": {
        "primary_scored_start_offsets_seconds": [300, 900],
        "primary_minimum_scored_seconds": 1200,
        "fallback_first_offset_seconds": 2,
        "fallback_step_seconds": 4,
        "duration_seconds": 180,
        "evaluation_phase_offsets_seconds": {
            "pre_early": [-60, -30], "pre_late": [-30, 0],
            "peak": [0, 180], "recovery_early": [180, 240],
            "recovery_late": [240, 300],
        },
    },
    "context": {"exercise_seconds": [2700, 2820], "exercise_gain_bpm": 25.0,
                "motion_baseline_g": 0.02, "motion_exercise_g": 0.35,
                "dropout_seconds": [3300, 3360],
                "dropout_assignment": "subject_index_mod_6_eq_5",
                "dropout_label_dependent": False},
    "clock": {"sample_time": "integer_us", "availability": "integer_ms",
              "window_ms": 10000,
              "causal_beat_state": "bounded_previous_observed_chunk_cache"},
    "generator_source_sha256": _generator_source_sha256(),
}
PROTOCOL_SHA256 = hashlib.sha256(canonical_bytes(PROTOCOL)).hexdigest()
