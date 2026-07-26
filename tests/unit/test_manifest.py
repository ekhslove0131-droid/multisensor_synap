from __future__ import annotations

from multisensor_synth.export.manifest import deterministic_manifest_subset


def test_deterministic_manifest_subset_excludes_volatile_fields() -> None:
    manifest = {
        "run_id": "quick-abc",
        "config_sha256": "abc",
        "seed": 7,
        "run_contract": {
            "participant_count": 3,
            "duration_sec": 21_600,
            "canonical_rate_hz": 1,
            "minimum_gap_between_target_sec": 1800,
        },
        "created_at_utc": "2026-01-01T00:00:00Z",
        "wall_clock_duration_sec": 12.3,
        "output_path": "/tmp/one",
        "child_seed_namespaces": {"population": 42},
        "tables": {"truth/events.parquet": {"logical_sha256": "def"}},
    }

    subset = deterministic_manifest_subset(manifest)

    assert subset == {
        "run_id": "quick-abc",
        "config_sha256": "abc",
        "seed": 7,
        "run_contract": {
            "participant_count": 3,
            "duration_sec": 21_600,
            "canonical_rate_hz": 1,
            "minimum_gap_between_target_sec": 1800,
        },
        "child_seed_namespaces": {"population": 42},
        "tables": {"truth/events.parquet": {"logical_sha256": "def"}},
    }
