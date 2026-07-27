from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from multisensor_ml.personalization import (
    classify_ood,
    discover_standard_types,
    fit_standard_type_artifacts,
    select_adaptation_cap,
)


def test_standard_type_discovery_selects_two_stable_train_clusters() -> None:
    rng = np.random.default_rng(17)
    frame = pd.DataFrame(
        {
            "person_key": [f"P{index:03d}" for index in range(24)],
            "baseline_a": np.concatenate(
                [rng.normal(-4, 0.15, 12), rng.normal(4, 0.15, 12)]
            ),
            "baseline_b": np.concatenate(
                [rng.normal(-3, 0.15, 12), rng.normal(3, 0.15, 12)]
            ),
        }
    )

    result = discover_standard_types(frame, random_state=17)

    assert result.selected_k == 2
    assert list(result.memberships.columns) == ["person_key", "STD-A", "STD-B"]
    np.testing.assert_allclose(
        result.memberships[["STD-A", "STD-B"]].sum(axis=1),
        1,
        atol=1e-6,
    )


def test_standard_type_discovery_falls_back_to_one_type_without_separation() -> None:
    frame = pd.DataFrame(
        {
            "person_key": [f"P{index:03d}" for index in range(12)],
            "baseline_a": np.ones(12),
            "baseline_b": np.ones(12),
        }
    )

    result = discover_standard_types(frame, random_state=17)

    assert result.selected_k == 1
    assert result.memberships["STD-A"].eq(1.0).all()


def test_adaptation_cap_prefers_smallest_near_best_safe_candidate() -> None:
    metrics = pd.DataFrame(
        [
            {
                "cap": 0.00,
                "aucpr": 0.50,
                "event_recall": 0.70,
                "event_f1": 0.60,
                "false_alerts_per_hour": 0.10,
                "ece": 0.05,
                "lead_time_sec": 30,
            },
            {
                "cap": 0.20,
                "aucpr": 0.52,
                "event_recall": 0.71,
                "event_f1": 0.64,
                "false_alerts_per_hour": 0.105,
                "ece": 0.055,
                "lead_time_sec": 31,
            },
            {
                "cap": 0.35,
                "aucpr": 0.52,
                "event_recall": 0.71,
                "event_f1": 0.645,
                "false_alerts_per_hour": 0.106,
                "ece": 0.056,
                "lead_time_sec": 31,
            },
            {
                "cap": 0.50,
                "aucpr": 0.45,
                "event_recall": 0.60,
                "event_f1": 0.55,
                "false_alerts_per_hour": 0.20,
                "ece": 0.09,
                "lead_time_sec": 20,
            },
        ]
    )

    assert select_adaptation_cap(metrics) == 0.20


def test_ood_requires_persistent_multi_day_evidence_for_retraining() -> None:
    assert (
        classify_ood(
            distance=10,
            p95=3,
            p99=5,
            quality_valid=True,
            adaptation_capped=False,
            persistent_sessions=2,
            distinct_days=2,
            valid_seconds=4000,
        )
        == "OOD_MONITOR"
    )
    assert (
        classify_ood(
            distance=10,
            p95=3,
            p99=5,
            quality_valid=True,
            adaptation_capped=False,
            persistent_sessions=3,
            distinct_days=2,
            valid_seconds=1800,
        )
        == "RETRAIN_CANDIDATE"
    )
    assert (
        classify_ood(
            distance=1,
            p95=3,
            p99=5,
            quality_valid=False,
            adaptation_capped=False,
            persistent_sessions=0,
            distinct_days=0,
            valid_seconds=0,
        )
        == "NOT_DECISIONABLE"
    )


def test_standard_type_artifacts_fit_train_people_and_score_all_people(
    tmp_path: Path,
) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    people = [
        {
            "person_key": f"run/P{index:03d}",
            "split_role": "train" if index < 12 else "validation",
        }
        for index in range(16)
    ]
    (prepared / "manifest.json").write_text(
        __import__("json").dumps({"people": people})
    )
    baseline = pd.DataFrame(
        {
            "person_key": [row["person_key"] for row in people],
            "factor_a__center": [-4.0] * 6 + [4.0] * 6 + [-3.8] * 2 + [3.8] * 2,
            "factor_a__mad": [0.4] * 16,
            "factor_b__center": [-3.0] * 6 + [3.0] * 6 + [-2.8] * 2 + [2.8] * 2,
            "factor_b__mad": [0.3] * 16,
        }
    )
    pq.write_table(
        pa.Table.from_pandas(baseline, preserve_index=False),
        prepared / "personal_baseline.parquet",
    )

    result = fit_standard_type_artifacts(
        prepared,
        tmp_path / "types",
        random_state=17,
    )

    assert result.selected_k == 2
    memberships = pq.read_table(
        result.root / "person_standard_types.parquet"
    ).to_pandas()
    assert len(memberships) == 16
    assert (result.root / "standard_type_model.skops").is_file()
    assert (result.root / "ood_status.parquet").is_file()
