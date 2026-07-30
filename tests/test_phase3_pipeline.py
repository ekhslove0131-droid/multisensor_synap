import json
from pathlib import Path

import numpy as np
import pandas as pd

from multisensor_ml.contracts import ORACLE_LATENT_FACTORS
from multisensor_ml.phase3_pipeline import run_phase3_experiment


def quick_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    for person_number in range(36):
        role = (
            "train"
            if person_number < 24
            else "validation"
            if person_number < 30
            else "locked_test"
        )
        person_key = f"R{person_number // 12 + 1}:P{person_number:02d}"
        for second in range(12):
            event = int(6 <= second <= 7)
            baseline = (person_number % 6) * 0.4
            row: dict[str, object] = {
                "series_id": "quick-phase3",
                "dataset_id": f"dataset-{person_number:02d}",
                "run_id": f"R{person_number // 12 + 1}",
                "person_id": f"P{person_number:02d}",
                "person_key": person_key,
                "day_key": "2026-01-01",
                "session_id": f"{person_key}:day-1",
                "timestamp_utc": start + pd.Timedelta(seconds=second),
                "canonical_time": start + pd.Timedelta(seconds=second),
                "context": "focused_task" if second < 9 else "wake_rest",
                "is_awake": True,
                "event_binary": event,
                "hard_negative": int(second in {3, 4}),
                "split_role": role,
            }
            for factor_index, factor in enumerate(ORACLE_LATENT_FACTORS):
                direction = -1.0 if factor == "recovery_capacity" else 1.0
                row[factor] = (
                    baseline
                    + factor_index * 0.05
                    + direction * event * 2.0
                    + np.sin(second / 3) * 0.05
                )
            rows.append(row)
    return pd.DataFrame(rows)


def test_quick_phase3_pipeline_is_group_safe_and_deterministic(tmp_path: Path) -> None:
    output = run_phase3_experiment(
        quick_frame(),
        tmp_path / "phase3",
        warmup_sec=2,
        lookback_sec=8,
        refresh_sec=1,
        horizons_sec=(4, 8),
        random_state=20260730,
        windows=(3,),
        lags=(1,),
    )

    manifest = json.loads(output.manifest_json.read_text())
    metrics = pd.read_parquet(output.candidate_metrics_parquet)
    uplift = pd.read_parquet(output.group_uplift_parquet)
    predictions = pd.read_parquet(output.person_predictions_parquet)

    assert manifest["data_status"] == "oracle/sanity"
    assert manifest["real_accuracy_status"] == "NOT VERIFIED"
    assert manifest["locked_test_read"] is False
    assert manifest["split_person_counts"] == {"train": 24, "validation": 6}
    assert manifest["candidate_ids"] == [
        "G0",
        manifest["selected_p1"],
        manifest["selected_p2"],
        "P2+CL",
        "P2+CL-B",
    ]
    assert metrics["candidate_id"].nunique() == 5
    assert metrics["sampling_hash"].nunique() == 1
    assert set(predictions["split_role"]) == {"validation"}
    assert predictions["person_key"].nunique() == 6
    assert {"dataset_id", "person_key", "context", "load_state"}.issubset(
        set(uplift["group_dimension"])
    )
    assert not predictions["person_key"].str.contains("P3[0-5]").any()
    assert output.baseline_diagnostics_parquet.exists()
    assert output.cumulative_load_parquet.exists()
