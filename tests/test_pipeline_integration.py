from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from multisensor_ml.bundle import load_model_bundle
from multisensor_ml.contracts import ORACLE_LATENT_FACTORS
from multisensor_ml.knime import export_knime_artifacts
from multisensor_ml.pipeline import (
    evaluate_prepared_bundle,
    prepare_series,
    train_prepared,
)
from multisensor_ml.registry import register_series


def _integration_run(root: Path) -> Path:
    run = root / "oracle-integration"
    truth = run / "truth"
    truth.mkdir(parents=True)
    start = pd.Timestamp("2026-01-01", tz="UTC")
    frames: list[pd.DataFrame] = []
    events: list[dict[str, object]] = []
    for person_number in range(1, 13):
        person_id = f"P{person_number:03d}"
        rows = 1900
        seconds = np.arange(rows)
        frame: dict[str, object] = {
            "run_id": ["oracle-integration"] * rows,
            "person_id": [person_id] * rows,
            "timestamp_utc": pd.date_range(start, periods=rows, freq="s"),
            "context_state": ["focused_task"] * rows,
            "is_awake": [True] * rows,
        }
        for factor_number, factor in enumerate(ORACLE_LATENT_FACTORS):
            values = np.sin(seconds / (20 + factor_number)).astype(np.float32)
            if factor_number == 0:
                values[1810:1821] += 3
            frame[factor] = values
        frames.append(pd.DataFrame(frame))
        base_ns = start.value
        events.extend(
            [
                {
                    "run_id": "oracle-integration",
                    "person_id": person_id,
                    "event_id": f"{person_id}-T1",
                    "is_target": True,
                    "hard_negative_kind": None,
                    "start_time_ns": base_ns + 1800 * 1_000_000_000,
                    "onset_time_ns": base_ns + 1810 * 1_000_000_000,
                    "end_time_ns": base_ns + 1820 * 1_000_000_000,
                },
                {
                    "run_id": "oracle-integration",
                    "person_id": person_id,
                    "event_id": f"{person_id}-H1",
                    "is_target": False,
                    "hard_negative_kind": "quiet_cognitive_load",
                    "start_time_ns": base_ns + 1850 * 1_000_000_000,
                    "onset_time_ns": None,
                    "end_time_ns": base_ns + 1860 * 1_000_000_000,
                },
            ]
        )
    pq.write_table(
        pa.Table.from_pandas(pd.concat(frames), preserve_index=False),
        truth / "latent_timeline.parquet",
    )
    pq.write_table(
        pa.Table.from_pandas(pd.DataFrame(events), preserve_index=False),
        truth / "events.parquet",
    )
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "oracle-integration",
                "config_sha256": "b" * 64,
                "git_commit": "a" * 40,
                "schema_version": "1.0",
                "tables": {
                    "truth/latent_timeline.parquet": {"logical_sha256": "c" * 64},
                    "truth/events.parquet": {"logical_sha256": "d" * 64},
                },
            }
        ),
        encoding="utf-8",
    )
    return run


def test_prepare_train_bundle_reload_full_goal15_path(tmp_path: Path) -> None:
    run = _integration_run(tmp_path / "source")
    series = register_series("integration", [run], tmp_path / "registry")

    prepared = prepare_series(series.registry_dir, tmp_path / "prepared")
    bundle = train_prepared(
        prepared.root,
        tmp_path / "bundle",
        random_state=23,
        locked_test_audit_reason="integration acceptance verification",
        uv_lock=Path(__file__).parents[1] / "uv.lock",
    )
    loaded = load_model_bundle(bundle)

    assert set(loaded.models) == {
        ("event_binary", "hist_gradient_boosting"),
        ("event_binary", "logistic_regression"),
        ("forecast_60s", "hist_gradient_boosting"),
        ("forecast_60s", "logistic_regression"),
    }
    metrics = pq.read_table(bundle / "metrics.parquet").to_pandas()
    assert {"validation", "locked_test", "stress"}.issubset(set(metrics["role"]))
    assert {
        "aucpr",
        "brier_score",
        "calibration_error",
        "event_recall",
        "false_alerts_per_hour",
    }.issubset(set(metrics["metric"]))
    sync = pq.read_table(bundle / "synchronization.parquet").to_pandas()
    assert sync["status"].tolist() == ["NOT_AVAILABLE_TRUTH_ONLY"]
    assert loaded.manifest["real_data_status"] == "NOT VERIFIED"

    try:
        evaluate_prepared_bundle(
            bundle,
            prepared.root,
            role="locked_test",
            audit_reason="attempted duplicate integration verification",
        )
    except ValueError as exc:
        assert "duplicate locked_test" in str(exc)
    else:
        raise AssertionError("duplicate locked_test evaluation was accepted")

    export = export_knime_artifacts(bundle, prepared.root, tmp_path / "knime-export")
    assert (export / "dataset_split.parquet").exists()
    assert (export / "pattern_performance.parquet").exists()
    assert (export / "noise_stress.parquet").exists()
    assert (export / "dashboard_manifest.json").exists()
