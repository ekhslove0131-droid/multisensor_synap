from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from multisensor_ml.model_comparison import (
    build_comparison_artifact,
    collect_comparison_metrics,
    render_comparison_html,
)


def _write_metric(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_collect_comparison_metrics_keeps_validation_and_oracle_boundary(tmp_path: Path) -> None:
    root = tmp_path / "project"
    availability = root / "artifacts/availability-model/oracle-sanity-v3/extracted/profiles"
    for profile in (
        "watch_only",
        "polar_only",
        "muse_only",
        "watch_polar",
        "watch_muse",
        "polar_muse",
        "watch_polar_muse",
    ):
        _write_metric(
            availability / profile / "validation_metrics.parquet",
            [
                {
                    "head": "event",
                    "model_name": "logistic_regression",
                    "aucpr": 0.6,
                    "event_recall": 0.7,
                    "event_f1": 0.65,
                    "false_alerts_per_hour": 0.01,
                    "brier_score": 0.2,
                    "calibration_error": 0.05,
                    "macro_f1": None,
                },
                {
                    "head": "stage",
                    "model_name": "logistic_regression",
                    "aucpr": None,
                    "event_recall": None,
                    "event_f1": None,
                    "false_alerts_per_hour": None,
                    "brier_score": None,
                    "calibration_error": None,
                    "macro_f1": 0.5,
                },
            ],
        )
    _write_metric(
        root / "artifacts/registry/mvp3-oracle-v1/stage-model/validation_metrics.parquet",
        [
            {
                "head": "event",
                "model_name": "logistic_regression",
                "aucpr": 0.2,
                "event_recall": 0.4,
                "event_f1": 0.3,
                "false_alerts_per_hour": 0.1,
                "brier_score": 0.3,
                "calibration_error": 0.2,
                "macro_f1": None,
            },
            {
                "head": "event",
                "model_name": "hist_gradient_boosting",
                "aucpr": 0.4,
                "event_recall": 0.5,
                "event_f1": 0.45,
                "false_alerts_per_hour": 0.02,
                "brier_score": 0.1,
                "calibration_error": 0.1,
                "macro_f1": None,
            },
            {
                "head": "stage",
                "model_name": "logistic_regression",
                "aucpr": None,
                "event_recall": None,
                "event_f1": None,
                "false_alerts_per_hour": None,
                "brier_score": None,
                "calibration_error": None,
                "macro_f1": 0.3,
            },
            {
                "head": "stage",
                "model_name": "hist_gradient_boosting",
                "aucpr": None,
                "event_recall": None,
                "event_f1": None,
                "false_alerts_per_hour": None,
                "brier_score": None,
                "calibration_error": None,
                "macro_f1": 0.4,
            },
        ],
    )
    frame = collect_comparison_metrics(root)
    assert set(frame["data_status"]) == {"oracle/sanity"}
    assert set(frame["role"]) == {"validation"}
    assert len(frame) == 18
    assert set(frame["model_id"]) >= {"watch_only", "hierarchical_hgb"}


def test_comparison_artifact_and_korean_html_are_self_contained(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "model_id": "watch_only",
                "model_label_ko": "Watch 단독",
                "head": "event",
                "model_name": "logistic_regression",
                "sensor_profile": "watch_only",
                "role": "validation",
                "data_status": "oracle/sanity",
                "aucpr": 0.7,
                "event_recall": 0.8,
                "event_f1": 0.75,
                "false_alerts_per_hour": 0.01,
                "brier_score": 0.2,
                "calibration_error": 0.05,
                "macro_f1": None,
            },
            {
                "model_id": "watch_only",
                "model_label_ko": "Watch 단독",
                "head": "stage",
                "model_name": "logistic_regression",
                "sensor_profile": "watch_only",
                "role": "validation",
                "data_status": "oracle/sanity",
                "aucpr": None,
                "event_recall": None,
                "event_f1": None,
                "false_alerts_per_hour": None,
                "brier_score": None,
                "calibration_error": None,
                "macro_f1": 0.5,
            },
        ]
    )
    artifact = build_comparison_artifact(frame, generated_at="2026-08-02T00:00:00+09:00")
    assert artifact["manifest"]["language"] == "ko"
    assert len(artifact["manifest"]["charts"]) == 6
    assert artifact["snapshot"]["datasets"]["model_comparison"]

    output = tmp_path / "comparison.html"
    render_comparison_html(artifact, output, font_family="NanumGothic")
    html = output.read_text(encoding="utf-8")
    assert "lang=\"ko\"" in html
    assert "NanumGothic" in html
    assert "AUCPR" in html
    assert "<svg" in html
    json.loads((tmp_path / "comparison.artifact.json").read_text(encoding="utf-8"))
