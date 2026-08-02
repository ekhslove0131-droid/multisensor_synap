#!/usr/bin/env python3
# ruff: noqa: E402
"""Run the XGBoost/LightGBM challenger benchmark and render Korean graphs.

The benchmark is deliberately bounded and deterministic.  It uses train and
validation people only, records the row caps in its manifest, and never reads
the locked-test files.  Feature-importance spread and seed variance are part
of the anchor decision rather than an after-the-fact visualization.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

_mpl_cache_root = (
    Path("/private/tmp" if Path("/private/tmp").is_dir() else "/tmp")
    / "multisensor-mpl-cache"
)
_mpl_cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache_root))

import matplotlib
import numpy as np
import pandas as pd

from multisensor_ml.models import select_training_rows
from multisensor_ml.tree_benchmark import (
    TREE_MODEL_IDS,
    TREE_MODEL_LABELS_KO,
    TreeBenchmarkConfig,
    fit_tree_challengers,
    reference_metrics_from_validation,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_keep_indices(frame: pd.DataFrame, limit: int, *, seed: int) -> np.ndarray:
    if limit <= 0 or len(frame) <= limit:
        return np.arange(len(frame), dtype="int64")
    positive = frame["event_binary"].astype(bool).to_numpy()
    hard_negative = frame["hard_negative"].astype(bool).to_numpy()
    must_keep = np.flatnonzero(positive | hard_negative)
    if len(must_keep) >= limit:
        return must_keep[:limit]
    candidates = np.flatnonzero(~(positive | hard_negative))
    remaining = limit - len(must_keep)
    rng = np.random.default_rng(seed)
    picked = np.sort(rng.choice(candidates, size=min(remaining, len(candidates)), replace=False))
    return np.sort(np.concatenate([must_keep, picked]))


def _load_split(
    project_root: Path,
    series_id: str,
    split_role: str,
    feature_names: list[str],
    *,
    row_cap: int,
    random_state: int,
) -> pd.DataFrame:
    root = project_root / "data" / "prepared" / series_id
    manifest = _read_json(root / "manifest.json")
    entries = [entry for entry in manifest["people"] if entry["split_role"] == split_role]
    if not entries:
        raise ValueError(f"no people found for split role {split_role}")
    columns = [
        "person_key",
        "context",
        "timestamp_utc",
        "event_binary",
        "hard_negative",
        *feature_names,
    ]
    frames: list[pd.DataFrame] = []
    for entry_index, entry in enumerate(entries):
        frame = pd.read_parquet(root / str(entry["path"]), columns=columns)
        if split_role == "train":
            selected = select_training_rows(frame, target="event_binary", baseline_ratio=3)
            frame = frame.loc[selected].reset_index(drop=True)
        if row_cap > 0:
            keep = _stable_keep_indices(
                frame,
                min(row_cap, len(frame)),
                seed=random_state + entry_index,
            )
            frame = frame.iloc[keep].reset_index(drop=True)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if row_cap > 0 and len(result) > row_cap:
        keep = _stable_keep_indices(result, row_cap, seed=random_state + 991)
        result = result.iloc[keep].reset_index(drop=True)
    return result


def _configure_korean_font() -> str:
    from matplotlib import font_manager

    candidates = [
        Path.home() / "Library/Fonts/NanumGothic-Regular.ttf",
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
    ]
    font_path = next((path for path in candidates if path.is_file()), None)
    if font_path is None:
        raise RuntimeError(
            "NanumGothic is not installed; run scripts/install_nanum_font.sh first"
        )
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams["font.family"] = "NanumGothic"
    plt.rcParams["axes.unicode_minus"] = False
    return "NanumGothic"


def _metric_plot(metrics: pd.DataFrame, output: Path) -> None:
    labels_map = {
        **TREE_MODEL_LABELS_KO,
        "existing_hist_gradient_boosting": "기존 HGB",
        "existing_logistic_regression": "기존 Logistic",
    }
    labels = [labels_map.get(str(value), str(value)) for value in metrics["model_id"]]
    colors = [
        "#7f8c8d"
        if str(value).startswith("existing_")
        else "#e15759"
        if value == "soft_ensemble"
        else "#20639b"
        for value in metrics["model_id"]
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    chart_specs = [
        ("aucpr", "AUCPR", "높을수록 좋음"),
        ("event_f1", "사건 F1", "높을수록 좋음"),
        ("false_alerts_per_hour", "시간당 오탐", "낮을수록 좋음"),
        ("calibration_error", "ECE", "낮을수록 좋음"),
    ]
    for axis, (metric, title, subtitle) in zip(axes.flat, chart_specs, strict=True):
        values = metrics[metric].astype(float).to_numpy()
        axis.bar(labels, values, color=colors)
        axis.set_title(f"{title} ({subtitle})")
        axis.tick_params(axis="x", rotation=22)
        axis.grid(axis="y", alpha=0.25)
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("XGBoost·LightGBM·앙상블 모델 비교", fontsize=16)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _feature_label_ko(name: str) -> str:
    labels = {
        "autonomic_arousal": "자율 각성",
        "motor_activation": "운동 활성",
        "cognitive_load": "인지 부하",
        "sleep_pressure": "수면 압력",
        "sensory_context": "감각 맥락",
        "recovery_capacity": "회복 능력",
        "social_context": "사회 맥락",
        "time_sin": "시간 주기 sin",
        "time_cos": "시간 주기 cos",
        "weekday_sin": "요일 주기 sin",
        "weekday_cos": "요일 주기 cos",
        "is_awake": "각성 상태",
    }
    parts = name.split("__")
    base = labels.get(parts[0], parts[0])
    if len(parts) == 1:
        return base
    return f"{base} · {parts[1]}"


def _importance_plot(summary: pd.DataFrame, importance: pd.DataFrame, output: Path) -> None:
    model_ids = [model_id for model_id in TREE_MODEL_IDS if model_id in set(importance["model_id"])]
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    for axis, model_id in zip(axes, model_ids, strict=False):
        model_table = importance.loc[importance["model_id"].eq(model_id)]
        means = (
            model_table.groupby("feature_name", sort=True)["normalized_importance"]
            .mean()
            .sort_values(ascending=False)
            .head(15)
            .sort_values()
        )
        axis.barh(
            [_feature_label_ko(str(name)) for name in means.index],
            means.to_numpy(),
            color="#59a14f",
        )
        axis.set_title(f"{TREE_MODEL_LABELS_KO[model_id]} 상위 피처 중요도")
        axis.set_xlabel("평균 정규화 중요도")
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle("피처 중요도 퍼짐과 안정성 확인", fontsize=16)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _stability_plot(metrics: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    merged = metrics.loc[metrics["model_id"].isin(TREE_MODEL_IDS)].merge(
        summary, on="model_id", how="inner", validate="one_to_one"
    )
    fig, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for _, row in merged.iterrows():
        axis.scatter(row["aucpr"], row["importance_variance_mean"], s=100)
        axis.annotate(
            TREE_MODEL_LABELS_KO[str(row["model_id"])],
            (row["aucpr"], row["importance_variance_mean"]),
        )
    axis.set_xlabel("AUCPR")
    axis.set_ylabel("피처 중요도 평균 분산")
    axis.set_title("성능과 피처 중요도 안정성")
    axis.grid(alpha=0.25)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _html_report(
    output: Path,
    *,
    artifact: dict[str, Any],
    metric_png: Path,
    importance_png: Path,
    stability_png: Path,
) -> None:
    table = pd.DataFrame(artifact["metrics"])
    rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{row.get(column, '')}</td>"
            for column in (
                "model_label_ko",
                "aucpr",
                "event_f1",
                "false_alerts_per_hour",
                "brier_score",
                "calibration_error",
                "evaluation_scope",
                "anchor_model",
            )
        )
        + "</tr>"
        for row in table.to_dict(orient="records")
    )
    html = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>트리 모델 비교</title>
<style>body{{font-family:NanumGothic,'Apple SD Gothic Neo',sans-serif;margin:2rem;color:#172033}}
img{{max-width:100%;height:auto;border:1px solid #e5e7eb;margin:1rem 0}}
table{{border-collapse:collapse;width:100%;font-size:.9rem}}
th,td{{border:1px solid #d9dee8;padding:.45rem;text-align:right}}
th:first-child,td:first-child{{text-align:left}}
.notice{{background:#fff5e6;padding:1rem;border-left:4px solid #e15759}}
</style></head><body>
<h1>XGBoost·LightGBM·앙상블 비교</h1>
<div class="notice">합성 <b>oracle/sanity</b> bounded validation 결과입니다.
실제 정확도는 <b>NOT VERIFIED</b>이며 locked test는 읽지 않았습니다.</div>
<p>도전자 앵커: <b>{artifact['challenger_anchor_model']}</b>
· 운영 기준 모델: <b>{artifact['operational_reference_model']}</b>
· 승격 상태: <b>{artifact['promotion_status_ko']}</b>
· 폰트: <b>NanumGothic</b> · 실행 시각: {artifact['generated_at']}</p>
<img src="{metric_png.name}" alt="모델 지표 비교 그래프">
<img src="{importance_png.name}" alt="피처 중요도 그래프">
<img src="{stability_png.name}" alt="성능과 중요도 안정성 그래프">
<table><thead><tr><th>모델</th><th>AUCPR</th><th>사건 F1</th>
<th>시간당 오탐</th><th>Brier</th><th>ECE</th><th>평가 범위</th><th>앵커</th></tr></thead>
<tbody>{rows}</tbody></table>
<p>ExtraTrees는 요청대로 이번 비교에서 제외하고 마지막 후보로 남겼습니다.
도전자와 기존 HGB는 평가 범위·임계값이 달라 자동 승격하지 않았습니다.</p>
</body></html>"""
    output.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--series", default="mvp3-oracle-v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-train-rows", type=int, default=600_000)
    parser.add_argument("--max-validation-rows", type=int, default=300_000)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=160)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    font_family = _configure_korean_font()
    root = args.project_root.resolve()
    manifest = _read_json(root / "data/prepared" / args.series / "manifest.json")
    feature_names = [str(value) for value in manifest["feature_names"]]
    train = _load_split(
        root,
        args.series,
        "train",
        feature_names,
        row_cap=args.max_train_rows,
        random_state=20260725,
    )
    validation = _load_split(
        root,
        args.series,
        "validation",
        feature_names,
        row_cap=args.max_validation_rows,
        random_state=20260802,
    )
    train_features = train[feature_names].astype("float32")
    validation_features = validation[feature_names].astype("float32")
    config = TreeBenchmarkConfig(
        seed_count=args.seed_count,
        n_estimators=args.n_estimators,
        threshold=args.threshold,
        n_jobs=1,
    )
    result = fit_tree_challengers(
        train_features,
        train["event_binary"].to_numpy(dtype="int8"),
        validation_features,
        validation["event_binary"].to_numpy(dtype="int8"),
        config=config,
        validation_timestamps=validation["timestamp_utc"].tolist(),
        validation_groups=validation["person_key"].tolist(),
    )
    challenger_metrics = result.metrics.copy()
    challenger_metrics["evaluation_scope"] = "bounded_validation_sample"
    reference_path = (
        root / "artifacts/registry" / args.series / "stage-model/validation_metrics.parquet"
    )
    if reference_path.is_file():
        reference_metrics = reference_metrics_from_validation(pd.read_parquet(reference_path))
        metrics = pd.concat([reference_metrics, challenger_metrics], ignore_index=True)
    else:
        metrics = challenger_metrics
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metric_png = output.with_name(output.stem + "_metrics.png")
    importance_png = output.with_name(output.stem + "_importance.png")
    stability_png = output.with_name(output.stem + "_stability.png")
    _metric_plot(metrics, metric_png)
    _importance_plot(result.importance_summary, result.feature_importance, importance_png)
    _stability_plot(result.metrics, result.importance_summary, stability_png)
    metrics_path = output.with_name(output.stem + ".metrics.parquet")
    importance_path = output.with_name(output.stem + ".importance.parquet")
    predictions_path = output.with_name(output.stem + ".predictions.parquet")
    metrics.to_parquet(metrics_path, index=False)
    result.feature_importance.to_parquet(importance_path, index=False)
    result.predictions.to_parquet(predictions_path, index=False)
    generated_at = pd.Timestamp.now(tz="Asia/Seoul").isoformat(timespec="seconds")
    artifact: dict[str, Any] = {
        "schema_version": "goal1.5/tree-model-benchmark/v1",
        "generated_at": generated_at,
        "language": "ko",
        "font_family": font_family,
        "data_status": "oracle/sanity",
        "real_data_status": "NOT VERIFIED",
        "locked_test_read": False,
        "extra_trees_included": False,
        "anchor_model": result.anchor_model,
        "challenger_anchor_model": result.anchor_model,
        "operational_reference_model": "existing_hist_gradient_boosting",
        "promotion_status": "NOT_PROMOTED_THRESHOLD_SCOPE_MISMATCH",
        "promotion_status_ko": "승격 보류(평가 범위·임계값 불일치)",
        "config": {
            "series": args.series,
            "train_row_cap": args.max_train_rows,
            "validation_row_cap": args.max_validation_rows,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "seed_count": args.seed_count,
            "n_estimators": args.n_estimators,
            "threshold": args.threshold,
            "n_jobs": 1,
        },
        "metrics": metrics.to_dict(orient="records"),
        "importance_summary": result.importance_summary.to_dict(orient="records"),
        "artifacts": {
            "metrics": metrics_path.name,
            "importance": importance_path.name,
            "predictions": predictions_path.name,
            "metric_graph": metric_png.name,
            "importance_graph": importance_png.name,
            "stability_graph": stability_png.name,
        },
    }
    artifact_path = output.with_suffix(".artifact.json")
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, default=str) + "\n")
    _html_report(
        output,
        artifact=artifact,
        metric_png=metric_png,
        importance_png=importance_png,
        stability_png=stability_png,
    )
    print(
        json.dumps(
            {
                "status": "READY",
                "anchor_model": result.anchor_model,
                "metrics": str(metrics_path),
                "report": str(output),
                "artifact": str(artifact_path),
                "font_family": font_family,
                "data_status": "oracle/sanity",
                "real_data_status": "NOT VERIFIED",
                "locked_test_read": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
