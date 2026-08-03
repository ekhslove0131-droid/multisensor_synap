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
    FEATURE_GROUP_LABELS_KO,
    TREE_MODEL_IDS,
    TREE_MODEL_LABELS_KO,
    TreeBenchmarkConfig,
    evaluate_anchor_gate,
    feature_group_map,
    fit_tree_challengers,
    permutation_importance_by_person,
    reference_metrics_from_validation,
    select_gated_anchor_model,
    summarize_permutation_importance,
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
    rng = np.random.default_rng(seed)
    positive_indices = np.flatnonzero(positive)
    hard_indices = np.flatnonzero(hard_negative & ~positive)
    selected_parts: list[np.ndarray] = []

    def choose(indices: np.ndarray, count: int) -> np.ndarray:
        if count <= 0 or len(indices) == 0:
            return np.array([], dtype="int64")
        if count >= len(indices):
            return indices
        return np.sort(rng.choice(indices, size=count, replace=False))

    positive_quota = min(len(positive_indices), max(1, limit // 2))
    hard_quota = min(len(hard_indices), max(1, limit // 6))
    selected_parts.extend(
        [choose(positive_indices, positive_quota), choose(hard_indices, hard_quota)]
    )
    selected = np.unique(np.concatenate(selected_parts))
    if len(selected) >= limit:
        return np.sort(selected[:limit])
    candidates = np.setdiff1d(np.arange(len(frame), dtype="int64"), selected, assume_unique=True)
    remaining = limit - len(selected)
    picked = np.sort(rng.choice(candidates, size=min(remaining, len(candidates)), replace=False))
    return np.sort(np.concatenate([selected, picked]))


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
    per_person_caps: dict[int, int] = {}
    if row_cap > 0:
        base_cap, remainder = divmod(row_cap, len(entries))
        for entry_index in range(len(entries)):
            per_person_caps[entry_index] = max(
                1, base_cap + int(entry_index < remainder)
            )
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
                min(per_person_caps[entry_index], len(frame)),
                seed=random_state + entry_index,
            )
            frame = frame.iloc[keep].reset_index(drop=True)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
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
    ncols = min(3, max(1, len(model_ids)))
    nrows = int(np.ceil(len(model_ids) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(8 * ncols, 7 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()
    for axis, model_id in zip(axes_flat, model_ids, strict=False):
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
    for axis in axes_flat[len(model_ids) :]:
        axis.axis("off")
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


def _ablation_plot(
    full_metrics: pd.DataFrame,
    time_ablation_metrics: pd.DataFrame,
    output: Path,
) -> None:
    tree_full = full_metrics.loc[full_metrics["model_id"].isin(TREE_MODEL_IDS)]
    tree_ablation = time_ablation_metrics.loc[
        time_ablation_metrics["model_id"].isin(TREE_MODEL_IDS)
    ]
    merged = tree_full.merge(
        tree_ablation,
        on="model_id",
        suffixes=("_full", "_without_time"),
        validate="one_to_one",
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    labels = [TREE_MODEL_LABELS_KO[str(value)] for value in merged["model_id"]]
    aucpr_drop = (
        (merged["aucpr_full"] - merged["aucpr_without_time"])
        / merged["aucpr_full"].abs().clip(lower=1e-12)
        * 100.0
    )
    recall_drop = (
        merged["event_recall_full"] - merged["event_recall_without_time"]
    ) * 100.0
    for axis, values, title in zip(
        axes,
        (aucpr_drop, recall_drop),
        ("AUCPR 상대 저하율(%)", "사건 recall 절대 저하(%p)"),
        strict=True,
    ):
        axis.bar(labels, values, color="#f28e2b")
        axis.axhline(0.0, color="#333333", linewidth=0.8)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=18)
        for index, value in enumerate(values):
            axis.text(index, float(value), f"{float(value):.2f}", ha="center", va="bottom")
    fig.suptitle("시간 피처 제거 후 성능 유지 점검", fontsize=15)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _permutation_plot(summary: pd.DataFrame, output: Path) -> None:
    work = summary.copy()
    if work.empty:
        work = pd.DataFrame(
            [{"model_id": "없음", "group_id": "없음", "mean_importance_drop": 0.0}]
        )
    work["label"] = work.apply(
        lambda row: f"{TREE_MODEL_LABELS_KO.get(str(row['model_id']), row['model_id'])} · "
        f"{FEATURE_GROUP_LABELS_KO.get(str(row['group_id']), row['group_id'])}",
        axis=1,
    )
    ordered = work.sort_values("mean_importance_drop", kind="mergesort")
    fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    values = ordered["mean_importance_drop"].astype(float).to_numpy()
    colors = ["#59a14f" if value > 0.0 else "#e15759" for value in values]
    axis.barh(ordered["label"], values, color=colors)
    axis.axvline(0.0, color="#333333", linewidth=0.8)
    axis.set_xlabel("사람별 AUCPR permutation 감소량(평균)")
    axis.set_title("사람별 생리·상황 파생변수 그룹 중요도")
    axis.grid(axis="x", alpha=0.25)
    for index, (_, row) in enumerate(ordered.iterrows()):
        fraction = row.get("positive_person_fraction")
        if pd.notna(fraction):
            axis.text(
                float(row["mean_importance_drop"]),
                index,
                f"  양수 사람 비율 {float(fraction):.2f}",
                va="center",
                fontsize=8,
            )
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _html_report(
    output: Path,
    *,
    artifact: dict[str, Any],
    metric_png: Path,
    importance_png: Path,
    stability_png: Path,
    ablation_png: Path,
    permutation_png: Path,
) -> None:
    def rows_for(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
        return "".join(
            "<tr>"
            + "".join(f"<td>{row.get(column, '')}</td>" for column in columns)
            + "</tr>"
            for row in frame.to_dict(orient="records")
        )

    table = pd.DataFrame(artifact["display_metrics"])
    rows = rows_for(
        table,
        (
            "model_label_ko",
            "aucpr",
            "event_recall",
            "event_f1",
            "false_alerts_per_hour",
            "brier_score",
            "calibration_error",
            "evaluation_scope",
        ),
    )
    gate = pd.DataFrame(artifact["anchor_gate"])
    gate_rows = rows_for(
        gate,
        (
            "model_id",
            "aucpr_relative_drop",
            "event_recall_drop",
            "derived_mean_importance_drop",
            "derived_positive_person_fraction",
            "gate_pass",
            "gate_reason",
        ),
    )
    anchor = artifact["challenger_anchor_model"] or "없음(게이트 실패)"
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
<p>도전자 앵커: <b>{anchor}</b>
· 운영 기준 모델: <b>{artifact['operational_reference_model']}</b>
· 감사 게이트: <b>{artifact['anchor_gate_status_ko']}</b>
· GPU 요청: <b>{artifact['config']['use_gpu']}</b>
· 폰트: <b>NanumGothic</b> · 실행 시각: {artifact['generated_at']}</p>
<img src="{metric_png.name}" alt="모델 지표 비교 그래프">
<img src="{importance_png.name}" alt="피처 중요도 그래프">
<img src="{stability_png.name}" alt="성능과 중요도 안정성 그래프">
<img src="{ablation_png.name}" alt="시간 피처 제거 성능 그래프">
<img src="{permutation_png.name}" alt="사람별 파생변수 permutation 중요도 그래프">
<table><thead><tr><th>모델</th><th>AUCPR</th><th>사건 recall</th><th>사건 F1</th>
<th>시간당 오탐</th><th>Brier</th><th>ECE</th><th>평가 범위</th></tr></thead>
<tbody>{rows}</tbody></table>
<h2>앵커 채택 게이트</h2>
<p>시간 피처 제거 후 AUCPR 상대 저하 ≤
{artifact['config']['time_ablation_aucpr_relative_drop_limit']:.2%},
사건 recall 절대 저하 ≤
{artifact['config']['time_ablation_event_recall_absolute_drop_limit']:.2%},
파생변수 그룹의 사람별 반복 양수 비율 ≥ {artifact['config']['min_positive_person_fraction']:.0%}를
사전에 고정했습니다.</p>
<table><thead><tr><th>모델</th><th>AUCPR 저하</th><th>recall 저하</th>
<th>파생변수 평균 drop</th><th>양수 사람 비율</th><th>게이트</th><th>사유</th></tr></thead>
<tbody>{gate_rows}</tbody></table>
<p>ExtraTrees는 scikit-learn CUDA 구현이 없어 CPU 후보로 별도 기록했습니다. GPU 요청 시
XGBoost·LightGBM은 GPU 실행이 확인되지 않으면 실패하도록 구성했습니다. 도전자와 기존
HGB는 평가 범위·임계값이 달라 직접 운영 승격 비교하지 않았습니다.</p>
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
    parser.add_argument("--include-extra-trees", action="store_true")
    parser.add_argument("--extra-trees-estimators", type=int, default=64)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--gpu-required", action="store_true")
    parser.add_argument("--gpu-device", default="cuda")
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
        include_extra_trees=args.include_extra_trees,
        extra_trees_n_estimators=args.extra_trees_estimators,
        use_gpu=args.use_gpu,
        gpu_required=args.gpu_required,
        gpu_device=args.gpu_device,
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
    feature_groups = feature_group_map(feature_names)
    time_features = set(feature_groups.get("time", ()))
    if not time_features:
        raise ValueError("time feature group is required for the anchor audit")
    without_time_names = [name for name in feature_names if name not in time_features]
    time_ablation_result = fit_tree_challengers(
        train[without_time_names].astype("float32"),
        train["event_binary"].to_numpy(dtype="int8"),
        validation[without_time_names].astype("float32"),
        validation["event_binary"].to_numpy(dtype="int8"),
        config=config,
        validation_timestamps=validation["timestamp_utc"].tolist(),
        validation_groups=validation["person_key"].tolist(),
    )
    if result.fitted_models is None:
        raise RuntimeError("full challenger models are required for permutation audit")
    permutation_detail = permutation_importance_by_person(
        result.fitted_models,
        validation_features,
        validation["event_binary"].to_numpy(dtype="int8"),
        validation["person_key"].tolist(),
        feature_groups,
        repeats=config.permutation_repeats,
        random_state=config.random_state,
    )
    permutation_summary = summarize_permutation_importance(
        permutation_detail,
        min_positive_person_fraction=config.min_positive_person_fraction,
        min_repeat_positive_rate=config.min_repeat_positive_rate,
    )
    challenger_metrics = result.metrics.copy()
    challenger_metrics["feature_set"] = "full"
    challenger_metrics["evaluation_scope"] = "bounded_validation_sample"
    time_ablation_metrics = time_ablation_result.metrics.copy()
    time_ablation_metrics["feature_set"] = "without_time"
    time_ablation_metrics["evaluation_scope"] = "bounded_validation_sample"
    anchor_gate = evaluate_anchor_gate(
        challenger_metrics.loc[challenger_metrics["model_id"].isin(TREE_MODEL_IDS)],
        time_ablation_metrics.loc[time_ablation_metrics["model_id"].isin(TREE_MODEL_IDS)],
        permutation_summary,
        aucpr_relative_drop_limit=config.time_ablation_aucpr_relative_drop_limit,
        event_recall_absolute_drop_limit=config.time_ablation_event_recall_absolute_drop_limit,
        min_positive_person_fraction=config.min_positive_person_fraction,
        min_repeat_positive_rate=config.min_repeat_positive_rate,
    )
    gated_anchor = select_gated_anchor_model(
        challenger_metrics.loc[challenger_metrics["model_id"].isin(TREE_MODEL_IDS)],
        result.importance_summary,
        anchor_gate,
    )
    gate_passed = gated_anchor is not None
    reference_path = (
        root / "artifacts/registry" / args.series / "stage-model/validation_metrics.parquet"
    )
    if reference_path.is_file():
        reference_metrics = reference_metrics_from_validation(pd.read_parquet(reference_path))
        reference_metrics["feature_set"] = "full"
        metrics = pd.concat(
            [reference_metrics, challenger_metrics, time_ablation_metrics], ignore_index=True
        )
    else:
        metrics = pd.concat([challenger_metrics, time_ablation_metrics], ignore_index=True)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metric_png = output.with_name(output.stem + "_metrics.png")
    importance_png = output.with_name(output.stem + "_importance.png")
    stability_png = output.with_name(output.stem + "_stability.png")
    ablation_png = output.with_name(output.stem + "_ablation.png")
    permutation_png = output.with_name(output.stem + "_permutation.png")
    display_metrics = metrics.loc[metrics["feature_set"].eq("full")].copy()
    _metric_plot(display_metrics, metric_png)
    _importance_plot(result.importance_summary, result.feature_importance, importance_png)
    _stability_plot(display_metrics, result.importance_summary, stability_png)
    _ablation_plot(challenger_metrics, time_ablation_metrics, ablation_png)
    _permutation_plot(permutation_summary, permutation_png)
    metrics_path = output.with_name(output.stem + ".metrics.parquet")
    importance_path = output.with_name(output.stem + ".importance.parquet")
    predictions_path = output.with_name(output.stem + ".predictions.parquet")
    ablation_path = output.with_name(output.stem + ".ablation_metrics.parquet")
    permutation_detail_path = output.with_name(output.stem + ".permutation_detail.parquet")
    permutation_summary_path = output.with_name(
        output.stem + ".permutation_summary.parquet"
    )
    metrics.to_parquet(metrics_path, index=False)
    result.feature_importance.to_parquet(importance_path, index=False)
    result.predictions.to_parquet(predictions_path, index=False)
    time_ablation_metrics.to_parquet(ablation_path, index=False)
    permutation_detail.to_parquet(permutation_detail_path, index=False)
    permutation_summary.to_parquet(permutation_summary_path, index=False)
    generated_at = pd.Timestamp.now(tz="Asia/Seoul").isoformat(timespec="seconds")
    promotion_status = (
        "CHALLENGER_ANCHOR_GATE_PASSED"
        if gate_passed
        else "NOT_PROMOTED_AUDIT_GATE_FAILED"
    )
    promotion_status_ko = (
        "도전자 앵커 채택 조건 통과"
        if gate_passed
        else "승격 보류(시간 제거·파생변수 반복성 게이트 실패)"
    )
    artifact: dict[str, Any] = {
        "schema_version": "goal1.5/tree-model-benchmark/v2",
        "status": "READY",
        "generated_at": generated_at,
        "language": "ko",
        "font_family": font_family,
        "data_status": "oracle/sanity",
        "real_data_status": "NOT VERIFIED",
        "locked_test_read": False,
        "extra_trees_included": bool(config.include_extra_trees),
        "anchor_model": gated_anchor,
        "candidate_anchor_model": result.anchor_model,
        "challenger_anchor_model": gated_anchor,
        "operational_reference_model": "existing_hist_gradient_boosting",
        "promotion_status": promotion_status,
        "promotion_status_ko": promotion_status_ko,
        "anchor_gate_status": "PASS" if gate_passed else "FAIL",
        "anchor_gate_status_ko": "통과" if gate_passed else "실패",
        "anchor_gate": anchor_gate.to_dict(orient="records"),
        "permutation_summary": permutation_summary.to_dict(orient="records"),
        "time_ablation_metrics": time_ablation_metrics.to_dict(orient="records"),
        "feature_groups": {key: list(value) for key, value in feature_groups.items()},
        "candidate_model_ids": list(result.fitted_models or {}),
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
            "include_extra_trees": config.include_extra_trees,
            "extra_trees_n_estimators": config.extra_trees_n_estimators,
            "use_gpu": config.use_gpu,
            "gpu_required": config.gpu_required,
            "gpu_device": config.gpu_device,
            "time_ablation_aucpr_relative_drop_limit": (
                config.time_ablation_aucpr_relative_drop_limit
            ),
            "time_ablation_event_recall_absolute_drop_limit": (
                config.time_ablation_event_recall_absolute_drop_limit
            ),
            "min_positive_person_fraction": config.min_positive_person_fraction,
            "min_repeat_positive_rate": config.min_repeat_positive_rate,
            "permutation_repeats": config.permutation_repeats,
        },
        "metrics": metrics.to_dict(orient="records"),
        "display_metrics": display_metrics.to_dict(orient="records"),
        "importance_summary": result.importance_summary.to_dict(orient="records"),
        "artifacts": {
            "metrics": metrics_path.name,
            "importance": importance_path.name,
            "predictions": predictions_path.name,
            "ablation_metrics": ablation_path.name,
            "permutation_detail": permutation_detail_path.name,
            "permutation_summary": permutation_summary_path.name,
            "metric_graph": metric_png.name,
            "importance_graph": importance_png.name,
            "stability_graph": stability_png.name,
            "ablation_graph": ablation_png.name,
            "permutation_graph": permutation_png.name,
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
        ablation_png=ablation_png,
        permutation_png=permutation_png,
    )
    print(
        json.dumps(
            {
                "status": "READY",
                "anchor_model": gated_anchor,
                "candidate_anchor_model": result.anchor_model,
                "anchor_gate_status": "PASS" if gate_passed else "FAIL",
                "candidate_model_ids": list(result.fitted_models or {}),
                "use_gpu": config.use_gpu,
                "gpu_required": config.gpu_required,
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
