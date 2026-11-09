"""Build Korean AUCPR/AUROC figures for the current synthetic Watch bundle.

The observational 30-minute standard is a continuous regressor, so AUCPR and
AUROC are deliberately not assigned to it.  This script plots its valid
forecast errors beside the AUCPR/AUROC of the downstream virtual-pattern
classification heads, using validation people only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data/model_ready/standard-pattern-v2-quick"
BUNDLE_ROOT = PROJECT_ROOT / "services/onnx_api/models/goal15-monitor-v2"
REPORT_ROOT = PROJECT_ROOT / "reports"
SUMMARY_PNG = REPORT_ROOT / "current_standard_model_aucpr_auroc_ko.png"
CURVES_PNG = REPORT_ROOT / "current_standard_model_pr_roc_curves_ko.png"
METRICS_JSON = REPORT_ROOT / "current_standard_model_aucpr_auroc_ko.json"

VARIANTS = ("galaxy_watch", "galaxy_watch_h10")
VARIANT_LABELS = {"galaxy_watch": "Watch 단독", "galaxy_watch_h10": "Watch + H10"}
PATTERN_LABELS = {
    "meltdown_like": "멜트다운",
    "tantrum_like": "턴트럼",
    "sensory_seeking_like": "감각추구",
    "stereotypy_like": "상동행동",
    "shutdown_like": "셧다운",
}
PATTERN_COLORS = {
    "meltdown_like": "#20639B",
    "tantrum_like": "#F28E2B",
    "sensory_seeking_like": "#7A5195",
    "stereotypy_like": "#2A9D8F",
    "shutdown_like": "#B56576",
}
VARIANT_COLORS = {"galaxy_watch": "#20639B", "galaxy_watch_h10": "#F28E2B"}
FONT_CANDIDATES = (
    Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
    Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configure_font() -> str:
    for path in FONT_CANDIDATES:
        if not path.is_file():
            continue
        try:
            font_manager.fontManager.addfont(str(path))
            name = FontProperties(fname=str(path)).get_name()
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
        except (OSError, ValueError):
            continue
    matplotlib.rcParams["axes.unicode_minus"] = False
    return str(matplotlib.rcParams.get("font.family", ["sans-serif"])[0])


def _session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def _single_output(session: ort.InferenceSession, matrix: np.ndarray) -> np.ndarray:
    outputs = session.run(None, {session.get_inputs()[0].name: matrix.astype("float32")})
    if not outputs:
        raise RuntimeError("ONNX model returned no output")
    values = np.asarray(outputs[0], dtype="float64")
    return values.reshape(-1)


def _positive_probability(session: ort.InferenceSession, matrix: np.ndarray) -> np.ndarray:
    outputs = session.run(None, {session.get_inputs()[0].name: matrix.astype("float32")})
    for values in outputs:
        array = np.asarray(values)
        if array.ndim == 2 and array.shape[1] >= 2:
            return array[:, 1].astype("float64")
    last = np.asarray(outputs[-1], dtype="float64").reshape(-1)
    return last


def _load_validation() -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = json.loads((DATA_ROOT / "manifest.json").read_text(encoding="utf-8"))
    entries = [entry for entry in manifest["people"] if entry["split_role"] == "validation"]
    if len(entries) != 6:
        raise ValueError(f"expected six validation people, got {len(entries)}")
    frames = [pd.read_parquet(DATA_ROOT / str(entry["path"])) for entry in entries]
    validation = pd.concat(frames, ignore_index=True)
    if validation["split_role"].astype(str).ne("validation").any():
        raise ValueError("validation fixture contains a non-validation row")
    return validation, manifest


def _evaluate_variant(validation: pd.DataFrame, variant: str) -> dict[str, Any]:
    bundle = BUNDLE_ROOT / variant
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    base_names = list(manifest["feature_names"])
    pattern_names = list(manifest["pattern_feature_names"])
    standard = _session(bundle / str(manifest["standard_model_path"]))
    base_matrix = validation[base_names].to_numpy(dtype="float32")
    standard_prediction = _single_output(standard, base_matrix)

    valid = validation["standard_valid"].astype(bool).to_numpy()
    target = validation["standard_target"].to_numpy(dtype="float64")
    valid_target = valid & np.isfinite(target)
    standard_mae = float(np.mean(np.abs(standard_prediction[valid_target] - target[valid_target])))
    standard_rmse = float(
        np.sqrt(np.mean((standard_prediction[valid_target] - target[valid_target]) ** 2))
    )
    persistence = validation["watch_load_mean_300"].to_numpy(dtype="float64")
    persistence_valid = valid_target & np.isfinite(persistence)
    persistence_mae = float(
        np.mean(np.abs(persistence[persistence_valid] - target[persistence_valid]))
    )

    standard_forecast = np.full(len(validation), np.nan, dtype="float64")
    standard_forecast[valid] = standard_prediction[valid]
    standard_forecast = (
        pd.Series(standard_forecast)
        .fillna(validation["watch_load_mean_300"])
        .to_numpy(dtype="float64")
    )
    pattern_input = validation[base_names].copy()
    pattern_input["standard_forecast"] = standard_forecast
    pattern_input["standard_residual"] = (
        validation["watch_load"].to_numpy(dtype="float64") - standard_forecast
    )

    pattern_metrics: dict[str, Any] = {}
    curves: dict[str, Any] = {}
    for code in manifest["pattern_codes"]:
        model = _session(bundle / str(manifest["pattern_model_paths"][code]))
        probability = _positive_probability(
            model, pattern_input[pattern_names].to_numpy(dtype="float32")
        )
        truth = validation[f"label_{code}"].to_numpy(dtype="int8")
        if len(np.unique(truth)) < 2:
            raise ValueError(f"validation label has one class: {variant}/{code}")
        aucpr = float(average_precision_score(truth, probability))
        auroc = float(roc_auc_score(truth, probability))
        recorded_aucpr = float(manifest["pattern_metrics"][code]["aucpr"])
        if abs(aucpr - recorded_aucpr) > 5e-4:
            raise ValueError(
                f"AUCPR reproduction mismatch for {variant}/{code}: {aucpr} vs {recorded_aucpr}"
            )
        precision, recall, _ = precision_recall_curve(truth, probability)
        fpr, tpr, _ = roc_curve(truth, probability)
        pattern_metrics[code] = {
            "aucpr": aucpr,
            "aucpr_manifest": recorded_aucpr,
            "auroc": auroc,
            "positive_rows": int(truth.sum()),
            "rows": len(truth),
        }
        curves[code] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        }

    return {
        "variant": variant,
        "standard_regression": {
            "mae": standard_mae,
            "rmse": standard_rmse,
            "persistence_mae": persistence_mae,
            "rows": int(valid_target.sum()),
            "manifest": manifest["standard_metrics"],
        },
        "pattern_metrics": pattern_metrics,
        "curves": curves,
        "manifest": manifest,
    }


def _bar_label(axis: Any, bars: Any, fmt: str = "{:.3f}") -> None:
    for bar in bars:
        value = float(bar.get_height())
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _build_summary_plot(results: dict[str, dict[str, Any]], font_name: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    fig.patch.set_facecolor("#FFFFFF")
    fig.suptitle(
        "현재 표준선 번들의 성능 지표",
        fontsize=18,
        fontweight="bold",
        color="#172033",
    )
    subtitle = (
        "validation 6명 · 합성 oracle/sanity · locked_test_read=false · "
        "실제 성능 NOT VERIFIED"
    )
    fig.text(0.5, 0.965, subtitle, ha="center", va="top", fontsize=10, color="#5B6475")

    labels = [VARIANT_LABELS[v] for v in VARIANTS]
    x = np.arange(len(labels))
    width = 0.34

    ax = axes[0, 0]
    mae = [results[v]["standard_regression"]["mae"] for v in VARIANTS]
    rmse = [results[v]["standard_regression"]["rmse"] for v in VARIANTS]
    b1 = ax.bar(x - width / 2, mae, width, label="MAE", color="#20639B")
    b2 = ax.bar(x + width / 2, rmse, width, label="RMSE", color="#F28E2B")
    _bar_label(ax, b1)
    _bar_label(ax, b2)
    ax.set_title("표준선 30분 예측 오차", loc="left", color="#172033", fontweight="bold")
    ax.set_ylabel("오차 (낮을수록 좋음)")
    ax.set_xticks(x, labels)
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", color="#E5E7EB")
    ax.set_axisbelow(True)
    ax.text(
        0.0,
        -0.27,
        "표준선은 연속 회귀 타깃이므로 AUCPR/AUROC 대신 MAE·RMSE를 사용합니다.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#5B6475",
    )

    pattern_codes = list(results[VARIANTS[0]]["pattern_metrics"])
    pattern_labels = [PATTERN_LABELS[c] for c in pattern_codes]
    x_pattern = np.arange(len(pattern_codes))

    for axis, metric, title in (
        (axes[0, 1], "aucpr", "가상 패턴 AUCPR"),
        (axes[1, 0], "auroc", "가상 패턴 AUROC"),
    ):
        for offset, variant in zip((-width / 2, width / 2), VARIANTS, strict=True):
            values = [results[variant]["pattern_metrics"][code][metric] for code in pattern_codes]
            bars = axis.bar(
                x_pattern + offset,
                values,
                width,
                label=VARIANT_LABELS[variant],
                color=VARIANT_COLORS[variant],
                alpha=0.92,
            )
            _bar_label(axis, bars)
        axis.set_title(title, loc="left", color="#172033", fontweight="bold")
        axis.set_ylabel("점수 (높을수록 좋음)")
        axis.set_xticks(x_pattern, pattern_labels, rotation=18, ha="right")
        axis.set_ylim(0, 1)
        axis.grid(axis="y", color="#E5E7EB")
        axis.set_axisbelow(True)
        axis.legend(frameon=False, ncol=2)

    ax = axes[1, 1]
    ax.axis("off")
    watch = results["galaxy_watch"]["standard_regression"]
    dual = results["galaxy_watch_h10"]["standard_regression"]
    note = (
        "지표 해석\n\n"
        "• M0 표준선: 다음 30분 정상 부하 평균을 예측하는 회귀 모델\n"
        f"• Watch MAE {watch['mae']:.3f} / RMSE {watch['rmse']:.3f}\n"
        f"• Watch + H10 MAE {dual['mae']:.3f} / RMSE {dual['rmse']:.3f}\n\n"
        "• AUCPR·AUROC: M2 가상 패턴 head에만 계산\n"
        "• 패턴 라벨: synthetic_prior_v1\n"
        "• 임상·행동 정확도 또는 실제 Watch 성능을 의미하지 않음"
    )
    ax.text(
        0.02,
        0.93,
        note,
        va="top",
        fontsize=11,
        color="#172033",
        bbox={"boxstyle": "round,pad=0.8", "facecolor": "#F7F8FA", "edgecolor": "#D9DEE7"},
    )
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(colors="#5B6475")
    fig.savefig(SUMMARY_PNG, dpi=180, facecolor="white")
    plt.close(fig)


def _build_curve_plot(results: dict[str, dict[str, Any]], font_name: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    fig.patch.set_facecolor("#FFFFFF")
    fig.suptitle("가상 패턴 AUCPR·AUROC 곡선", fontsize=18, fontweight="bold", color="#172033")
    fig.text(
        0.5,
        0.965,
        "validation 6명 · synthetic oracle/sanity · 패턴별 곡선은 순위 성능 확인용",
        ha="center",
        va="top",
        fontsize=10,
        color="#5B6475",
    )
    for row, variant in enumerate(VARIANTS):
        pr_axis, roc_axis = axes[row]
        for code in results[variant]["pattern_metrics"]:
            metrics = results[variant]["pattern_metrics"][code]
            curve = results[variant]["curves"][code]
            color = PATTERN_COLORS[code]
            label = (
                f"{PATTERN_LABELS[code]} (AP {metrics['aucpr']:.3f} / "
                f"ROC {metrics['auroc']:.3f})"
            )
            pr_axis.plot(
                curve["recall"],
                curve["precision"],
                color=color,
                linewidth=1.8,
                label=label,
            )
            roc_axis.plot(
                curve["fpr"],
                curve["tpr"],
                color=color,
                linewidth=1.8,
                label=label,
            )
        pr_axis.set_title(f"{VARIANT_LABELS[variant]} · PR curve", loc="left", fontweight="bold")
        roc_axis.set_title(f"{VARIANT_LABELS[variant]} · ROC curve", loc="left", fontweight="bold")
        pr_axis.set_xlabel("재현율 (Recall)")
        pr_axis.set_ylabel("정밀도 (Precision)")
        roc_axis.set_xlabel("거짓 양성률 (FPR)")
        roc_axis.set_ylabel("참 양성률 (TPR)")
        pr_axis.set_xlim(0, 1)
        pr_axis.set_ylim(0, 1)
        roc_axis.set_xlim(0, 1)
        roc_axis.set_ylim(0, 1)
        roc_axis.plot([0, 1], [0, 1], linestyle="--", color="#8A9099", linewidth=1)
        for axis in (pr_axis, roc_axis):
            axis.grid(color="#E5E7EB")
            axis.set_axisbelow(True)
            axis.legend(loc="best", fontsize=8, frameon=False)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(colors="#5B6475")
    fig.savefig(CURVES_PNG, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    font_name = _configure_font()
    validation, dataset_manifest = _load_validation()
    results = {variant: _evaluate_variant(validation, variant) for variant in VARIANTS}
    _build_summary_plot(results, font_name)
    _build_curve_plot(results, font_name)
    evidence = {
        "status": "oracle/sanity",
        "real_data_status": "NOT VERIFIED",
        "locked_test_read": False,
        "font": font_name,
        "standard_contract": {
            "model_role": "continuous_30m_forecast",
            "target": "next_30m_normal_watch_load_mean",
            "aucpr_auroc": "NOT_APPLICABLE_TO_REGRESSION",
        },
        "evaluation": {
            "split_role": "validation",
            "validation_people": 6,
            "validation_rows": len(validation),
            "label_source": dataset_manifest["label_source"],
        },
        "variants": {
            variant: {
                "standard_regression": result["standard_regression"],
                "pattern_metrics": result["pattern_metrics"],
                "model_version": result["manifest"]["model_version"],
                "bundle_sha256": {
                    name: _sha256(BUNDLE_ROOT / variant / name)
                    for name in [
                        "manifest.json",
                        "standard_30m.onnx",
                        *result["manifest"]["pattern_model_paths"].values(),
                    ]
                },
            }
            for variant, result in results.items()
        },
        "source_sha256": {
            "dataset_manifest.json": _sha256(DATA_ROOT / "manifest.json"),
            "result.json": _sha256(
                PROJECT_ROOT / "artifacts/monitoring-v2/standard-pattern-v2-quick/result.json"
            ),
        },
        "artifacts": {
            "summary_png": str(SUMMARY_PNG),
            "curves_png": str(CURVES_PNG),
        },
    }
    METRICS_JSON.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "summary": str(SUMMARY_PNG),
                "curves": str(CURVES_PNG),
                "metrics": str(METRICS_JSON),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
