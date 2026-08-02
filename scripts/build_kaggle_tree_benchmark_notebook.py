# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NOTEBOOK_SOURCE = dedent(
    r'''
    from __future__ import annotations

    import json
    import sys
    from pathlib import Path

    import matplotlib
    import numpy as np
    import pandas as pd

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Kaggle 실행 전 이 플래그만 명시적으로 확인합니다.
    RUN_TRAINING = True
    RUN_LOCKED_TEST = False
    USE_GPU = False
    N_JOBS = 1
    MAX_TRAIN_ROWS = 600_000
    MAX_VALIDATION_ROWS = 300_000
    SEED_COUNT = 3
    N_ESTIMATORS = 160
    THRESHOLD = 0.5
    DATASET_ROOT_OVERRIDE = None
    OUTPUT_ROOT = Path("/kaggle/working/tree_model_benchmark")
    assert not RUN_LOCKED_TEST and not USE_GPU and N_JOBS == 1


    def _read_json(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))


    def _discover_dataset() -> Path:
        if DATASET_ROOT_OVERRIDE:
            root = Path(DATASET_ROOT_OVERRIDE)
            if (root / "manifest.json").is_file():
                return root
            raise FileNotFoundError(f"prepared manifest missing: {root}")
        candidates = []
        for manifest_path in sorted(Path("/kaggle/input").rglob("manifest.json")):
            try:
                manifest = _read_json(manifest_path)
            except (OSError, json.JSONDecodeError):
                continue
            if manifest.get("prepared_schema") == "goal1.5/prepared/v1":
                candidates.append(manifest_path.parent)
        if len(candidates) != 1:
            raise RuntimeError(
                "prepared Dataset manifest must be unique; "
                f"found {len(candidates)}: {candidates}"
            )
        return candidates[0]


    def _install_wheel_if_present() -> str:
        wheel_paths = sorted(Path("/kaggle/input").rglob("multisensor_ml-*.whl"))
        if wheel_paths:
            import subprocess

            runtime = Path("/kaggle/working/multisensor_ml_runtime")
            runtime.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(runtime), str(wheel_paths[-1])],
                check=True,
            )
            sys.path.insert(0, str(runtime))
            return str(wheel_paths[-1])
        return "preinstalled"


    def _load_split(root: Path, role: str, feature_names: list[str], cap: int, seed: int) -> pd.DataFrame:
        entries = [item for item in _read_json(root / "manifest.json")["people"] if item["split_role"] == role]
        if not entries:
            raise ValueError(f"no people for split role={role}")
        columns = ["person_key", "timestamp_utc", "event_binary", "hard_negative", *feature_names]
        frames = []
        for offset, entry in enumerate(entries):
            frame = pd.read_parquet(root / entry["path"], columns=columns)
            if role == "train":
                frame = frame.loc[select_training_rows(frame, target="event_binary", baseline_ratio=3)]
            frame = frame.reset_index(drop=True)
            if cap > 0 and len(frame) > cap:
                positive = frame["event_binary"].astype(bool).to_numpy()
                hard_negative = frame["hard_negative"].astype(bool).to_numpy()
                must_keep = np.flatnonzero(positive | hard_negative)
                if len(must_keep) < cap:
                    candidates = np.flatnonzero(~(positive | hard_negative))
                    rng = np.random.default_rng(seed + offset)
                    extra = rng.choice(candidates, size=min(cap - len(must_keep), len(candidates)), replace=False)
                    keep = np.sort(np.concatenate([must_keep, extra]))
                else:
                    keep = must_keep[:cap]
                frame = frame.iloc[keep].reset_index(drop=True)
            frames.append(frame)
        result = pd.concat(frames, ignore_index=True)
        if cap > 0 and len(result) > cap:
            result = result.iloc[:cap].reset_index(drop=True)
        return result


    def _configure_nanum() -> str:
        from matplotlib import font_manager

        candidates = [Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")]
        candidates += sorted(Path("/kaggle/input").rglob("NanumGothic-Regular.ttf"))
        candidates += sorted(Path("/kaggle/input").rglob("NanumGothic.ttf"))
        font_path = next((path for path in candidates if path.is_file()), None)
        if font_path is None:
            raise RuntimeError(
                "NanumGothic font is required. Attach a NanumGothic font Dataset "
                "before running this notebook."
            )
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = "NanumGothic"
        plt.rcParams["axes.unicode_minus"] = False
        return str(font_path)


    def _feature_label(name: str) -> str:
        labels = {
            "autonomic_arousal": "자율 각성", "motor_activation": "운동 활성",
            "cognitive_load": "인지 부하", "sleep_pressure": "수면 압력",
            "sensory_context": "감각 맥락", "recovery_capacity": "회복 능력",
            "social_context": "사회 맥락", "time_sin": "시간 주기 sin",
            "time_cos": "시간 주기 cos", "weekday_sin": "요일 주기 sin",
            "weekday_cos": "요일 주기 cos", "is_awake": "각성 상태",
        }
        parts = str(name).split("__")
        return labels.get(parts[0], parts[0]) if len(parts) == 1 else f"{labels.get(parts[0], parts[0])} · {parts[1]}"


    def _render(metrics: pd.DataFrame, importance: pd.DataFrame, summary: pd.DataFrame, out: Path, font_path: str) -> None:
        labels = {**TREE_MODEL_LABELS_KO, "existing_hist_gradient_boosting": "기존 HGB", "existing_logistic_regression": "기존 Logistic"}
        names = [labels.get(str(value), str(value)) for value in metrics["model_id"]]
        colors = ["#7f8c8d" if str(value).startswith("existing_") else "#e15759" if value == "soft_ensemble" else "#20639b" for value in metrics["model_id"]]
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
        for axis, metric, title, lower in zip(axes.flat, ("aucpr", "event_f1", "false_alerts_per_hour", "calibration_error"), ("AUCPR", "사건 F1", "시간당 오탐", "ECE"), (False, False, True, True), strict=True):
            values = metrics[metric].astype(float).to_numpy()
            axis.bar(names, values, color=colors)
            axis.set_title(f"{title} ({'낮을수록 좋음' if lower else '높을수록 좋음'})")
            axis.tick_params(axis="x", rotation=22)
            axis.grid(axis="y", alpha=0.25)
            for index, value in enumerate(values):
                axis.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
        fig.suptitle("XGBoost·LightGBM·앙상블 모델 비교", fontsize=16)
        fig.savefig(out.with_name(out.stem + "_metrics.png"), dpi=160, bbox_inches="tight")
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
        for axis, model_id in zip(axes, TREE_MODEL_IDS, strict=True):
            table = importance.loc[importance["model_id"].eq(model_id)]
            means = table.groupby("feature_name")["normalized_importance"].mean().nlargest(15).sort_values()
            axis.barh([_feature_label(name) for name in means.index], means.to_numpy(), color="#59a14f")
            axis.set_title(f"{TREE_MODEL_LABELS_KO[model_id]} 상위 피처 중요도")
            axis.set_xlabel("평균 정규화 중요도")
            axis.grid(axis="x", alpha=0.25)
        fig.suptitle("피처 중요도 퍼짐과 안정성 확인", fontsize=16)
        fig.savefig(out.with_name(out.stem + "_importance.png"), dpi=160, bbox_inches="tight")
        plt.close(fig)
        merged = metrics.loc[metrics["model_id"].isin(TREE_MODEL_IDS)].merge(summary, on="model_id", validate="one_to_one")
        fig, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
        for _, row in merged.iterrows():
            axis.scatter(row["aucpr"], row["importance_variance_mean"], s=100)
            axis.annotate(TREE_MODEL_LABELS_KO[str(row["model_id"])], (row["aucpr"], row["importance_variance_mean"]))
        axis.set_xlabel("AUCPR")
        axis.set_ylabel("피처 중요도 평균 분산")
        axis.set_title("성능과 피처 중요도 안정성")
        axis.grid(alpha=0.25)
        fig.savefig(out.with_name(out.stem + "_stability.png"), dpi=160, bbox_inches="tight")
        plt.close(fig)
        receipt = {
            "schema_version": "goal1.5/kaggle-tree-benchmark/v1",
            "font_family": "NanumGothic", "font_path": font_path,
            "data_status": "oracle/sanity", "real_data_status": "NOT VERIFIED",
            "locked_test_read": False, "extra_trees_included": False,
            "metrics": metrics.to_dict(orient="records"),
            "importance_summary": summary.to_dict(orient="records"),
        }
        out.with_suffix(".artifact.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")


    if RUN_TRAINING:
        dataset_root = _discover_dataset()
        wheel = _install_wheel_if_present()
        from multisensor_ml.models import select_training_rows
        from multisensor_ml.tree_benchmark import (
            TREE_MODEL_IDS,
            TREE_MODEL_LABELS_KO,
            TreeBenchmarkConfig,
            fit_tree_challengers,
            reference_metrics_from_validation,
        )
        manifest = _read_json(dataset_root / "manifest.json")
        feature_names = [str(item) for item in manifest["feature_names"]]
        train = _load_split(dataset_root, "train", feature_names, MAX_TRAIN_ROWS, 20260725)
        validation = _load_split(dataset_root, "validation", feature_names, MAX_VALIDATION_ROWS, 20260802)
        assert not any(item["split_role"] == "locked_test" for item in manifest["people"] if item["split_role"] in {"train", "validation"})
        result = fit_tree_challengers(
            train[feature_names].astype("float32"), train["event_binary"].to_numpy("int8"),
            validation[feature_names].astype("float32"), validation["event_binary"].to_numpy("int8"),
            config=TreeBenchmarkConfig(seed_count=SEED_COUNT, n_estimators=N_ESTIMATORS, threshold=THRESHOLD, n_jobs=N_JOBS),
            validation_timestamps=validation["timestamp_utc"].tolist(),
            validation_groups=validation["person_key"].tolist(),
        )
        metrics = result.metrics.copy()
        metrics["evaluation_scope"] = "bounded_validation_sample"
        reference_paths = sorted(Path("/kaggle/input").rglob("validation_metrics.parquet"))
        if reference_paths:
            metrics = pd.concat([reference_metrics_from_validation(pd.read_parquet(reference_paths[0])), metrics], ignore_index=True)
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        font_path = _configure_nanum()
        metrics.to_parquet(OUTPUT_ROOT / "metrics.parquet", index=False)
        result.feature_importance.to_parquet(OUTPUT_ROOT / "importance.parquet", index=False)
        result.predictions.to_parquet(OUTPUT_ROOT / "predictions.parquet", index=False)
        _render(metrics, result.feature_importance, result.importance_summary, OUTPUT_ROOT / "tree_model_benchmark", font_path)
        print(json.dumps({"status": "READY", "wheel": wheel, "anchor_model": result.anchor_model, "output": str(OUTPUT_ROOT)}, ensure_ascii=False))
    else:
        print("RUN_TRAINING=False; no fit executed")
    ''').strip()


def build_notebook() -> nbformat.NotebookNode:
    notebook = new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3.12"}
    notebook.cells = [
        new_markdown_cell(
            "# Goal 1.5 XGBoost·LightGBM·앙상블 비교\n\n"
            "합성 `oracle/sanity` 데이터의 train/validation만 사용합니다. "
            "ExtraTrees는 마지막 후보로 보류하고, locked test는 열지 않습니다. "
            "실제 성능은 `NOT VERIFIED`입니다."
        ),
        new_markdown_cell(
            "## 실행 계약\n\n"
            "`RUN_TRAINING=True`, `RUN_LOCKED_TEST=False`, `USE_GPU=False`, `N_JOBS=1`을 확인합니다. "
            "입력 Dataset에는 prepared manifest와 새 `multisensor_ml` wheel을 연결하고, "
            "NanumGothic 폰트 Dataset을 함께 연결해야 한글 그래프가 생성됩니다."
        ),
        new_code_cell(NOTEBOOK_SOURCE),
    ]
    return notebook


def main() -> None:
    destination = Path("kaggle/09_tree_model_benchmark.ipynb")
    notebook = build_notebook()
    nbformat.validate(notebook)
    nbformat.write(notebook, destination)


if __name__ == "__main__":
    main()
