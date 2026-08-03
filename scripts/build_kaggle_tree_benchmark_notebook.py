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
    USE_GPU = True
    GPU_REQUIRED = True
    INCLUDE_EXTRA_TREES = True
    N_JOBS = 1
    MAX_TRAIN_ROWS = 600_000
    MAX_VALIDATION_ROWS = 300_000
    SEED_COUNT = 3
    N_ESTIMATORS = 160
    EXTRA_TREES_ESTIMATORS = 64
    THRESHOLD = 0.5
    DATASET_ROOT_OVERRIDE = None
    OUTPUT_ROOT = Path("/kaggle/working/tree_model_benchmark")
    assert not RUN_LOCKED_TEST and USE_GPU and GPU_REQUIRED and N_JOBS == 1


    def _read_json(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))


    def _resolve_data_path(root: Path, relative_path: str) -> Path:
        direct = root / relative_path
        if direct.is_file():
            return direct
        flattened_name = relative_path.replace("/", "__")
        for prefix in ("prepared__", ""):
            flattened = root / (prefix + flattened_name)
            if flattened.is_file():
                return flattened
            matches = sorted(root.rglob(flattened.name))
            if len(matches) == 1:
                return matches[0]
        raise FileNotFoundError(f"prepared file missing or ambiguous: {relative_path}")


    def _discover_dataset() -> tuple[Path, Path]:
        if DATASET_ROOT_OVERRIDE:
            root = Path(DATASET_ROOT_OVERRIDE)
            if (root / "manifest.json").is_file():
                return root, root / "manifest.json"
            prefixed = sorted(root.rglob("*manifest.json"))
            for manifest_path in prefixed:
                manifest = _read_json(manifest_path)
                if manifest.get("prepared_schema") == "goal1.5/prepared/v1":
                    return root, manifest_path
            raise FileNotFoundError(f"prepared manifest missing: {root}")
        candidates = []
        for manifest_path in sorted(Path("/kaggle/input").rglob("*manifest.json")):
            try:
                manifest = _read_json(manifest_path)
            except (OSError, json.JSONDecodeError):
                continue
            if manifest.get("prepared_schema") == "goal1.5/prepared/v1":
                candidates.append((manifest_path.parent, manifest_path))
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


    def _load_split(root: Path, manifest_path: Path, role: str, feature_names: list[str], cap: int, seed: int) -> pd.DataFrame:
        entries = [item for item in _read_json(manifest_path)["people"] if item["split_role"] == role]
        if not entries:
            raise ValueError(f"no people for split role={role}")
        per_person_caps = {}
        if cap > 0:
            base_cap, remainder = divmod(cap, len(entries))
            per_person_caps = {
                offset: max(1, base_cap + int(offset < remainder))
                for offset in range(len(entries))
            }
        # ``context`` is an auxiliary deterministic sampling key used by
        # select_training_rows; it is deliberately not a model feature.
        columns = list(dict.fromkeys(["person_key", "timestamp_utc", "context", "event_binary", "hard_negative", *feature_names]))
        frames = []
        for offset, entry in enumerate(entries):
            frame = pd.read_parquet(_resolve_data_path(root, entry["path"]), columns=columns)
            if role == "train":
                frame = frame.loc[select_training_rows(frame, target="event_binary", baseline_ratio=3)]
            frame = frame.reset_index(drop=True)
            person_cap = per_person_caps.get(offset, cap)
            if cap > 0 and len(frame) > person_cap:
                positive = frame["event_binary"].astype(bool).to_numpy()
                hard_negative = frame["hard_negative"].astype(bool).to_numpy()
                must_keep = np.flatnonzero(positive | hard_negative)
                if len(must_keep) < person_cap:
                    candidates = np.flatnonzero(~(positive | hard_negative))
                    rng = np.random.default_rng(seed + offset)
                    extra = rng.choice(candidates, size=min(person_cap - len(must_keep), len(candidates)), replace=False)
                    keep = np.sort(np.concatenate([must_keep, extra]))
                else:
                    keep = must_keep[:person_cap]
                frame = frame.iloc[keep].reset_index(drop=True)
            frames.append(frame)
        result = pd.concat(frames, ignore_index=True)
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


    def _render(metrics: pd.DataFrame, importance: pd.DataFrame, summary: pd.DataFrame, out: Path, font_path: str, ablation: pd.DataFrame, permutation: pd.DataFrame, gate: pd.DataFrame) -> None:
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
        model_ids = [model_id for model_id in TREE_MODEL_IDS if model_id in set(importance["model_id"])]
        ncols = min(3, max(1, len(model_ids)))
        nrows = int(np.ceil(len(model_ids) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 7 * nrows), constrained_layout=True, squeeze=False)
        axes_flat = axes.ravel()
        for axis, model_id in zip(axes_flat, model_ids, strict=False):
            table = importance.loc[importance["model_id"].eq(model_id)]
            means = table.groupby("feature_name")["normalized_importance"].mean().nlargest(15).sort_values()
            axis.barh([_feature_label(name) for name in means.index], means.to_numpy(), color="#59a14f")
            axis.set_title(f"{TREE_MODEL_LABELS_KO[model_id]} 상위 피처 중요도")
            axis.set_xlabel("평균 정규화 중요도")
            axis.grid(axis="x", alpha=0.25)
        for axis in axes_flat[len(model_ids):]:
            axis.axis("off")
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
        merged = metrics.loc[metrics["model_id"].isin(TREE_MODEL_IDS)].merge(
            ablation.loc[ablation["model_id"].isin(TREE_MODEL_IDS)],
            on="model_id", suffixes=("_full", "_without_time"), validate="one_to_one"
        )
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        labels = [TREE_MODEL_LABELS_KO[str(value)] for value in merged["model_id"]]
        drops = (
            (merged["aucpr_full"] - merged["aucpr_without_time"])
            / merged["aucpr_full"].abs().clip(lower=1e-12) * 100.0,
            (merged["event_recall_full"] - merged["event_recall_without_time"]) * 100.0,
        )
        for axis, values, title in zip(axes, drops, ("AUCPR 상대 저하율(%)", "recall 절대 저하(%p)"), strict=True):
            axis.bar(labels, values, color="#f28e2b")
            axis.axhline(0.0, color="#333333", linewidth=0.8)
            axis.set_title(title)
            axis.tick_params(axis="x", rotation=18)
            axis.grid(axis="y", alpha=0.25)
        fig.suptitle("시간 피처 제거 후 성능 유지 점검", fontsize=15)
        fig.savefig(out.with_name(out.stem + "_ablation.png"), dpi=160, bbox_inches="tight")
        plt.close(fig)
        permutation_plot = permutation.copy()
        permutation_plot["label"] = permutation_plot.apply(
            lambda row: f"{TREE_MODEL_LABELS_KO.get(str(row['model_id']), row['model_id'])} · {row['group_id']}",
            axis=1,
        )
        fig, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
        ordered = permutation_plot.sort_values("mean_importance_drop", kind="mergesort")
        axis.barh(ordered["label"], ordered["mean_importance_drop"], color="#59a14f")
        axis.axvline(0.0, color="#333333", linewidth=0.8)
        axis.set_xlabel("사람별 AUCPR permutation 감소량")
        axis.set_title("사람별 파생변수 그룹 중요도")
        axis.grid(axis="x", alpha=0.25)
        fig.savefig(out.with_name(out.stem + "_permutation.png"), dpi=160, bbox_inches="tight")
        plt.close(fig)
        receipt = {
            "schema_version": "goal1.5/kaggle-tree-benchmark/v2",
            "status": "READY",
            "font_family": "NanumGothic", "font_path": font_path,
            "data_status": "oracle/sanity", "real_data_status": "NOT VERIFIED",
            "locked_test_read": False,
            "metrics": metrics.to_dict(orient="records"),
            "importance_summary": summary.to_dict(orient="records"),
            "time_ablation_metrics": ablation.to_dict(orient="records"),
            "permutation_summary": permutation.to_dict(orient="records"),
            "anchor_gate": gate.to_dict(orient="records"),
            "challenger_anchor_model": None,
            "candidate_model_ids": list(TREE_MODEL_IDS),
            "use_gpu": USE_GPU,
            "gpu_required": GPU_REQUIRED,
            "extra_trees_included": INCLUDE_EXTRA_TREES,
        }
        out.with_suffix(".artifact.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")


    if RUN_TRAINING:
        dataset_root, manifest_path = _discover_dataset()
        wheel = _install_wheel_if_present()
        from multisensor_ml.models import select_training_rows
        from multisensor_ml.tree_benchmark import (
            evaluate_anchor_gate,
            feature_group_map,
            TREE_MODEL_IDS,
            TREE_MODEL_LABELS_KO,
            TreeBenchmarkConfig,
            fit_tree_challengers,
            permutation_importance_by_person,
            reference_metrics_from_validation,
            select_gated_anchor_model,
            summarize_permutation_importance,
        )
        manifest = _read_json(manifest_path)
        feature_names = [str(item) for item in manifest["feature_names"]]
        train = _load_split(dataset_root, manifest_path, "train", feature_names, MAX_TRAIN_ROWS, 20260725)
        validation = _load_split(dataset_root, manifest_path, "validation", feature_names, MAX_VALIDATION_ROWS, 20260802)
        assert not any(item["split_role"] == "locked_test" for item in manifest["people"] if item["split_role"] in {"train", "validation"})
        result = fit_tree_challengers(
            train[feature_names].astype("float32"), train["event_binary"].to_numpy("int8"),
            validation[feature_names].astype("float32"), validation["event_binary"].to_numpy("int8"),
            config=TreeBenchmarkConfig(seed_count=SEED_COUNT, n_estimators=N_ESTIMATORS, extra_trees_n_estimators=EXTRA_TREES_ESTIMATORS, include_extra_trees=INCLUDE_EXTRA_TREES, use_gpu=USE_GPU, gpu_required=GPU_REQUIRED, n_jobs=N_JOBS, threshold=THRESHOLD),
            validation_timestamps=validation["timestamp_utc"].tolist(),
            validation_groups=validation["person_key"].tolist(),
        )
        feature_groups = feature_group_map(feature_names)
        without_time_names = [name for name in feature_names if name not in set(feature_groups.get("time", ()))]
        time_result = fit_tree_challengers(
            train[without_time_names].astype("float32"), train["event_binary"].to_numpy("int8"),
            validation[without_time_names].astype("float32"), validation["event_binary"].to_numpy("int8"),
            config=TreeBenchmarkConfig(seed_count=SEED_COUNT, n_estimators=N_ESTIMATORS, extra_trees_n_estimators=EXTRA_TREES_ESTIMATORS, include_extra_trees=INCLUDE_EXTRA_TREES, use_gpu=USE_GPU, gpu_required=GPU_REQUIRED, n_jobs=N_JOBS, threshold=THRESHOLD),
            validation_timestamps=validation["timestamp_utc"].tolist(),
            validation_groups=validation["person_key"].tolist(),
        )
        permutation_detail = permutation_importance_by_person(
            result.fitted_models, validation[feature_names], validation["event_binary"].to_numpy("int8"),
            validation["person_key"].tolist(), feature_groups,
            repeats=3, random_state=20260725,
        )
        permutation_summary = summarize_permutation_importance(permutation_detail)
        metrics = result.metrics.copy()
        metrics["feature_set"] = "full"
        metrics["evaluation_scope"] = "bounded_validation_sample"
        ablation = time_result.metrics.copy()
        ablation["feature_set"] = "without_time"
        ablation["evaluation_scope"] = "bounded_validation_sample"
        gate = evaluate_anchor_gate(
            metrics.loc[metrics["model_id"].isin(TREE_MODEL_IDS)],
            ablation.loc[ablation["model_id"].isin(TREE_MODEL_IDS)],
            permutation_summary,
        )
        gated_anchor = select_gated_anchor_model(metrics.loc[metrics["model_id"].isin(TREE_MODEL_IDS)], result.importance_summary, gate)
        reference_paths = sorted(Path("/kaggle/input").rglob("validation_metrics.parquet"))
        if reference_paths:
            references = reference_metrics_from_validation(pd.read_parquet(reference_paths[0]))
            references["feature_set"] = "full"
            metrics = pd.concat([references, metrics], ignore_index=True)
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        font_path = _configure_nanum()
        metrics.to_parquet(OUTPUT_ROOT / "metrics.parquet", index=False)
        ablation.to_parquet(OUTPUT_ROOT / "ablation_metrics.parquet", index=False)
        permutation_detail.to_parquet(OUTPUT_ROOT / "permutation_detail.parquet", index=False)
        permutation_summary.to_parquet(OUTPUT_ROOT / "permutation_summary.parquet", index=False)
        result.feature_importance.to_parquet(OUTPUT_ROOT / "importance.parquet", index=False)
        result.predictions.to_parquet(OUTPUT_ROOT / "predictions.parquet", index=False)
        _render(metrics.loc[metrics["feature_set"].eq("full")], result.feature_importance, result.importance_summary, OUTPUT_ROOT / "tree_model_benchmark", font_path, ablation, permutation_summary, gate)
        receipt = json.loads((OUTPUT_ROOT / "tree_model_benchmark.artifact.json").read_text())
        receipt["challenger_anchor_model"] = gated_anchor
        receipt["anchor_gate_status"] = "PASS" if gated_anchor else "FAIL"
        receipt["anchor_gate_status_ko"] = "통과" if gated_anchor else "실패"
        receipt["candidate_model_ids"] = list(result.fitted_models or {})
        (OUTPUT_ROOT / "tree_model_benchmark.artifact.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"status": "READY", "wheel": wheel, "anchor_model": gated_anchor, "candidate_anchor_model": result.anchor_model, "anchor_gate_status": receipt["anchor_gate_status"], "output": str(OUTPUT_ROOT)}, ensure_ascii=False))
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
            "# Goal 1.5 GPU 트리 모델 비교\n\n"
            "합성 `oracle/sanity` 데이터의 train/validation만 사용합니다. "
            "XGBoost·LightGBM은 GPU를 강제하고 ExtraTrees는 CPU 기준 후보로 함께 기록합니다. "
            "locked test는 열지 않으며 실제 성능은 `NOT VERIFIED`입니다."
        ),
        new_markdown_cell(
            "## 실행 계약\n\n"
            "`RUN_TRAINING=True`, `RUN_LOCKED_TEST=False`, `USE_GPU=True`, `GPU_REQUIRED=True`, `N_JOBS=1`을 확인합니다. "
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
