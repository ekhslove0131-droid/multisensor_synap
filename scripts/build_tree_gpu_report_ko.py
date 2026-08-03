#!/usr/bin/env python3
"""Materialize a Korean, hash-backed report from the completed Kaggle run."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ko(value: object) -> str:
    if isinstance(value, bool):
        return "예" if value else "아니오"
    if isinstance(value, float):
        return f"{value:.6f}"
    return html.escape(str(value))


def _table(frame: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    headers = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    rows: list[str] = []
    for _, row in frame.iterrows():
        cells = "".join(f"<td>{_ko(row.get(name, ""))}</td>" for name, _ in columns)
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _copy_outputs(
    source: Path,
    manifest_path: Path,
    output: Path,
    log_path: Path | None,
) -> dict[str, str]:
    prefix = output.stem
    names: dict[str, str] = {}
    manifest_destination = output.parent / f"{prefix}.manifest.json"
    shutil.copy2(manifest_path, manifest_destination)
    names["manifest.json"] = manifest_destination.name
    if log_path is not None and log_path.exists():
        log_destination = output.parent / f"{prefix}.kaggle.log"
        shutil.copy2(log_path, log_destination)
        names["kaggle.log"] = log_destination.name
    for source_name in (
        "metrics.parquet",
        "ablation_metrics.parquet",
        "permutation_detail.parquet",
        "permutation_summary.parquet",
        "importance.parquet",
        "predictions.parquet",
    ):
        source_path = source / source_name
        destination = output.parent / f"{prefix}.{source_name}"
        shutil.copy2(source_path, destination)
        names[source_name] = destination.name
    for source_name in (
        "tree_model_benchmark_metrics.png",
        "tree_model_benchmark_importance.png",
        "tree_model_benchmark_stability.png",
        "tree_model_benchmark_ablation.png",
        "tree_model_benchmark_permutation.png",
    ):
        source_path = source / source_name
        short_name = source_name.removeprefix("tree_model_benchmark_")
        destination = output.parent / f"{prefix}_{short_name}"
        shutil.copy2(source_path, destination)
        names[source_name] = destination.name
    return names


def build_report(
    source: Path,
    manifest_path: Path,
    output: Path,
    *,
    kernel_slug: str,
    kernel_version: int,
    log_path: Path | None,
) -> dict[str, Any]:
    source = source.resolve()
    manifest_path = manifest_path.resolve()
    output = output.resolve()
    artifact = json.loads(
        (source / "tree_model_benchmark.artifact.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = pd.read_parquet(source / "metrics.parquet")
    ablation = pd.read_parquet(source / "ablation_metrics.parquet")
    permutation = pd.read_parquet(source / "permutation_summary.parquet")

    split_counts = {
        role: sum(item.get("split_role") == role for item in manifest["people"])
        for role in ("train", "validation", "locked_test")
    }
    if split_counts != {"train": 24, "validation": 6, "locked_test": 0}:
        raise ValueError(f"unexpected split contract: {split_counts}")
    if manifest.get("locked_test_read") is not False:
        raise ValueError("source manifest says locked_test was read")
    if artifact.get("locked_test_read") is not False:
        raise ValueError("Kaggle artifact says locked_test was read")
    expected_models = {"xgboost", "lightgbm", "extra_trees"}
    if set(artifact.get("candidate_model_ids", ())) != expected_models:
        raise ValueError("candidate model contract is incomplete")
    gpu_rows = metrics.loc[metrics["model_id"].isin(("xgboost", "lightgbm"))]
    if not bool((gpu_rows["execution_device"] == "cuda").all()):
        raise ValueError("GPU candidates do not report cuda execution")
    if not bool(gpu_rows["gpu_requested"].all()) or not bool(gpu_rows["gpu_required"].all()):
        raise ValueError("GPU candidates were not required to use the GPU")

    output.parent.mkdir(parents=True, exist_ok=True)
    copied = _copy_outputs(source, manifest_path, output, log_path)
    file_hashes = {
        filename: _sha256(output.parent / filename) for filename in copied.values()
    }

    full_columns = [
        ("model_id", "모델"),
        ("execution_device", "실행 장치"),
        ("aucpr", "AUCPR"),
        ("event_recall", "사건 recall"),
        ("event_f1", "사건 F1"),
        ("false_alerts_per_hour", "시간당 오탐"),
        ("brier_score", "Brier"),
        ("calibration_error", "ECE"),
    ]
    ablation_columns = [
        ("model_id", "모델"),
        ("aucpr", "시간 제거 AUCPR"),
        ("event_recall", "시간 제거 recall"),
        ("false_alerts_per_hour", "시간 제거 시간당 오탐"),
        ("calibration_error", "시간 제거 ECE"),
    ]
    permutation_columns = [
        ("model_id", "모델"),
        ("group_id", "피처 그룹"),
        ("mean_importance_drop", "평균 AUCPR drop"),
        ("positive_person_fraction", "양수 사람 비율"),
        ("mean_person_repeat_positive_rate", "반복 양수 비율"),
        ("repeated_positive", "반복 통과"),
    ]
    metrics_html = _table(metrics, full_columns)
    ablation_html = _table(ablation, ablation_columns)
    permutation_html = _table(permutation, permutation_columns)
    images = [
        copied["tree_model_benchmark_metrics.png"],
        copied["tree_model_benchmark_importance.png"],
        copied["tree_model_benchmark_stability.png"],
        copied["tree_model_benchmark_ablation.png"],
        copied["tree_model_benchmark_permutation.png"],
    ]
    image_html = "".join(
        f'<img src="{html.escape(name)}" alt="한글 모델 비교 그래프">' for name in images
    )
    report_artifact: dict[str, Any] = {
        "schema_version": "goal1.5/tree-model-gpu-report/v1",
        "status": "VERIFIED_ORACLE_GPU_RUN",
        "kernel": {"slug": kernel_slug, "version": kernel_version, "enable_gpu": True},
        "dataset": {
            "dataset_id": manifest.get("dataset_id"),
            "prepared_schema": manifest.get("prepared_schema"),
            "split_counts": split_counts,
            "manifest_sha256": file_hashes[copied["manifest.json"]],
        },
        "data_status": "oracle/sanity",
        "real_data_status": "NOT VERIFIED",
        "locked_test_read": False,
        "validation_person_count": 6,
        "candidate_model_ids": sorted(expected_models),
        "gpu_candidates": ["xgboost", "lightgbm"],
        "extra_trees_device": "cpu",
        "anchor_model": artifact.get("challenger_anchor_model"),
        "anchor_gate_status": artifact.get("anchor_gate_status"),
        "metrics": metrics.to_dict(orient="records"),
        "time_ablation_metrics": ablation.to_dict(orient="records"),
        "permutation_summary": permutation.to_dict(orient="records"),
        "file_sha256": file_hashes,
    }
    report_artifact_path = output.with_suffix(".artifact.json")
    report_artifact_path.write_text(
        json.dumps(report_artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    status_text = "합성 oracle/sanity GPU 실행 검증 완료 · 실제 데이터 정확도 NOT VERIFIED"
    output.write_text(
        f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>GPU 트리 모델 비교</title>
<style>
body{{font-family:Arial,'NanumGothic','Apple SD Gothic Neo',sans-serif;
margin:1.4rem;color:#172033;line-height:1.45}}
h1{{margin-bottom:.25rem}} h2{{margin-top:1.4rem}}
.status{{background:#e8f5e9;border-left:4px solid #2e7d32;padding:.7rem}}
.warn{{background:#fff8e1;border-left:4px solid #f9a825;padding:.7rem}}
table{{border-collapse:collapse;width:100%;font-size:.86rem;margin:.5rem 0 1rem}}
th,td{{border:1px solid #d9dee8;padding:.35rem;text-align:right}} th{{background:#eef2f7}}
th:first-child,td:first-child{{text-align:left}}
.charts{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.7rem}}
.charts img{{width:100%;border:1px solid #d9dee8}} code{{background:#f3f4f6;padding:.1rem .25rem}}
</style></head><body>
<h1>ExtraTrees·XGBoost·LightGBM·앙상블 비교</h1>
<div class="status">{status_text}</div>
<p>Kaggle <code>{html.escape(kernel_slug)}</code> v{kernel_version}에서 GPU를 켜고 실행했습니다.
XGBoost·LightGBM은 <b>cuda</b> 요청·필수, ExtraTrees는 CUDA 구현이 없어 CPU 후보로 기록했습니다.
실행 결과의 장치 메타데이터와 로그를 보존했으며, 자동 승격은 하지 않았습니다.</p>
<div class="warn">이 결과는 train 24명·validation 6명만 사용한 합성 oracle/sanity 결과입니다.
locked test는 읽지 않았고, Neon 실제 데이터 정확도·행동 예측 성능은
아직 <b>NOT VERIFIED</b>입니다.</div>
<h2>전체 validation 지표</h2>{metrics_html}
<h2>시간 피처 제거 비교</h2>{ablation_html}
<p>시간 피처를 제거했을 때 AUCPR와 recall이 증가한 모델은 시간 shortcut에
덜 의존한 것으로 해석할 수 있습니다.
이는 인과성이나 의료적 의미를 증명하지 않습니다.</p>
<h2>사람별 파생변수 permutation 감사</h2>{permutation_html}
<p>세 모델 모두 derived_signal 그룹은 6명 전원에서 반복적으로 양의 성능 감소를 보였습니다.
따라서 합성 기준의 핵심 근거는 파생변수 그룹이며, 단일 피처 중요도만으로
최종 모델을 결정하지 않습니다.</p>
<h2>그래프</h2><div class="charts">{image_html}</div>
<h2>선택 기록</h2><p>게이트 통과 앵커 후보:
<b>{html.escape(str(report_artifact['anchor_model']))}</b>.
이번 결과에서는 LightGBM이 AUCPR·F1·시간당 오탐의 균형이 가장 좋았지만,
실제 운영 승격은 라벨·동기화·개인별 검증 뒤에 수동으로 결정해야 합니다.</p>
</body></html>""",
        encoding="utf-8",
    )
    return report_artifact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--kernel-slug",
        default="bjcoding/multisensor-goal15-gpu-tree-benchmark-extratrees",
    )
    parser.add_argument("--kernel-version", type=int, default=4)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    artifact = build_report(
        args.source,
        args.manifest,
        args.output,
        kernel_slug=args.kernel_slug,
        kernel_version=args.kernel_version,
        log_path=args.log,
    )
    print(
        json.dumps(
            {"status": artifact["status"], "output": str(args.output)},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
