from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from multisensor_ml.model_registry import ModelRegistry
from multisensor_ml.personalization import fit_standard_type_artifacts
from multisensor_ml.pipeline import prepare_series
from multisensor_ml.receipts import (
    StageReceipt,
    create_stage_receipt,
    validate_stage_receipt,
)
from multisensor_ml.registry import sha256_file
from multisensor_ml.settings import TrainingRegistryConfig
from multisensor_ml.training_pipeline import (
    create_korean_router_artifacts,
    create_prediction_artifacts,
    evaluate_hierarchical_artifacts,
    fit_behavior_model_artifacts,
    fit_stage_model_artifacts,
)

RegistryStage = Literal[
    "labels",
    "baseline",
    "types",
    "stage-model",
    "behavior-model",
    "evaluate",
    "predict",
    "route-ko",
]
STAGES: tuple[RegistryStage, ...] = (
    "labels",
    "baseline",
    "types",
    "stage-model",
    "behavior-model",
    "evaluate",
    "predict",
    "route-ko",
)
_EXPECTED_INPUT_STAGE: dict[RegistryStage, str] = {
    "labels": "labels",
    "baseline": "labels",
    "types": "baseline",
    "stage-model": "types",
    "behavior-model": "stage-model",
    "evaluate": "behavior-model",
    "predict": "evaluate",
    "route-ko": "predict",
}
_MESSAGE_KO: dict[RegistryStage, str] = {
    "labels": "라벨 검증 및 등록 완료",
    "baseline": "전역·개인 기준선 준비 완료",
    "types": "표준 타입과 분포 이탈 상태 생성 완료",
    "stage-model": "사건·5단계 모델 학습 완료",
    "behavior-model": "행동 다중 라벨 모델 학습 완료",
    "evaluate": "검증·잠긴 테스트·노이즈 평가 완료",
    "predict": "예측 감사 데이터 생성 완료",
    "route-ko": "한국어 결과 라우팅 완료",
}


@dataclass(frozen=True, slots=True)
class RegistryRunResult:
    run_id: str
    series_id: str
    final_receipt: Path
    release_id: str
    status: Literal["oracle/sanity"]


def _write_manifest(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=False)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _existing_or_create_receipt(
    *,
    run_id: str,
    stage: RegistryStage,
    artifact: Path,
    output_receipt: Path,
    versions: dict[str, str],
    reused: bool,
) -> StageReceipt:
    if output_receipt.exists():
        existing = validate_stage_receipt(output_receipt)
        if (
            existing.pipeline_run_id != run_id
            or existing.stage_id != stage
            or Path(existing.artifact_uri).resolve() != artifact.resolve()
            or existing.artifact_sha256 != sha256_file(artifact)
            or existing.versions != versions
        ):
            raise ValueError(f"existing receipt conflicts with requested stage: {output_receipt}")
        return existing
    return create_stage_receipt(
        pipeline_run_id=run_id,
        stage_id=stage,
        artifact_uri=artifact,
        output_path=output_receipt,
        status="REUSED" if reused else "SUCCESS",
        versions=versions,
        message_ko=(
            f"{_MESSAGE_KO[stage]} (기존 산출물 재사용)"
            if reused
            else _MESSAGE_KO[stage]
        ),
    )


def _validate_stage_input(
    stage: RegistryStage,
    input_receipt: Path,
) -> StageReceipt:
    receipt = validate_stage_receipt(input_receipt)
    if receipt.status not in {"SUCCESS", "REUSED"}:
        raise ValueError(f"upstream stage is not successful: {receipt.status}")
    expected = _EXPECTED_INPUT_STAGE[stage]
    if receipt.stage_id != expected:
        raise ValueError(
            f"{stage} requires {expected} receipt, got {receipt.stage_id}"
        )
    return receipt


def run_registry_stage(
    config: TrainingRegistryConfig,
    *,
    stage: RegistryStage,
    run_id: str,
    input_receipt: Path,
    output_receipt: Path,
) -> StageReceipt:
    """Run one hash-validated stage and connect its receipt to the next stage."""

    upstream = _validate_stage_input(stage, input_receipt)
    series_id = upstream.versions.get("dataset_version", config.series_id)
    if series_id != config.series_id:
        raise ValueError(
            f"receipt dataset {series_id!r} does not match config {config.series_id!r}"
        )
    series_root = config.artifact_root / series_id
    prepared_root = config.data_root / "prepared" / series_id
    outcome_root = config.outcome_root / series_id
    stage_root = series_root / stage
    versions = dict(upstream.versions)
    reused = False

    if stage == "labels":
        outcome_manifest = outcome_root / "manifest.json"
        payload = json.loads(outcome_manifest.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "goal1.5/behavior-outcomes/v1":
            raise ValueError("unsupported behavior outcome manifest")
        if payload.get("status") != "oracle/sanity":
            raise ValueError("synthetic label stage must remain oracle/sanity")
        artifact = stage_root / "manifest.json"
        if artifact.exists():
            reused = True
        else:
            artifact = _write_manifest(
                artifact,
                {
                    "schema_version": "goal1.5/registered-label-set/v1",
                    "status": "oracle/sanity",
                    "real_data_status": "NOT VERIFIED",
                    "source_manifest": str(outcome_manifest.resolve()),
                    "source_manifest_sha256": sha256_file(outcome_manifest),
                },
            )
        versions["dataset_version"] = series_id
        versions["label_version"] = f"labels-{sha256_file(artifact)[:24]}"
    elif stage == "baseline":
        artifact = prepared_root / "manifest.json"
        if artifact.exists():
            reused = True
        else:
            prepared = prepare_series(
                config.data_root / "registry" / series_id,
                prepared_root,
            )
            artifact = prepared.manifest_json
        versions["baseline_version"] = f"baseline-{sha256_file(artifact)[:24]}"
    elif stage == "types":
        artifact = stage_root / "manifest.json"
        if artifact.exists():
            reused = True
        else:
            type_result = fit_standard_type_artifacts(
                prepared_root,
                stage_root,
                random_state=config.random_state,
            )
            artifact = type_result.manifest_json
        versions["standard_type_version"] = f"types-{sha256_file(artifact)[:24]}"
    elif stage == "stage-model":
        artifact = stage_root / "manifest.json"
        if artifact.exists():
            reused = True
        else:
            stage_result = fit_stage_model_artifacts(
                prepared_root,
                outcome_root,
                series_root / "types",
                stage_root,
                random_state=config.random_state,
            )
            artifact = stage_result.manifest_json
        versions["stage_model_version"] = f"stage-{sha256_file(artifact)[:24]}"
    elif stage == "behavior-model":
        artifact = stage_root / "manifest.json"
        if artifact.exists():
            reused = True
        else:
            behavior_result = fit_behavior_model_artifacts(
                prepared_root,
                outcome_root,
                series_root / "types",
                series_root / "stage-model",
                stage_root,
                random_state=config.random_state,
            )
            artifact = behavior_result.manifest_json
        versions["behavior_model_version"] = f"behavior-{sha256_file(artifact)[:24]}"
    elif stage == "evaluate":
        artifact = stage_root / "manifest.json"
        registry = ModelRegistry(config.registry_path)
        registry.initialize()
        model_sha = sha256_file(series_root / "stage-model" / "manifest.json")
        dataset_sha = sha256_file(prepared_root / "manifest.json")
        if artifact.exists():
            reused = True
            if not registry.locked_test_exists(
                model_sha256=model_sha,
                dataset_sha256=dataset_sha,
            ):
                registry.record_locked_test(
                    model_sha256=model_sha,
                    dataset_sha256=dataset_sha,
                    audit_reason=f"Adopt existing selected evaluation for {run_id}",
                )
        else:
            if registry.locked_test_exists(
                model_sha256=model_sha,
                dataset_sha256=dataset_sha,
            ):
                raise ValueError(
                    "locked_test was already consumed for this model/dataset pair"
                )
            evaluation_result = evaluate_hierarchical_artifacts(
                prepared_root,
                outcome_root,
                series_root / "types",
                series_root / "stage-model",
                series_root / "behavior-model",
                stage_root,
                random_state=config.random_state,
            )
            artifact = evaluation_result.manifest_json
            registry.record_locked_test(
                model_sha256=model_sha,
                dataset_sha256=dataset_sha,
                audit_reason=f"Selected validation champion for {run_id}",
            )
        versions["evaluation_version"] = f"evaluation-{sha256_file(artifact)[:24]}"
    elif stage == "predict":
        artifact = stage_root / "manifest.json"
        if artifact.exists():
            reused = True
        else:
            prediction_result = create_prediction_artifacts(
                series_root / "behavior-model",
                series_root / "evaluate",
                stage_root,
            )
            artifact = prediction_result.manifest_json
        versions["prediction_version"] = f"prediction-{sha256_file(artifact)[:24]}"
    elif stage == "route-ko":
        artifact = stage_root / "manifest.json"
        if artifact.exists():
            reused = True
        else:
            router_result = create_korean_router_artifacts(
                series_root / "predict",
                stage_root,
                router_version=config.korean_router_version,
            )
            artifact = router_result.manifest_json
        versions["router_version"] = f"router-{sha256_file(artifact)[:24]}"
    else:
        raise AssertionError(f"unhandled registry stage: {stage}")

    receipt = _existing_or_create_receipt(
        run_id=run_id,
        stage=stage,
        artifact=artifact,
        output_receipt=output_receipt,
        versions=versions,
        reused=reused,
    )
    registry = ModelRegistry(config.registry_path)
    registry.initialize()
    registry.record_stage_receipt(output_receipt)
    return receipt


def _register_release(
    config: TrainingRegistryConfig,
    final_receipt: StageReceipt,
) -> str:
    registry = ModelRegistry(config.registry_path)
    registry.initialize()
    series_root = config.artifact_root / config.series_id
    artifacts = {
        "dataset": config.data_root / "registry" / config.series_id / "manifest.json",
        "label_set": series_root / "labels" / "manifest.json",
        "baseline": config.data_root / "prepared" / config.series_id / "manifest.json",
        "standard_type": series_root / "types" / "manifest.json",
        "stage_model": series_root / "stage-model" / "manifest.json",
        "behavior_model": series_root / "behavior-model" / "manifest.json",
        "router": series_root / "route-ko" / "manifest.json",
    }
    version_ids: dict[str, str] = {}
    for kind, artifact in artifacts.items():
        digest = sha256_file(artifact)
        version_id = f"{kind}-{config.series_id}-{digest[:12]}"
        registry.register_version(
            kind=kind,
            version_id=version_id,
            artifact_uri=str(artifact.resolve()),
            artifact_sha256=digest,
            metadata={
                "source_domain": "synthetic_truth_oracle",
                "status": "oracle/sanity",
                "real_data_status": "NOT VERIFIED",
                "pipeline_run_id": final_receipt.pipeline_run_id,
            },
        )
        version_ids[kind] = version_id
    for kind, stage in (("evaluation", "evaluate"), ("prediction", "predict")):
        artifact = series_root / stage / "manifest.json"
        digest = sha256_file(artifact)
        registry.register_version(
            kind=kind,
            version_id=f"{kind}-{config.series_id}-{digest[:12]}",
            artifact_uri=str(artifact.resolve()),
            artifact_sha256=digest,
            metadata={
                "source_domain": "synthetic_truth_oracle",
                "status": "oracle/sanity",
                "real_data_status": "NOT VERIFIED",
                "pipeline_run_id": final_receipt.pipeline_run_id,
            },
        )
    release_material = json.dumps(version_ids, sort_keys=True).encode()
    release_id = (
        f"goal15-{config.series_id}-"
        f"{hashlib.sha256(release_material).hexdigest()[:12]}"
    )
    try:
        registry.release_status(release_id)
    except ValueError:
        registry.create_release(
            release_id=release_id,
            versions=version_ids,
            source_domain="synthetic_truth_oracle",
        )
    return release_id


def run_registry_all(config: TrainingRegistryConfig) -> RegistryRunResult:
    input_receipt = config.input_receipt
    receipts_root = config.artifact_root / config.series_id / "receipts"
    current = input_receipt
    final: StageReceipt | None = None
    for stage in STAGES:
        output = receipts_root / f"{stage}.json"
        final = run_registry_stage(
            config,
            stage=stage,
            run_id=config.run_id,
            input_receipt=current,
            output_receipt=output,
        )
        current = output
    if final is None:
        raise RuntimeError("registry workflow did not execute any stage")
    release_id = _register_release(config, final)
    return RegistryRunResult(
        run_id=config.run_id,
        series_id=config.series_id,
        final_receipt=current,
        release_id=release_id,
        status="oracle/sanity",
    )


def export_registry_knime_tables(
    config: TrainingRegistryConfig,
    *,
    run_id: str,
) -> Path:
    series_root = config.artifact_root / config.series_id
    output = series_root / "knime-export"
    if output.exists():
        return output
    output.mkdir(parents=True)
    tables = {
        "stage_metrics.parquet": series_root
        / "stage-model"
        / "validation_metrics.parquet",
        "behavior_metrics.parquet": series_root
        / "behavior-model"
        / "validation_metrics.parquet",
        "stress_metrics.parquet": series_root / "evaluate" / "stress_metrics.parquet",
        "korean_results.parquet": series_root
        / "route-ko"
        / "korean_results.parquet",
        "standard_type_diagnostics.parquet": series_root
        / "types"
        / "standard_type_diagnostics.parquet",
        "personal_calibration_status.parquet": series_root
        / "behavior-model"
        / "personal_calibration_status.parquet",
    }
    export_rows: list[dict[str, object]] = []
    for name, source in tables.items():
        frame = pq.read_table(source).to_pandas()
        pq.write_table(
            pa.Table.from_pandas(frame, preserve_index=False),
            output / name,
            compression="zstd",
        )
        export_rows.append(
            {
                "table": name,
                "rows": len(frame),
                "sha256": sha256_file(output / name),
            }
        )
    pd.DataFrame(export_rows).to_csv(output / "manifest.csv", index=False)
    (output / "status.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "series_id": config.series_id,
                "status": "oracle/sanity",
                "real_data_status": "NOT VERIFIED",
                "synchronization_status": "NOT_AVAILABLE_TRUTH_ONLY",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output
